import json
import math
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import time  # if not already imported

from qdrant_client.http import exceptions as qexc

from utils.qdrant_connection import (
    get_matches_by_owner,
    _retrieve_payload,
    _fetch_user_payloads,
    _iter_owner_ids,
    get_owner_profile,
    get_user_profile, 
)
from config.llm_config import llm

# ---- LLM token logging (same pattern as your parser) ----
from utils.logger import init_log_file, log_token_usage
from langchain_community.callbacks.manager import get_openai_callback
LOG_FILE = "logs/showing_llm_tokens.csv"
init_log_file(LOG_FILE)

OWNER_COLLECTION = "sampled_owner_agent_listings2"
USER_COLLECTION  = "user_agent_listings"

MAX_INVITES      = 10
MIN_CANDIDATES   = 3
QUALITY_GATE     = 0.45   # mean(top5) threshold

# Fairness/starvation knobs
STARVATION_DAYS  = 21     # owner application is older than this
STARVATION_SHOWS = 0      # and owner number_of_shows <= this

# ======================================================
# Small helpers
# ======================================================

def _summ(owner_or_user_payload: dict) -> dict:
    """Trim payload to essentials for LLM."""
    if not owner_or_user_payload:
        return {}
    return {
        "state": owner_or_user_payload.get("state"),
        "price": owner_or_user_payload.get("price"),
        "bedrooms": owner_or_user_payload.get("bedrooms"),
        "available_from": owner_or_user_payload.get("available_from"),
        "soft_attributes": (owner_or_user_payload.get("soft_attributes") or "")[:300],
    }

def _mean_topk(values: List[float], k: int = 5) -> float:
    """Mean of top-k non-NaN numeric values (descending)."""
    nums = []
    for v in values:
        try:
            f = float(v)
            if not math.isnan(f):
                nums.append(f)
        except Exception:
            continue
    if not nums:
        return 0.0
    nums.sort(reverse=True)
    top = nums[:min(k, len(nums))]
    return sum(top) / len(top)

def _parse_iso_date(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        # Expect "YYYY-MM-DD" (or ISO starting with that)
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _days_since(d: Optional[datetime]) -> Optional[int]:
    if not d:
        return None
    now = datetime.now(timezone.utc)
    delta = now - d
    return max(0, delta.days)

# ======================================================
# LLM decision (Function 1)
# ======================================================

_DECISION_SYSTEM_PROMPT = """
You are the Manage Showings decision agent for a real-estate matching system.
Decide whether to schedule a showing TODAY for the given listing and how many users to invite.

Consider and balance:
- Match quality: use the provided scores and their distribution (not just the max).
- Candidate depth: avoid scheduling if there are too few viable candidates.
- Basic fit signals (price/bedrooms/availability) have been pre-filtered already.
- Fairness & starvation prevention:
  • If the listing has waited a long time with few or zero shows, lower the threshold and invite a reasonable count even if scores are modest.
  • Prefer candidates with fewer prior show opportunities ("number_of_shows" smaller).
  • Consider application recency: very new applicants can wait if others had no opportunities.

Hard rules:
- Output STRICT JSON only, no commentary.
- Schema: {"show":"0|1","num":"0|1|2|3|4|5|6|7|8|9|10"}
- If "show" == "1", then "num" >= "1".
- "num" must be ≤ number of provided candidates and ≤ 10.

Typical strategies:
- Good quality & sufficient depth → invite 3–6.
- Borderline quality but owner starvation (old application_date & low number_of_shows) → invite 2–4; be fair to low-opportunity candidates.
- Very weak quality & shallow depth without starvation → do not schedule.

Return examples:
{"show":"1","num":"6"}
{"show":"0","num":"0"}
""".strip()

_JSON_RE = re.compile(r'\{.*\}', re.S)

def _parse_json_strict(raw: str) -> Optional[dict]:
    """Parse the first JSON object in raw string; tolerate small deviations."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def decide_showing_for_owner(
    owner_id: str,
    owner_payload: dict,
    matched_users: List[Dict[str, Any]],
    llm_client = llm
) -> Dict[str, str]:
    """
    matched_users: list of dicts with at least:
      {
        "user_id": str,
        "score": float,
        "payload": {... trimmed user payload ...},
        "user_fairness": {"application_date": str|None, "number_of_shows": int|str|None, "days_since_application": int|None}
      }
    Returns JSON-compatible dict: {"show": "0|1", "num": "0..10"}
    """
    # ---------- Cheap pre-gates (avoid LLM spend for obvious NO) ----------
    scores = [float(m.get("score", 0.0)) for m in matched_users]
    mean5 = _mean_topk(scores, 5)

    # Owner fairness (already attached by caller in owner_payload_fairness)
    owner_fair = owner_payload.get("_owner_fairness") or {}
    owner_days_since_app = owner_fair.get("days_since_application")
    try:
        owner_shows = int(owner_fair.get("number_of_shows", 0)) if owner_fair else 0
    except Exception:
        owner_shows = 0

    starvation_override = (
        owner_days_since_app is not None
        and owner_days_since_app >= STARVATION_DAYS
        and owner_shows <= STARVATION_SHOWS
    )

    if not starvation_override:
        if len(matched_users) < MIN_CANDIDATES or mean5 < QUALITY_GATE:
            return {"show": "0", "num": "0"}

    # ---------- Build compact LLM input ----------
    owner_obj = {
        "owner_id": owner_id,
        "owner": _summ(owner_payload),
        "owner_fairness": owner_fair  # includes application_date, number_of_shows, days_since_application
    }

    users_obj = []
    for m in matched_users[:MAX_INVITES]:
        uf = m.get("user_fairness") or {}
        users_obj.append({
            "user_id": m["user_id"],
            "score": round(float(m.get("score", 0.0)), 4),
            "user": _summ(m.get("payload", {})),
            "user_fairness": uf
        })

    user_prompt = {
        "owner": owner_obj,
        "candidates": users_obj,
        "today_iso": datetime.utcnow().date().isoformat(),
        "stats": {
            "candidate_count": len(matched_users),
            "mean_top5": round(mean5, 4),
            "starvation_override": starvation_override
        }
    }

    messages = [
        {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
    ]

    # ---------- LLM call w/ token usage logging ----------
    try:
        # We log the exact user message we sent (the JSON prompt).
        user_input_for_log = messages[-1]["content"]
        with get_openai_callback() as cb:
            resp = llm_client.invoke(messages)  # keep your wrapper
            log_token_usage(LOG_FILE, cb, user_input_for_log)
        raw = getattr(resp, "content", "").strip()
    except Exception:
        return {"show": "0", "num": "0"}

    data = _parse_json_strict(raw) or {"show": "0", "num": "0"}

    # ---------- Enforce contract ----------
    show = str(data.get("show", "0")).strip()
    try:
        n = int(str(data.get("num", "0")).strip())
    except Exception:
        n = 0

    n_candidates = len(users_obj)
    if show not in ("0", "1"):
        show = "0"

    n = max(0, min(n, n_candidates, MAX_INVITES))

    if show == "1" and n == 0:
        n = 1 if n_candidates >= 1 else 0
        if n == 0:
            show = "0"

    return {"show": show, "num": str(n)}

# ======================================================
# Daily loop (Function 2)
# ======================================================

def daily_llm_showing_decisions(top_k: int = 10, show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Loop all owner listings (from OWNER_COLLECTION):
      - retrieve owner payload
      - get top-k matches from similarity_collection
      - join user payloads
      - attach owner/user fairness (application_date, number_of_shows, days_since_application)
      - call LLM decision (with pre-gates)
    Returns a list of decision dicts, one per owner.
    """
    results: List[Dict[str, Any]] = []

    # Materialize owner IDs so tqdm can display total progress
    owner_ids = list(_iter_owner_ids(owner_collection=OWNER_COLLECTION))
    total = len(owner_ids)
    iterator = tqdm(owner_ids, total=total, desc="ManageShowings", unit="owner") if show_progress else owner_ids

    for oid in iterator:
        try:
            owner_payload = _retrieve_payload(OWNER_COLLECTION, oid)
            if not owner_payload:
                results.append({"owner_id": oid, "error": "no_owner_payload"})
                if show_progress:
                    iterator.set_postfix_str("error=no_payload")
                continue

            # ---- Owner profile for fairness ----
            op = get_owner_profile(oid) or {}
            o_app_dt = _parse_iso_date(op.get("application_date"))
            owner_payload["_owner_fairness"] = {
                "application_date": op.get("application_date"),
                "number_of_shows": op.get("number_of_shows", 0),
                "days_since_application": _days_since(o_app_dt),
            }

            # ---- Get top-k matches (user IDs + scores) ----
            matches = get_matches_by_owner(oid, top_k=top_k)  # [{owner_id, user_id, score, ...}]
            user_ids = [str(m["user_id"]) for m in matches]
            user_payloads = _fetch_user_payloads(user_ids) or {}

            matched_users = []
            for m in matches:
                uid = str(m["user_id"])
                try:
                    score = float(m.get("score", 0.0))
                except Exception:
                    score = 0.0

                # per-user fairness annotation
                up = get_user_profile(uid) or {}  # singular helper you already have
                u_app_dt = _parse_iso_date(up.get("application_date"))
                user_fair = {
                    "application_date": up.get("application_date"),
                    "number_of_shows": up.get("number_of_shows", 0),
                    "days_since_application": _days_since(u_app_dt),
                }

                matched_users.append({
                    "user_id": uid,
                    "score": score,
                    "payload": user_payloads.get(uid, {}) or {},
                    "user_fairness": user_fair
                })

            # Precompute mean_top5 for logging (same as in decide)
            mean5 = _mean_topk([mu["score"] for mu in matched_users], 5)

            decision = decide_showing_for_owner(
                owner_id=oid,
                owner_payload=owner_payload,
                matched_users=matched_users,
                llm_client=llm
            )

            results.append({
                "owner_id": oid,
                "decision": decision,
                "considered": len(matched_users),
                "mean_top5": round(mean5, 4),
                "sample": [{"user_id": u["user_id"], "score": round(u["score"], 4)} for u in matched_users[:3]],
                "owner_profile": owner_payload.get("_owner_fairness")
            })

            if show_progress:
                iterator.set_postfix_str(
                    f"cand={len(matched_users)} mean5={mean5:.2f} show={decision.get('show')} num={decision.get('num')}"
                )

        except qexc.UnexpectedResponse as e:
            results.append({"owner_id": oid, "error_type": "qdrant_unexpected_response", "error": str(e)})
            if show_progress:
                iterator.set_postfix_str("error=qdrant")
        except Exception as e:
            results.append({"owner_id": oid, "error_type": "generic_exception", "error": str(e)})
            if show_progress:
                iterator.set_postfix_str("error=generic")

    return results
