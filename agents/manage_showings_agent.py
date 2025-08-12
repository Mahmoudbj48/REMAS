# agents/manage_showings_agent_llm.py
import json
import math
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import time  # if not already imported


from qdrant_client.models import Record
from qdrant_client.http import exceptions as qexc

from utils.qdrant_connection import get_matches_by_owner, _retrieve_payload, _fetch_user_payloads, _iter_owner_ids
from config.llm_config import llm

OWNER_COLLECTION = "sampled_owner_agent_listings"
USER_COLLECTION  = "user_agent_listings"
MAX_INVITES      = 10
MIN_CANDIDATES   = 3
QUALITY_GATE     = 0.45  # mean(top5) threshold

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


# ======================================================
# LLM decision (Function 1)
# ======================================================

_DECISION_SYSTEM_PROMPT = """
You are the Manage Showings decision agent for a real-estate matching system.
Decide whether to schedule a showing TODAY for the given listing and how many users to invite.

Consider:
- Match quality (distribution, not only max), candidate depth, basic fit (price/bedrooms/availability).
- Only invite at most the number of provided candidates (max 10).
- Prefer showings when top matches are strong and there is enough candidate depth.

Hard rules:
- Output STRICT JSON only, with no commentary.
- Schema: {"show":"0|1","num":"0|1|2|3|4|5|6|7|8|9|10"}
- If "show" == "1", then "num" >= "1".
- "num" ≤ number of provided candidates and ≤ 10.

Return examples:
{"show":"1","num":"6"}
{"show":"0","num":"0"}
""".strip()

_JSON_RE = re.compile(r'\{.*\}', re.S)

def _parse_json_strict(raw: str) -> Optional[dict]:
    """Parse the first JSON object in raw string; tolerate small deviations."""
    if not raw:
        return None
    # First try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Then try extracting the first {...}
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
      { "user_id": str, "score": float, "payload": {... trimmed user payload ...} }
    Returns JSON-compatible dict: {"show": "0|1", "num": "0..10"}
    """
    # ---------- Cheap pre-gates (avoid LLM spend for obvious NO) ----------
    scores = [float(m.get("score", 0.0)) for m in matched_users]
    mean5 = _mean_topk(scores, 5)
    if len(matched_users) < MIN_CANDIDATES or mean5 < QUALITY_GATE:
        return {"show": "0", "num": "0"}

    # ---------- Build compact LLM input ----------
    owner_obj = {"owner_id": owner_id, "owner": _summ(owner_payload)}
    users_obj = [{
        "user_id": m["user_id"],
        "score": round(float(m.get("score", 0.0)), 4),
        "user": _summ(m.get("payload", {}))
    } for m in matched_users[:MAX_INVITES]]

    user_prompt = {
        "owner": owner_obj,
        "candidates": users_obj,
        "today_iso": datetime.utcnow().date().isoformat(),
        "stats": {
            "candidate_count": len(matched_users),
            "mean_top5": round(mean5, 4)
        }
    }

    messages = [
        {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
    ]

    # ---------- LLM call ----------
    try:
        resp = llm_client.invoke(messages)  # keep your wrapper
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

            # Get top-k matches (user IDs + scores)
            matches = get_matches_by_owner(oid, top_k=top_k)  # [{owner_id, user_id, score, ...}]
            user_ids = [str(m["user_id"]) for m in matches]
            user_payloads = _fetch_user_payloads(user_ids)

            matched_users = []
            for m in matches:
                uid = str(m["user_id"])
                try:
                    score = float(m.get("score", 0.0))
                except Exception:
                    score = 0.0
                matched_users.append({
                    "user_id": uid,
                    "score": score,
                    "payload": user_payloads.get(uid, {})
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
                "sample": [{"user_id": u["user_id"], "score": round(u["score"], 4)} for u in matched_users[:3]]
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
