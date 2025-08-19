import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any
import logging
from config.llm_config import llm
from langchain.schema import HumanMessage, SystemMessage

# ---- LLM token logging (same pattern as manage_showings_agent) ----
from utils.logger import init_log_file, log_token_usage
from langchain_community.callbacks.manager import get_openai_callback
LOG_FILE = "logs/showing_llm_tokens.csv"
init_log_file(LOG_FILE)


class LearningModule:
    """
    Core learning module that extracts property insights from feedback
    to enhance showing decision prompts
    """
    
    def __init__(self):
        self.setup_logging()
        self.insights_file = "data/property_insights.json"
    
    def setup_logging(self):
        """Setup logging for learning module"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_existing_insights(self) -> Dict[str, str]:
        """Load existing property insights from JSON file"""
        if os.path.exists(self.insights_file):
            try:
                with open(self.insights_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load insights: {e}, starting fresh")
        return {}
    
    def save_insights(self, insights: Dict[str, str]):
        """Save property insights to JSON file"""
        os.makedirs(os.path.dirname(self.insights_file), exist_ok=True)
        
        try:
            with open(self.insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
            self.logger.info(f"Property insights saved to {self.insights_file}")
        except Exception as e:
            self.logger.error(f"Could not save insights: {e}")
    
    def extract_property_insights(self, feedback_df: pd.DataFrame) -> Dict[str, str]:
        """Extract concise insights per property from feedback using LLM"""
        
        # Load existing insights to avoid regenerating
        existing_insights = self.load_existing_insights()
        new_insights = existing_insights.copy()
        
        # Group feedback by owner_id (each owner has one property)
        owner_feedback = feedback_df.groupby('owner_id')
        
        for owner_id, owner_data in owner_feedback:
            if owner_id in existing_insights:
                continue  # Skip already processed properties
                
            if len(owner_data) < 1:  # Need at least 1 feedback entry
                continue
                
            insight = self.generate_property_insight(owner_id, owner_data)
            if insight:
                new_insights[owner_id] = insight
        
        return new_insights
    
    def generate_property_insight(self, owner_id: str, owner_data: pd.DataFrame) -> str:
        """Generate a concise one-line insight for a property using LLM"""
        
        # Prepare minimal input for budget efficiency
        feedback_summary = {
            "positive_count": len(owner_data[owner_data['feedback_type'] == 'positive']),
            "negative_count": len(owner_data[owner_data['feedback_type'] == 'negative']),
            "avg_score": round(owner_data['feedback_score'].mean(), 1),
            "outcomes": owner_data['showing_outcome'].value_counts().to_dict(),
            "key_notes": owner_data['realtor_notes'].head(3).tolist()
        }
        
        system_prompt = "Generate ONE concise line (max 15 words) about how to improve this property's appeal based on feedback. Focus on actionable insights."
        
        user_prompt = f"Property feedback: {feedback_summary['positive_count']} positive, {feedback_summary['negative_count']} negative, avg {feedback_summary['avg_score']}/5. Notes: {'; '.join(feedback_summary['key_notes'][:2])}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            with get_openai_callback() as cb:
                resp = llm.invoke(messages)
                log_token_usage(LOG_FILE, cb, user_prompt)
            
            insight = resp.content.strip()
            # Ensure it's truly one line and concise
            if len(insight) > 100:
                insight = insight[:97] + "..."
            
            return insight
        except Exception as e:
            self.logger.warning(f"Failed to generate insight for {owner_id}: {e}")
            return None
    
    
    
    
    
    
    
    
    
    
    
    def generate_insights_report(self, feedback_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate property insights from feedback data"""
        
        # Extract property insights using LLM
        insights = self.extract_property_insights(feedback_df)
        
        # Save insights to JSON file
        self.save_insights(insights)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": {
                "total_entries": len(feedback_df),
                "unique_properties": len(feedback_df['owner_id'].unique()),
                "avg_score": feedback_df['feedback_score'].mean(),
                "success_rate": len(feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured'])]) / len(feedback_df)
            },
            "insights_generated": {
                "total_insights": len(insights),
                "new_insights": len(insights) - len(self.load_existing_insights()),
                "sample_insights": dict(list(insights.items())[:3])
            },
            "insights_file": self.insights_file
        }
        
        # Save report
        os.makedirs("data/learning_reports", exist_ok=True)
        report_file = f"data/learning_reports/insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Insights report saved to {report_file}")
        
        return report
    

def main():
    """Main function to run property insights extraction"""
    learning_module = LearningModule()
    
    # Load feedback data
    feedback_file = "data/synthetic/realtor_feedback.csv"
    if not os.path.exists(feedback_file):
        print(f"Feedback file not found: {feedback_file}")
        return
    
    feedback_df = pd.read_csv(feedback_file)
    print(f"Loaded {len(feedback_df)} feedback entries for insights extraction")
    
    # Generate insights report
    report = learning_module.generate_insights_report(feedback_df)
    
    print("\n=== PROPERTY INSIGHTS REPORT ===")
    print(f"Average feedback score: {report['feedback_summary']['avg_score']:.2f}")
    print(f"Success rate: {report['feedback_summary']['success_rate']:.1%}")
    print(f"Unique properties: {report['feedback_summary']['unique_properties']}")
    
    print(f"\nInsights generated: {report['insights_generated']['total_insights']}")
    print(f"New insights: {report['insights_generated']['new_insights']}")
    
    print("\n=== SAMPLE INSIGHTS ===")
    for owner_id, insight in report['insights_generated']['sample_insights'].items():
        print(f"- {owner_id}: {insight}")
    
    print(f"\nInsights saved to: {report['insights_file']}")
    print(f"Detailed report saved to data/learning_reports/")

if __name__ == "__main__":
    main()