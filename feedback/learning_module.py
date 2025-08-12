import pandas as pd
import numpy as np
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from config.llm_config import llm
from langchain.schema import HumanMessage, SystemMessage

@dataclass
class ParameterUpdate:
    """Represents a parameter update recommendation"""
    parameter_name: str
    current_value: float
    suggested_value: float
    confidence: float
    reasoning: str
    feedback_count: int
    timestamp: str

@dataclass
class LearningConfig:
    """Configuration for learning algorithms"""
    learning_rate: float = 0.1
    min_feedback_count: int = 5
    confidence_threshold: float = 0.7
    max_parameter_change: float = 0.2
    rollback_threshold: float = 0.3

class LearningModule:
    """
    Core learning module that analyzes feedback patterns and generates
    parameter adjustment recommendations
    """
    
    def __init__(self, config: LearningConfig = None):
        self.config = config or LearningConfig()
        self.current_parameters = self.load_current_parameters()
        self.parameter_history = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for learning module"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_current_parameters(self) -> Dict[str, float]:
        """Load current matching parameters"""
        default_params = {
            "location_weight": 0.3,
            "price_weight": 0.4,
            "soft_attributes_weight": 0.2,
            "amenities_weight": 0.1,
            "quality_gate_threshold": 0.45,
            "min_candidates": 3,
            "max_invites": 10,
            "price_tolerance": 0.15,
            "bedroom_strict_matching": 0.8
        }
        
        params_file = "data/parameters/current_weights.json"
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    loaded_params = json.load(f)
                    default_params.update(loaded_params)
            except Exception as e:
                self.logger.warning(f"Could not load parameters: {e}, using defaults")
        
        return default_params
    
    def save_parameter_recommendations(self, updates: List[ParameterUpdate], reason: str = ""):
        """Save parameter change recommendations without applying them"""
        os.makedirs("data/parameter_recommendations", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create recommendations entry
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "current_parameters": self.current_parameters.copy(),
            "recommended_changes": [asdict(update) for update in updates],
            "summary": {
                "total_recommendations": len(updates),
                "high_confidence": len([u for u in updates if u.confidence >= 0.7]),
                "parameter_types": list(set(u.parameter_name for u in updates))
            }
        }
        
        # Save as CSV only
        csv_file = f"data/parameter_recommendations/recommendations_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'parameter_name', 'current_value', 'suggested_value', 'change_amount',
                'confidence', 'reasoning', 'feedback_count', 'timestamp'
            ])
            
            for update in updates:
                change_amount = update.suggested_value - update.current_value
                writer.writerow([
                    update.parameter_name,
                    update.current_value,
                    update.suggested_value,
                    round(change_amount, 4),
                    round(update.confidence, 3),
                    update.reasoning,
                    update.feedback_count,
                    update.timestamp
                ])
        
        self.logger.info(f"Parameter recommendations saved to {csv_file}")
        return csv_file
    
    def analyze_weight_adjustments(self, feedback_df: pd.DataFrame) -> List[ParameterUpdate]:
        """Analyze feedback to suggest weight adjustments"""
        
        updates = []
        
        # Group feedback by parameter focus areas
        parameter_feedback = self.group_feedback_by_parameters(feedback_df)
        
        for param_area, feedback_group in parameter_feedback.items():
            if len(feedback_group) < self.config.min_feedback_count:
                continue
            
            # Calculate adjustment based on feedback patterns
            adjustment = self.calculate_weight_adjustment(param_area, feedback_group)
            
            if adjustment and abs(adjustment['change']) > 0.01:  # Minimum meaningful change
                current_val = self.current_parameters.get(adjustment['parameter'], 0.0)
                new_val = max(0.0, min(1.0, current_val + adjustment['change']))
                
                update = ParameterUpdate(
                    parameter_name=adjustment['parameter'],
                    current_value=current_val,
                    suggested_value=new_val,
                    confidence=adjustment['confidence'],
                    reasoning=adjustment['reasoning'],
                    feedback_count=len(feedback_group),
                    timestamp=datetime.now().isoformat()
                )
                updates.append(update)
        
        return updates
    
    def group_feedback_by_parameters(self, feedback_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group feedback entries by the parameter they most relate to"""
        
        parameter_groups = {
            "location": pd.DataFrame(),
            "price": pd.DataFrame(),
            "soft_attributes": pd.DataFrame(),
            "amenities": pd.DataFrame(),
            "quality_threshold": pd.DataFrame()
        }
        
        for _, row in feedback_df.iterrows():
            notes = row['realtor_notes'].lower()
            
            # Classify feedback by primary concern
            if any(word in notes for word in ['location', 'commute', 'neighborhood', 'area']):
                parameter_groups["location"] = pd.concat([parameter_groups["location"], row.to_frame().T])
            elif any(word in notes for word in ['price', 'budget', 'expensive', 'cheap', 'cost']):
                parameter_groups["price"] = pd.concat([parameter_groups["price"], row.to_frame().T])
            elif any(word in notes for word in ['lifestyle', 'quiet', 'pet', 'smoking', 'vibe']):
                parameter_groups["soft_attributes"] = pd.concat([parameter_groups["soft_attributes"], row.to_frame().T])
            elif any(word in notes for word in ['parking', 'furnished', 'laundry', 'amenities']):
                parameter_groups["amenities"] = pd.concat([parameter_groups["amenities"], row.to_frame().T])
            elif row['match_quality_rating'] <= 2:
                parameter_groups["quality_threshold"] = pd.concat([parameter_groups["quality_threshold"], row.to_frame().T])
        
        return parameter_groups
    
    def calculate_weight_adjustment(self, param_area: str, feedback_group: pd.DataFrame) -> Dict[str, Any]:
        """Calculate specific weight adjustment for a parameter area using LLM reasoning"""
        
        if feedback_group.empty:
            return None
        
        # Calculate statistical metrics (foundation)
        avg_score = feedback_group['feedback_score'].mean()
        positive_ratio = len(feedback_group[feedback_group['feedback_type'] == 'positive']) / len(feedback_group)
        success_ratio = len(feedback_group[feedback_group['showing_outcome'].isin(['leased', 'toured'])]) / len(feedback_group)
        
        # Gather feedback context for LLM
        sample_feedback = feedback_group['realtor_notes'].head(5).tolist()
        outcomes = feedback_group['showing_outcome'].value_counts().to_dict()
        
        # Use LLM for reasoning about parameter adjustments
        llm_analysis = self.get_llm_parameter_reasoning(
            param_area, sample_feedback, avg_score, positive_ratio, success_ratio, outcomes
        )
        
        # Combine statistical analysis with LLM reasoning
        statistical_change = self.calculate_statistical_change(avg_score, positive_ratio, success_ratio)
        
        # LLM can modify the statistical recommendation
        final_change = self.combine_statistical_and_llm_analysis(statistical_change, llm_analysis)
        
        # Map parameter area to actual parameter name
        param_mapping = {
            "location": "location_weight",
            "price": "price_weight", 
            "soft_attributes": "soft_attributes_weight",
            "amenities": "amenities_weight",
            "quality_threshold": "quality_gate_threshold"
        }
        
        parameter_name = param_mapping.get(param_area, param_area)
        
        # Enhanced confidence calculation
        statistical_confidence = min(1.0, len(feedback_group) / 10.0) * (1.0 - np.std(feedback_group['feedback_score']) / 5.0)
        llm_confidence = llm_analysis.get('confidence_modifier', 1.0)
        final_confidence = statistical_confidence * llm_confidence
        
        # Enhanced reasoning with LLM insights
        reasoning = f"Statistical: {len(feedback_group)} entries, avg score {avg_score:.2f}, success rate {success_ratio:.2f}. " + \
                   f"LLM reasoning: {llm_analysis.get('reasoning', 'No additional reasoning')}"
        
        return {
            "parameter": parameter_name,
            "change": final_change,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "llm_insights": llm_analysis
        }
    
    def get_llm_parameter_reasoning(self, param_area: str, sample_feedback: List[str], 
                                   avg_score: float, positive_ratio: float, 
                                   success_ratio: float, outcomes: Dict) -> Dict[str, Any]:
        """Use LLM to analyze feedback and provide reasoning for parameter adjustments"""
        
        system_prompt = f"""
        You are an expert real estate matching system analyst. Analyze feedback patterns to recommend parameter adjustments.
        
        You're analyzing the '{param_area}' parameter area. Based on the feedback patterns, determine:
        1. Should this parameter weight be increased, decreased, or maintained?
        2. How confident are you in this recommendation? (0.0-1.0)
        3. What's the reasoning behind your recommendation?
        4. Are there any nuances the statistics might miss?
        
        Provide JSON response:
        {{
            "recommendation": "increase|decrease|maintain",
            "adjustment_factor": 0.0-2.0,
            "confidence_modifier": 0.0-1.0,
            "reasoning": "detailed explanation",
            "key_insights": ["insight1", "insight2"],
            "statistical_validation": "agree|disagree|partially_agree"
        }}
        """
        
        feedback_summary = "\n".join([f"- {fb}" for fb in sample_feedback[:3]])
        
        user_prompt = f"""
        Parameter Area: {param_area}
        Statistical Summary:
        - Average feedback score: {avg_score:.2f}/5.0
        - Positive feedback ratio: {positive_ratio:.1%}
        - Success rate (leased/toured): {success_ratio:.1%}
        - Showing outcomes: {outcomes}
        
        Sample Realtor Feedback:
        {feedback_summary}
        
        Analyze this data and recommend parameter adjustments with reasoning.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            return json.loads(response.content)
        except Exception as e:
            self.logger.warning(f"LLM analysis failed for {param_area}: {e}")
            return {
                "recommendation": "maintain",
                "adjustment_factor": 1.0,
                "confidence_modifier": 0.5,
                "reasoning": "LLM analysis unavailable, using statistical baseline",
                "key_insights": [],
                "statistical_validation": "agree"
            }
    
    def calculate_statistical_change(self, avg_score: float, positive_ratio: float, success_ratio: float) -> float:
        """Calculate baseline statistical change recommendation"""
        
        if avg_score < 3.0 or positive_ratio < 0.4 or success_ratio < 0.5:
            # Need to improve this parameter
            adjustment_factor = 1.0 - max(avg_score/5.0, positive_ratio, success_ratio)
            change = min(self.config.max_parameter_change, adjustment_factor * self.config.learning_rate)
        else:
            # Parameter working well, small positive adjustment
            adjustment_factor = max(avg_score/5.0, positive_ratio, success_ratio) - 0.6
            change = min(self.config.max_parameter_change/2, adjustment_factor * self.config.learning_rate)
        
        return change
    
    def combine_statistical_and_llm_analysis(self, statistical_change: float, llm_analysis: Dict) -> float:
        """Combine statistical recommendation with LLM insights"""
        
        recommendation = llm_analysis.get('recommendation', 'maintain')
        adjustment_factor = llm_analysis.get('adjustment_factor', 1.0)
        statistical_validation = llm_analysis.get('statistical_validation', 'agree')
        
        if statistical_validation == 'disagree':
            # LLM disagrees with statistics - use LLM guidance but be conservative
            if recommendation == 'increase':
                return min(statistical_change * 0.5, self.config.max_parameter_change * 0.5)
            elif recommendation == 'decrease':
                return max(-statistical_change * 0.5, -self.config.max_parameter_change * 0.5)
            else:  # maintain
                return statistical_change * 0.1  # Very small change
        
        elif statistical_validation == 'partially_agree':
            # LLM partially agrees - moderate the statistical recommendation
            return statistical_change * adjustment_factor * 0.7
        
        else:  # agree
            # LLM agrees with statistics - amplify or moderate based on confidence
            return statistical_change * adjustment_factor
    
    def generate_llm_strategic_insights(self, feedback_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate strategic insights using LLM analysis of all feedback"""
        
        system_prompt = """
        You are a senior real estate technology consultant analyzing feedback patterns from a property matching system.
        
        Provide strategic insights and recommendations based on the feedback data. Focus on:
        1. Overall system performance assessment
        2. Key areas for improvement
        3. Potential new features or capabilities needed
        4. Risk areas that need immediate attention
        5. Long-term strategic recommendations
        
        Return JSON with:
        {
            "overall_assessment": "summary of system performance",
            "critical_issues": ["issue1", "issue2"],
            "improvement_opportunities": ["opportunity1", "opportunity2"], 
            "feature_recommendations": ["feature1", "feature2"],
            "risk_areas": ["risk1", "risk2"],
            "strategic_priorities": ["priority1", "priority2"]
        }
        """
        
        # Prepare summary data for LLM
        summary_stats = {
            "total_feedback": len(feedback_df),
            "avg_score": feedback_df['feedback_score'].mean(),
            "positive_ratio": len(feedback_df[feedback_df['feedback_type'] == 'positive']) / len(feedback_df),
            "success_rate": len(feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured'])]) / len(feedback_df),
            "common_complaints": feedback_df[feedback_df['feedback_type'] == 'negative']['realtor_notes'].head(5).tolist(),
            "success_stories": feedback_df[feedback_df['feedback_type'] == 'positive']['realtor_notes'].head(3).tolist()
        }
        
        user_prompt = f"""
        Real Estate Matching System Feedback Analysis:
        
        Performance Metrics:
        - Total feedback entries: {summary_stats['total_feedback']}
        - Average satisfaction score: {summary_stats['avg_score']:.2f}/5.0
        - Positive feedback ratio: {summary_stats['positive_ratio']:.1%}
        - Success rate (leased/toured): {summary_stats['success_rate']:.1%}
        
        Common Complaints:
        {chr(10).join([f"- {complaint}" for complaint in summary_stats['common_complaints']])}
        
        Success Stories:
        {chr(10).join([f"- {story}" for story in summary_stats['success_stories']])}
        
        Provide strategic analysis and recommendations.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            return json.loads(response.content)
        except Exception as e:
            self.logger.warning(f"LLM strategic analysis failed: {e}")
            return {
                "overall_assessment": "Unable to generate LLM assessment",
                "critical_issues": ["LLM analysis unavailable"],
                "improvement_opportunities": [],
                "feature_recommendations": [],
                "risk_areas": [],
                "strategic_priorities": ["Review system manually"]
            }
    
    def optimize_decision_thresholds(self, feedback_df: pd.DataFrame) -> List[ParameterUpdate]:
        """Optimize showing decision thresholds based on outcomes"""
        
        updates = []
        
        # Analyze quality gate threshold
        quality_update = self.optimize_quality_gate(feedback_df)
        if quality_update:
            updates.append(quality_update)
        
        # Analyze candidate count thresholds
        candidate_update = self.optimize_candidate_thresholds(feedback_df)
        if candidate_update:
            updates.append(candidate_update)
        
        return updates
    
    def optimize_quality_gate(self, feedback_df: pd.DataFrame) -> ParameterUpdate:
        """Optimize the quality gate threshold"""
        
        # Group by showing outcomes
        declined = feedback_df[feedback_df['showing_outcome'] == 'declined']
        successful = feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured'])]
        
        if len(declined) < 3 or len(successful) < 3:
            return None
        
        # Calculate average match quality for each group
        declined_avg_quality = declined['match_quality_rating'].mean()
        successful_avg_quality = successful['match_quality_rating'].mean()
        
        current_threshold = self.current_parameters.get('quality_gate_threshold', 0.45)
        
        # If declined matches have high quality, threshold might be too low
        # If successful matches have low quality, threshold might be too high
        
        if declined_avg_quality > 3.5:  # High quality matches being declined
            suggested_threshold = min(0.7, current_threshold + 0.05)
            reasoning = f"High quality matches ({declined_avg_quality:.1f}/5) being declined - increase threshold"
        elif successful_avg_quality < 3.0:  # Low quality matches succeeding
            suggested_threshold = max(0.3, current_threshold - 0.05)
            reasoning = f"Low quality matches ({successful_avg_quality:.1f}/5) succeeding - decrease threshold"
        else:
            return None  # Threshold seems appropriate
        
        confidence = min(1.0, (len(declined) + len(successful)) / 20.0)
        
        return ParameterUpdate(
            parameter_name="quality_gate_threshold",
            current_value=current_threshold,
            suggested_value=suggested_threshold,
            confidence=confidence,
            reasoning=reasoning,
            feedback_count=len(declined) + len(successful),
            timestamp=datetime.now().isoformat()
        )
    
    def optimize_candidate_thresholds(self, feedback_df: pd.DataFrame) -> ParameterUpdate:
        """Optimize minimum candidate and maximum invite thresholds"""
        
        # Analyze relationship between candidate count and success
        feedback_with_counts = feedback_df.dropna(subset=['realtor_notes'])
        
        # Extract candidate counts from data (would need to be added to feedback schema)
        # For now, simulate based on showing outcomes
        
        successful_outcomes = feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured', 'scheduled'])]
        failed_outcomes = feedback_df[feedback_df['showing_outcome'] == 'declined']
        
        if len(successful_outcomes) < 5 or len(failed_outcomes) < 5:
            return None
        
        # If success rate is very low, might need more candidates
        success_rate = len(successful_outcomes) / len(feedback_df)
        current_min_candidates = self.current_parameters.get('min_candidates', 3)
        
        if success_rate < 0.4:
            suggested_min = min(10, current_min_candidates + 1)
            reasoning = f"Low success rate ({success_rate:.1%}) - increase minimum candidates"
            confidence = 0.7
        elif success_rate > 0.8:
            suggested_min = max(2, current_min_candidates - 1)
            reasoning = f"High success rate ({success_rate:.1%}) - can reduce minimum candidates"
            confidence = 0.6
        else:
            return None
        
        return ParameterUpdate(
            parameter_name="min_candidates",
            current_value=current_min_candidates,
            suggested_value=suggested_min,
            confidence=confidence,
            reasoning=reasoning,
            feedback_count=len(feedback_df),
            timestamp=datetime.now().isoformat()
        )
    
    def calculate_feature_importance(self, feedback_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate importance scores for different matching features"""
        
        feature_importance = {}
        
        # Analyze correlation between feedback scores and different aspects
        location_mentions = feedback_df['realtor_notes'].str.contains('location|commute|neighborhood', case=False)
        price_mentions = feedback_df['realtor_notes'].str.contains('price|budget|expensive', case=False)
        amenity_mentions = feedback_df['realtor_notes'].str.contains('parking|furnished|amenities', case=False)
        lifestyle_mentions = feedback_df['realtor_notes'].str.contains('quiet|pet|lifestyle|vibe', case=False)
        
        # Calculate correlation with success
        successful = feedback_df['showing_outcome'].isin(['leased', 'toured'])
        
        if location_mentions.sum() > 0:
            location_success = feedback_df[location_mentions & successful]
            feature_importance['location'] = len(location_success) / location_mentions.sum() if location_mentions.sum() > 0 else 0
        
        if price_mentions.sum() > 0:
            price_success = feedback_df[price_mentions & successful]
            feature_importance['price'] = len(price_success) / price_mentions.sum() if price_mentions.sum() > 0 else 0
        
        if amenity_mentions.sum() > 0:
            amenity_success = feedback_df[amenity_mentions & successful]
            feature_importance['amenities'] = len(amenity_success) / amenity_mentions.sum() if amenity_mentions.sum() > 0 else 0
        
        if lifestyle_mentions.sum() > 0:
            lifestyle_success = feedback_df[lifestyle_mentions & successful]
            feature_importance['lifestyle'] = len(lifestyle_success) / lifestyle_mentions.sum() if lifestyle_mentions.sum() > 0 else 0
        
        return feature_importance
    
    def evaluate_parameter_updates(self, updates: List[ParameterUpdate]) -> Dict[str, Any]:
        """Evaluate parameter updates and save recommendations without applying them"""
        
        results = {
            "high_confidence_updates": [],
            "low_confidence_updates": [],
            "recommendations_saved": False
        }
        
        for update in updates:
            if update.confidence >= self.config.confidence_threshold:
                results["high_confidence_updates"].append(asdict(update))
                self.logger.info(f"High confidence recommendation: {update.parameter_name} = {update.suggested_value} (confidence: {update.confidence:.2f})")
            else:
                results["low_confidence_updates"].append({
                    **asdict(update),
                    "note": f"Low confidence {update.confidence:.2f} - needs manual review"
                })
                self.logger.info(f"Low confidence recommendation: {update.parameter_name} (confidence: {update.confidence:.2f})")
        
        # Save all recommendations for review
        if updates:
            reason = f"Learning analysis from {len(updates)} parameter recommendations"
            csv_file = self.save_parameter_recommendations(updates, reason)
            results["recommendations_saved"] = True
            results["files_created"] = [csv_file]
        
        return results
    
    def generate_learning_report(self, feedback_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        
        # Analyze feedback patterns with LLM enhancement
        weight_updates = self.analyze_weight_adjustments(feedback_df)
        threshold_updates = self.optimize_decision_thresholds(feedback_df)
        feature_importance = self.calculate_feature_importance(feedback_df)
        
        # Generate LLM strategic insights
        strategic_insights = self.generate_llm_strategic_insights(feedback_df)
        
        all_updates = weight_updates + threshold_updates
        
        # Evaluate updates (don't apply automatically)
        update_results = self.evaluate_parameter_updates(all_updates)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": {
                "total_entries": len(feedback_df),
                "avg_score": feedback_df['feedback_score'].mean(),
                "success_rate": len(feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured'])]) / len(feedback_df)
            },
            "feature_importance": feature_importance,
            "parameter_updates": update_results,
            "current_parameters": self.current_parameters,
            "llm_strategic_insights": strategic_insights,
            "recommendations": self.generate_strategic_recommendations(feedback_df, feature_importance)
        }
        
        # Save report
        os.makedirs("data/learning_reports", exist_ok=True)
        report_file = f"data/learning_reports/learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Learning report saved to {report_file}")
        
        return report
    
    def generate_strategic_recommendations(self, feedback_df: pd.DataFrame, feature_importance: Dict[str, float]) -> List[str]:
        """Generate high-level strategic recommendations"""
        
        recommendations = []
        
        # Analyze overall performance
        avg_score = feedback_df['feedback_score'].mean()
        success_rate = len(feedback_df[feedback_df['showing_outcome'].isin(['leased', 'toured'])]) / len(feedback_df)
        
        if avg_score < 3.5:
            recommendations.append("Overall satisfaction is low - consider comprehensive review of matching criteria")
        
        if success_rate < 0.5:
            recommendations.append("Low showing success rate - review quality thresholds and candidate screening")
        
        # Feature-specific recommendations
        if feature_importance.get('location', 0) < 0.3:
            recommendations.append("Location matching may need improvement - consider geographic/commute factors")
        
        if feature_importance.get('price', 0) < 0.4:
            recommendations.append("Price matching effectiveness is low - review budget interpretation and tolerance")
        
        # Pattern-based recommendations
        negative_feedback = feedback_df[feedback_df['feedback_type'] == 'negative']
        common_issues = negative_feedback['realtor_notes'].str.lower()
        
        if common_issues.str.contains('parking').sum() > len(feedback_df) * 0.1:
            recommendations.append("Parking is a frequent issue - consider adding parking requirements to matching")
        
        if common_issues.str.contains('accessibility').sum() > 0:
            recommendations.append("Accessibility needs mentioned - implement accessibility matching features")
        
        return recommendations

def main():
    """Main function to run learning analysis"""
    learning_module = LearningModule()
    
    # Load feedback data
    feedback_file = "data/synthetic/realtor_feedback.csv"
    if not os.path.exists(feedback_file):
        print(f"Feedback file not found: {feedback_file}")
        return
    
    feedback_df = pd.read_csv(feedback_file)
    print(f"Loaded {len(feedback_df)} feedback entries for learning analysis")
    
    # Generate learning report
    report = learning_module.generate_learning_report(feedback_df)
    
    print("\n=== LEARNING ANALYSIS REPORT ===")
    print(f"Average feedback score: {report['feedback_summary']['avg_score']:.2f}")
    print(f"Success rate: {report['feedback_summary']['success_rate']:.1%}")
    
    print(f"\nHigh confidence recommendations: {len(report['parameter_updates']['high_confidence_updates'])}")
    print(f"Low confidence recommendations: {len(report['parameter_updates']['low_confidence_updates'])}")
    
    if report['parameter_updates']['recommendations_saved']:
        print("Parameter recommendations saved to files for review")
    
    print("\n=== STRATEGIC RECOMMENDATIONS ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print(f"\nDetailed report saved to data/learning_reports/")

if __name__ == "__main__":
    main()