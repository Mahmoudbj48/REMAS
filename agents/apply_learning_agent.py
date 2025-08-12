import csv
import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class ApplyLearningAgent:
    """
    Agent that applies learned parameter changes from recommendation files
    """
    
    def __init__(self, 
                 recommendations_dir: str = "data/parameter_recommendations",
                 weights_file: str = "data/parameters/current_weights.json",
                 confidence_threshold: float = 0.7):
        self.recommendations_dir = recommendations_dir
        self.weights_file = weights_file
        self.confidence_threshold = confidence_threshold
        self.setup_logging()
        self.ensure_directories()
    
    def setup_logging(self):
        """Setup logging for apply learning agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        os.makedirs("logs/learning", exist_ok=True)
    
    def load_current_weights(self) -> Dict[str, float]:
        """Load current parameter weights"""
        default_weights = {
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
        
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    loaded_weights = json.load(f)
                    default_weights.update(loaded_weights)
                    self.logger.info(f"Loaded weights from {self.weights_file}")
            except Exception as e:
                self.logger.warning(f"Could not load weights: {e}, using defaults")
        else:
            self.logger.info("No existing weights file, using defaults")
        
        return default_weights
    
    def save_weights(self, weights: Dict[str, float], changes_applied: List[Dict]):
        """Save updated weights and log changes"""
        # Save weights
        with open(self.weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        # Log the changes
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "changes_applied": changes_applied,
            "new_weights": weights.copy()
        }
        
        log_file = "logs/learning/applied_changes.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.logger.info(f"Updated weights saved to {self.weights_file}")
        self.logger.info(f"Applied {len(changes_applied)} parameter changes")
    
    def get_latest_recommendations_file(self) -> Optional[str]:
        """Get the most recent recommendations CSV file"""
        if not os.path.exists(self.recommendations_dir):
            return None
        
        pattern = os.path.join(self.recommendations_dir, "recommendations_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Sort by modification time, get most recent
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    
    def load_recommendations(self, csv_file: str) -> List[Dict]:
        """Load recommendations from CSV file"""
        recommendations = []
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    row['current_value'] = float(row['current_value'])
                    row['suggested_value'] = float(row['suggested_value'])
                    row['change_amount'] = float(row['change_amount'])
                    row['confidence'] = float(row['confidence'])
                    row['feedback_count'] = int(row['feedback_count'])
                    
                    recommendations.append(row)
            
            self.logger.info(f"Loaded {len(recommendations)} recommendations from {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading recommendations: {e}")
            return []
        
        return recommendations
    
    def filter_high_confidence_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Filter recommendations by confidence threshold"""
        high_confidence = [r for r in recommendations if r['confidence'] >= self.confidence_threshold]
        
        self.logger.info(f"Found {len(high_confidence)} high-confidence recommendations (>= {self.confidence_threshold})")
        self.logger.info(f"Filtered out {len(recommendations) - len(high_confidence)} low-confidence recommendations")
        
        return high_confidence
    
    def validate_parameter_changes(self, recommendations: List[Dict], current_weights: Dict[str, float]) -> List[Dict]:
        """Validate that parameter changes are safe"""
        valid_recommendations = []
        
        for rec in recommendations:
            param_name = rec['parameter_name']
            suggested_value = rec['suggested_value']
            
            # Check parameter bounds
            if param_name in ['location_weight', 'price_weight', 'soft_attributes_weight', 'amenities_weight']:
                if 0.0 <= suggested_value <= 1.0:
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Rejecting {param_name}: value {suggested_value} out of bounds [0,1]")
            
            elif param_name == 'quality_gate_threshold':
                if 0.0 <= suggested_value <= 1.0:
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Rejecting {param_name}: value {suggested_value} out of bounds [0,1]")
            
            elif param_name in ['min_candidates', 'max_invites']:
                if 1 <= suggested_value <= 20:
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Rejecting {param_name}: value {suggested_value} out of bounds [1,20]")
            
            elif param_name == 'price_tolerance':
                if 0.0 <= suggested_value <= 0.5:
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Rejecting {param_name}: value {suggested_value} out of bounds [0,0.5]")
            
            else:
                # Unknown parameter - accept but warn
                self.logger.warning(f"Unknown parameter {param_name}, applying anyway")
                valid_recommendations.append(rec)
        
        self.logger.info(f"Validated {len(valid_recommendations)} out of {len(recommendations)} recommendations")
        return valid_recommendations
    
    def apply_recommendations(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Apply parameter recommendations and return summary"""
        
        if not recommendations:
            return {
                "success": False,
                "message": "No recommendations to apply",
                "changes_applied": [],
                "weights_updated": False
            }
        
        # Load current weights
        current_weights = self.load_current_weights()
        original_weights = current_weights.copy()
        
        # Filter and validate recommendations
        high_confidence = self.filter_high_confidence_recommendations(recommendations)
        valid_recommendations = self.validate_parameter_changes(high_confidence, current_weights)
        
        if not valid_recommendations:
            return {
                "success": False,
                "message": "No valid high-confidence recommendations to apply",
                "changes_applied": [],
                "weights_updated": False
            }
        
        # Apply changes
        changes_applied = []
        for rec in valid_recommendations:
            param_name = rec['parameter_name']
            old_value = current_weights.get(param_name, 0.0)
            new_value = rec['suggested_value']
            
            current_weights[param_name] = new_value
            
            change_info = {
                "parameter": param_name,
                "old_value": old_value,
                "new_value": new_value,
                "change": new_value - old_value,
                "confidence": rec['confidence'],
                "reasoning": rec['reasoning'],
                "feedback_count": rec['feedback_count']
            }
            changes_applied.append(change_info)
            
            self.logger.info(f"Applied: {param_name} {old_value:.3f} → {new_value:.3f} (Δ{new_value-old_value:+.3f})")
        
        # Save updated weights
        self.save_weights(current_weights, changes_applied)
        
        return {
            "success": True,
            "message": f"Successfully applied {len(changes_applied)} parameter changes",
            "changes_applied": changes_applied,
            "weights_updated": True,
            "original_weights": original_weights,
            "new_weights": current_weights
        }
    
    def apply_latest_learning(self) -> Dict[str, Any]:
        """Apply the latest learning recommendations"""
        
        # Find latest recommendations file
        latest_file = self.get_latest_recommendations_file()
        if not latest_file:
            return {
                "success": False,
                "message": "No recommendations file found",
                "changes_applied": [],
                "weights_updated": False
            }
        
        self.logger.info(f"Processing recommendations from {latest_file}")
        
        # Load recommendations
        recommendations = self.load_recommendations(latest_file)
        if not recommendations:
            return {
                "success": False,
                "message": "Could not load recommendations",
                "changes_applied": [],
                "weights_updated": False
            }
        
        # Apply recommendations
        result = self.apply_recommendations(recommendations)
        result["source_file"] = latest_file
        
        return result
    
    def apply_specific_file(self, recommendations_file: str) -> Dict[str, Any]:
        """Apply recommendations from a specific file"""
        
        if not os.path.exists(recommendations_file):
            return {
                "success": False,
                "message": f"Recommendations file not found: {recommendations_file}",
                "changes_applied": [],
                "weights_updated": False
            }
        
        self.logger.info(f"Processing recommendations from {recommendations_file}")
        
        # Load and apply recommendations
        recommendations = self.load_recommendations(recommendations_file)
        result = self.apply_recommendations(recommendations)
        result["source_file"] = recommendations_file
        
        return result
    
    def show_current_weights(self):
        """Display current parameter weights"""
        weights = self.load_current_weights()
        
        print("\n=== CURRENT PARAMETER WEIGHTS ===")
        for param, value in sorted(weights.items()):
            print(f"{param:<25}: {value}")
    
    def show_pending_recommendations(self) -> bool:
        """Show pending recommendations that could be applied"""
        
        latest_file = self.get_latest_recommendations_file()
        if not latest_file:
            print("No recommendations file found")
            return False
        
        recommendations = self.load_recommendations(latest_file)
        if not recommendations:
            print("Could not load recommendations")
            return False
        
        high_confidence = self.filter_high_confidence_recommendations(recommendations)
        
        print(f"\n=== PENDING RECOMMENDATIONS FROM {os.path.basename(latest_file)} ===")
        print(f"Total recommendations: {len(recommendations)}")
        print(f"High confidence (>= {self.confidence_threshold}): {len(high_confidence)}")
        
        if high_confidence:
            print("\nHigh Confidence Changes:")
            print(f"{'Parameter':<25} {'Current':<10} {'Suggested':<10} {'Change':<10} {'Confidence':<10}")
            print("-" * 75)
            
            current_weights = self.load_current_weights()
            for rec in high_confidence:
                param = rec['parameter_name']
                current = current_weights.get(param, 0.0)
                suggested = rec['suggested_value']
                change = suggested - current
                confidence = rec['confidence']
                
                print(f"{param:<25} {current:<10.3f} {suggested:<10.3f} {change:+<10.3f} {confidence:<10.3f}")
        
        return len(high_confidence) > 0

def main():
    """Main CLI for apply learning agent"""
    
    agent = ApplyLearningAgent()
    
    while True:
        print("\n=== APPLY LEARNING AGENT ===")
        print("1. Show current weights")
        print("2. Show pending recommendations") 
        print("3. Apply latest learning")
        print("4. Apply specific file")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            agent.show_current_weights()
        
        elif choice == "2":
            has_recommendations = agent.show_pending_recommendations()
            if not has_recommendations:
                print("No high-confidence recommendations to apply")
        
        elif choice == "3":
            result = agent.apply_latest_learning()
            print(f"\n{result['message']}")
            
            if result['success']:
                print(f"Changes applied: {len(result['changes_applied'])}")
                for change in result['changes_applied']:
                    print(f"  {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f}")
        
        elif choice == "4":
            file_path = input("Enter recommendations file path: ").strip()
            result = agent.apply_specific_file(file_path)
            print(f"\n{result['message']}")
            
            if result['success']:
                print(f"Changes applied: {len(result['changes_applied'])}")
                for change in result['changes_applied']:
                    print(f"  {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f}")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()