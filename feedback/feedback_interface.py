import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

@dataclass
class FeedbackInput:
    """Structure for realtor feedback input"""
    owner_id: str
    user_id: str
    feedback_type: str  # 'positive' or 'negative'
    feedback_score: float  # 1.0 - 5.0
    match_quality_rating: int  # 1 - 5
    showing_outcome: str  # 'scheduled', 'toured', 'leased', 'declined'
    realtor_notes: str
    suggested_adjustments: str = ""

class FeedbackInterface:
    """
    Interface for collecting and managing realtor feedback
    """
    
    def __init__(self, feedback_csv_path: str = "data/synthetic/realtor_feedback.csv"):
        self.feedback_csv_path = feedback_csv_path
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Ensure the feedback CSV file exists with proper headers"""
        if not os.path.exists(self.feedback_csv_path):
            os.makedirs(os.path.dirname(self.feedback_csv_path), exist_ok=True)
            
            headers = [
                'feedback_id', 'timestamp', 'owner_id', 'user_id', 'feedback_type',
                'feedback_score', 'realtor_notes', 'match_quality_rating', 
                'showing_outcome', 'suggested_adjustments', 'match_parameters_used'
            ]
            
            with open(self.feedback_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def collect_feedback_cli(self) -> Optional[FeedbackInput]:
        """Command-line interface for collecting feedback"""
        
        print("\n=== REALTOR FEEDBACK COLLECTION ===")
        
        try:
            # Get basic match information
            owner_id = input("Owner ID: ").strip()
            if not owner_id:
                print("Owner ID is required")
                return None
            
            user_id = input("User ID: ").strip()
            if not user_id:
                print("User ID is required")
                return None
            
            # Get feedback type
            print("\nFeedback Type:")
            print("1. Positive")
            print("2. Negative")
            feedback_choice = input("Select (1-2): ").strip()
            
            if feedback_choice == "1":
                feedback_type = "positive"
            elif feedback_choice == "2":
                feedback_type = "negative"
            else:
                print("Invalid feedback type")
                return None
            
            # Get feedback score
            while True:
                try:
                    feedback_score = float(input("Feedback Score (1.0-5.0): ").strip())
                    if 1.0 <= feedback_score <= 5.0:
                        break
                    else:
                        print("Score must be between 1.0 and 5.0")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get match quality rating
            while True:
                try:
                    match_quality = int(input("Match Quality Rating (1-5): ").strip())
                    if 1 <= match_quality <= 5:
                        break
                    else:
                        print("Rating must be between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get showing outcome
            print("\nShowing Outcome:")
            print("1. Scheduled")
            print("2. Toured") 
            print("3. Leased")
            print("4. Declined")
            outcome_choice = input("Select (1-4): ").strip()
            
            outcome_map = {
                "1": "scheduled",
                "2": "toured", 
                "3": "leased",
                "4": "declined"
            }
            
            showing_outcome = outcome_map.get(outcome_choice)
            if not showing_outcome:
                print("Invalid showing outcome")
                return None
            
            # Get detailed notes
            print("\nRealtor Notes (detailed feedback):")
            realtor_notes = input().strip()
            if not realtor_notes:
                print("Realtor notes are required")
                return None
            
            # Get suggested adjustments
            print("\nSuggested Adjustments (optional):")
            suggested_adjustments = input().strip()
            
            return FeedbackInput(
                owner_id=owner_id,
                user_id=user_id,
                feedback_type=feedback_type,
                feedback_score=feedback_score,
                match_quality_rating=match_quality,
                showing_outcome=showing_outcome,
                realtor_notes=realtor_notes,
                suggested_adjustments=suggested_adjustments
            )
            
        except KeyboardInterrupt:
            print("\nFeedback collection cancelled")
            return None
        except Exception as e:
            print(f"Error collecting feedback: {e}")
            return None
    
    def save_feedback(self, feedback: FeedbackInput, match_parameters: Dict[str, Any] = None) -> str:
        """Save feedback to CSV file"""
        
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Convert match parameters to JSON string
        match_params_str = json.dumps(match_parameters) if match_parameters else "{}"
        
        row = [
            feedback_id,
            timestamp,
            feedback.owner_id,
            feedback.user_id,
            feedback.feedback_type,
            feedback.feedback_score,
            feedback.realtor_notes,
            feedback.match_quality_rating,
            feedback.showing_outcome,
            feedback.suggested_adjustments,
            match_params_str
        ]
        
        with open(self.feedback_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f" Feedback saved with ID: {feedback_id}")
        return feedback_id
    
    def validate_feedback(self, feedback: FeedbackInput) -> List[str]:
        """Validate feedback input and return list of errors"""
        
        errors = []
        
        # Required fields
        if not feedback.owner_id:
            errors.append("Owner ID is required")
        
        if not feedback.user_id:
            errors.append("User ID is required")
        
        if not feedback.realtor_notes:
            errors.append("Realtor notes are required")
        
        # Value ranges
        if not (1.0 <= feedback.feedback_score <= 5.0):
            errors.append("Feedback score must be between 1.0 and 5.0")
        
        if not (1 <= feedback.match_quality_rating <= 5):
            errors.append("Match quality rating must be between 1 and 5")
        
        # Valid enums
        if feedback.feedback_type not in ['positive', 'negative']:
            errors.append("Feedback type must be 'positive' or 'negative'")
        
        if feedback.showing_outcome not in ['scheduled', 'toured', 'leased', 'declined']:
            errors.append("Invalid showing outcome")
        
        return errors
    
    def batch_feedback_collection(self) -> List[str]:
        """Collect multiple feedback entries in batch"""
        
        feedback_ids = []
        
        print("\n=== BATCH FEEDBACK COLLECTION ===")
        print("Enter feedback entries. Type 'done' when finished.\n")
        
        while True:
            feedback = self.collect_feedback_cli()
            
            if feedback is None:
                continue
            
            # Validate feedback
            errors = self.validate_feedback(feedback)
            if errors:
                print("\nValidation errors:")
                for error in errors:
                    print(f"- {error}")
                print("Please try again.\n")
                continue
            
            # Save feedback
            feedback_id = self.save_feedback(feedback)
            feedback_ids.append(feedback_id)
            
            # Ask if user wants to continue
            while True:
                continue_choice = input("\nAdd another feedback entry? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    break
                elif continue_choice in ['n', 'no']:
                    return feedback_ids
                else:
                    print("Please enter 'y' or 'n'")
        
        return feedback_ids
    
    def display_feedback_summary(self, num_recent: int = 10):
        """Display summary of recent feedback"""
        
        if not os.path.exists(self.feedback_csv_path):
            print("No feedback data available")
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(self.feedback_csv_path)
            
            if df.empty:
                print("No feedback entries found")
                return
            
            # Sort by timestamp (most recent first)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            recent_df = df.head(num_recent)
            
            print(f"\n=== RECENT FEEDBACK SUMMARY (Last {len(recent_df)} entries) ===")
            
            # Overall stats
            total_feedback = len(df)
            positive_count = len(df[df['feedback_type'] == 'positive'])
            negative_count = len(df[df['feedback_type'] == 'negative'])
            avg_score = df['feedback_score'].mean()
            avg_quality = df['match_quality_rating'].mean()
            
            print(f"Total Feedback Entries: {total_feedback}")
            print(f"Positive: {positive_count} ({positive_count/total_feedback:.1%})")
            print(f"Negative: {negative_count} ({negative_count/total_feedback:.1%})")
            print(f"Average Score: {avg_score:.2f}/5.0")
            print(f"Average Match Quality: {avg_quality:.2f}/5")
            
            # Outcome distribution
            print(f"\nShowing Outcomes:")
            outcomes = df['showing_outcome'].value_counts()
            for outcome, count in outcomes.items():
                print(f"  {outcome.title()}: {count} ({count/total_feedback:.1%})")
            
            # Recent entries
            print(f"\nRecent Entries:")
            for _, row in recent_df.iterrows():
                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"  {timestamp_str} | {row['feedback_type'].title()} | Score: {row['feedback_score']:.1f} | {row['showing_outcome'].title()}")
                print(f"    Notes: {row['realtor_notes'][:80]}{'...' if len(row['realtor_notes']) > 80 else ''}")
            
        except ImportError:
            print("pandas not available - showing basic summary")
            self._display_basic_summary(num_recent)
        except Exception as e:
            print(f"Error displaying summary: {e}")
    
    def _display_basic_summary(self, num_recent: int):
        """Basic summary without pandas"""
        
        with open(self.feedback_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            print("No feedback entries found")
            return
        
        total = len(rows)
        positive = sum(1 for row in rows if row['feedback_type'] == 'positive')
        negative = total - positive
        
        print(f"\n=== FEEDBACK SUMMARY ===")
        print(f"Total Entries: {total}")
        print(f"Positive: {positive} ({positive/total:.1%})")
        print(f"Negative: {negative} ({negative/total:.1%})")
        
        # Show recent entries
        recent_rows = rows[-num_recent:]
        print(f"\nRecent {len(recent_rows)} Entries:")
        for row in recent_rows:
            print(f"  {row['timestamp'][:16]} | {row['feedback_type'].title()} | Score: {row['feedback_score']}")
    
    def export_feedback_json(self, output_file: str = None) -> str:
        """Export feedback data to JSON format"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/exports/feedback_export_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        feedback_data = []
        
        with open(self.feedback_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                try:
                    row['feedback_score'] = float(row['feedback_score'])
                    row['match_quality_rating'] = int(row['match_quality_rating'])
                    if row['match_parameters_used']:
                        row['match_parameters_used'] = json.loads(row['match_parameters_used'])
                except (ValueError, json.JSONDecodeError):
                    pass  # Keep original string values if conversion fails
                
                feedback_data.append(row)
        
        with open(output_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print(f"Feedback data exported to {output_file}")
        return output_file

def main():
    """Main CLI interface for feedback collection"""
    
    interface = FeedbackInterface()
    
    while True:
        print("\n=== FEEDBACK INTERFACE ===")
        print("1. Collect single feedback")
        print("2. Batch feedback collection")
        print("3. View feedback summary")
        print("4. Export feedback to JSON")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            feedback = interface.collect_feedback_cli()
            if feedback:
                errors = interface.validate_feedback(feedback)
                if errors:
                    print("\nValidation errors:")
                    for error in errors:
                        print(f"- {error}")
                else:
                    interface.save_feedback(feedback)
        
        elif choice == "2":
            feedback_ids = interface.batch_feedback_collection()
            print(f"\nCollected {len(feedback_ids)} feedback entries")
        
        elif choice == "3":
            num_recent = input("Number of recent entries to show (default 10): ").strip()
            try:
                num_recent = int(num_recent) if num_recent else 10
            except ValueError:
                num_recent = 10
            interface.display_feedback_summary(num_recent)
        
        elif choice == "4":
            output_file = input("Output file (press Enter for auto-generated): ").strip()
            if not output_file:
                output_file = None
            interface.export_feedback_json(output_file)
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()