#!/usr/bin/env python3
"""
Demo for using rules-based alerts with different types of scorers in Judgeval.

This script demonstrates:
1. How to create API scorers and custom scorers
2. How to define rules with conditions for these scorers
3. How to run an evaluation with rules
4. How to check the alert results
"""

import sys
import os
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from typing import List, Dict, Any, Optional
from datetime import datetime

from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_correctness import AnswerCorrectnessScorer
from judgeval.data import Example
from judgeval.rules import Rule, Condition, Operator


# Create sample examples for evaluation
def create_examples() -> List[Example]:
    """
    Create sample examples for evaluation.
    
    Returns:
        A list of Example objects
    """
    return [
        Example(
            example_id="123",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of France?",
            actual_output="The capital of France is Paris, which is located on the Seine River.",
            expected_output="Paris is the capital of France.",
            context=["France is a country in Western Europe.", 
                     "Paris is the capital and most populous city of France."]
        ),
        Example(
            example_id="12",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of Germany?",
            actual_output="Berlin is located in northeastern Germany and is the capital of Germany.",
            expected_output="Berlin is the capital of Germany.",
            context=["Germany is a country in Central Europe.", 
                     "Berlin is the capital and largest city of Germany."]
        ),
        Example(
            example_id="10",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of Italy?",
            actual_output="The capital of Italy is Milano, which is a major financial center.",  # This is incorrect
            expected_output="Rome is the capital of Italy.",
            context=["Italy is a country in Southern Europe.", 
                     "Rome is the capital city of Italy."]
        ),
    ]


# Simple Custom Scorer Implementation
class SimpleKeywordScorer(JudgevalScorer):
    """
    A simple scorer that checks for keyword presence in the response.
    """
    
    def __init__(self, 
                 keywords: list[str], 
                 threshold: float = 0.5,
                 include_reason: bool = True):
        """
        Initialize the SimpleKeywordScorer.
        
        Args:
            keywords: List of keywords to look for
            threshold: Threshold for success (default 0.5)
            include_reason: Whether to include reason in output
        """
        super().__init__(
            score_type="Simple Keyword",
            threshold=threshold,
            include_reason=include_reason
        )
        self.keywords = [k.lower() for k in keywords]
    
    def score_example(self, example: Example, **kwargs) -> Dict[str, Any]:
        """
        Score an example based on keyword presence.
        
        Args:
            example: The example to evaluate
            
        Returns:
            Dictionary with score results
        """
        try:
            if not example.actual_output:
                self.score = 0.0
                self.reason = "Empty response"
                self.success = False
                return {
                    "success": False,
                    "score": 0.0,
                    "reason": "Empty response"
                }
            
            response_lower = example.actual_output.lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in self.keywords if keyword in response_lower)
            
            # Calculate score
            if len(self.keywords) == 0:
                self.score = 1.0  # Perfect score if no keywords specified
            else:
                self.score = matches / len(self.keywords)
            
            self.reason = f"Found {matches} out of {len(self.keywords)} keywords"
            self.success = self._success_check()
            
            return {
                "success": self.success,
                "score": self.score,
                "reason": self.reason
            }
        except Exception as e:
            self.error = str(e)
            self.success = False
            return {
                "success": False,
                "score": 0.0,
                "error": self.error
            }
    
    async def a_score_example(self, example: Example, **kwargs) -> Dict[str, Any]:
        """
        Async version of score_example (just calls the sync version)
        """
        return self.score_example(example, **kwargs)
    
    def _success_check(self) -> bool:
        """
        Check if the score meets the threshold
        """
        return self.score >= self.threshold if hasattr(self, 'score') else False
    
    @property
    def __name__(self):
        """Return a human-readable name for the scorer"""
        return "Simple Keyword Scorer"


def main():
    """Run the demo for rules-based alerts."""
    # Create client
    client = JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))
    
    # Create examples
    examples = create_examples()
    
    # Create scorers
    # API scorers (built-in)
    faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
    relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
    correctness_scorer = AnswerCorrectnessScorer(threshold=0.9)
    
    # Custom scorer - check for capital city keywords
    keyword_scorer = SimpleKeywordScorer(
        keywords=["capital", "city", "located"],
        threshold=0.6
    )
    
    # Create rules
    rules = [
        Rule(
            name="All Conditions Check",
            description="Check if all conditions are met",
            conditions=[
                # Use the scorer name as the metric ID, not the score_type
                Condition(metric="Faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="Answer Relevancy", operator=Operator.GTE, threshold=0.8),
                Condition(metric="Answer Correctness", operator=Operator.GTE, threshold=0.9)
            ],
            combine_type="all"  # Require all conditions to trigger
        ),
        Rule(
            name="Any Condition Check",
            description="Check if any condition is met",
            conditions=[
                Condition(metric="Faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="Answer Relevancy", operator=Operator.GTE, threshold=0.8),
                Condition(metric="Answer Correctness", operator=Operator.GTE, threshold=0.9)
            ],
            combine_type="any"  # Require any condition to trigger
        ),
        # New rule for custom scorer
        Rule(
            name="Keyword Check",
            description="Check if response contains enough keywords",
            conditions=[
                Condition(metric="Simple Keyword", operator=Operator.GTE, threshold=0.6)
            ],
            combine_type="all"
        ),
        # Combined rule with custom and API scorers
        Rule(
            name="Comprehensive Quality Check",
            description="Check for both keyword presence and correctness",
            conditions=[
                Condition(metric="Simple Keyword", operator=Operator.GTE, threshold=0.6),
                Condition(metric="Answer Correctness", operator=Operator.GTE, threshold=0.8)
            ],
            combine_type="all"
        )
    ]
    
    # Run evaluation with rules
    print("Running evaluation with rules...")
    results = client.run_evaluation(
        project_name="rules_demo",
        eval_run_name="rules_test",
        examples=examples,
        scorers=[faithfulness_scorer, relevancy_scorer, correctness_scorer, keyword_scorer],  # Added custom scorer
        model="gpt-4",  # Or any other supported model
        rules=rules, override=True
    )
    
    # Process and display results
    print("\nEvaluation Results:")
    for idx, result in enumerate(results):
        print(f"\nExample {idx+1} (ID: {result.example_id}):")
        print(f"  Input: {result.input}")
        print(f"  Actual Output: {result.actual_output}")
        print(f"  Expected Output: {result.expected_output}")
        
        # Display scorer results
        print("  Scorer Results:")
        for scorer_data in result.scorers_data:
            print(f"    {scorer_data.name}: {scorer_data.score:.2f} (Threshold: {scorer_data.threshold:.2f})")
        
        # Display alert results
        if result.additional_metadata and "alerts" in result.additional_metadata:
            print("  Alert Results:")
            alerts = result.additional_metadata["alerts"]
            for rule_id, alert in alerts.items():
                status = alert["status"]
                status_str = "TRIGGERED" if status == "triggered" else "NOT TRIGGERED"
                print(f"    Rule '{rule_id}': {status_str}")
                
                # Show condition results
                print("    Condition Results:")
                for cond in alert["conditions_results"]:
                    passed_str = "PASSED" if cond["passed"] else "FAILED"
                    print(f"      {cond['metric']}: {cond['value']:.2f} {cond['operator']} {cond['threshold']:.2f} - {passed_str}")
                
        print("-" * 80)


if __name__ == "__main__":
    main() 