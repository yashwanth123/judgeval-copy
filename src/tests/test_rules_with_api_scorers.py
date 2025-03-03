"""Tests to verify rules work correctly with API scorers."""

import pytest
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch

from judgeval.rules import Rule, Condition, Operator, RulesEngine, AlertStatus
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example

class TestRulesWithAPIScorers:
    
    def test_rules_with_api_scorers(self):
        """Test that rules work correctly with API scorers."""
        # Create API scorers
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        
        # Create rules with API scorers
        rules = {
            "quality_check": Rule(
                name="Quality Check",
                conditions=[
                    Condition(metric=faithfulness_scorer, operator=Operator.GTE, threshold=0.7),
                    Condition(metric=relevancy_scorer, operator=Operator.GTE, threshold=0.8)
                ],
                combine_type="all"
            )
        }
        
        # Create RulesEngine
        engine = RulesEngine(rules)
        
        # Create scores dictionary that matches the metric names
        # Important: the key should match what's returned by the metric_name property
        scores = {
            "faithfulness": 0.8,  # This should match faithfulness_scorer.score_type
            "answer_relevancy": 0.9  # This should match relevancy_scorer.score_type
        }
        
        # Print the metric names we're expecting
        print(f"Faithfulness metric name: {faithfulness_scorer.score_type}")
        print(f"Relevancy metric name: {relevancy_scorer.score_type}")
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Verify results
        assert "quality_check" in results
        assert results["quality_check"].status == AlertStatus.TRIGGERED
        
        # Verify conditions were not skipped
        for condition in results["quality_check"].conditions_result:
            assert condition["skipped"] is False
            assert condition["passed"] is True
    
    def test_rule_metric_name_resolution(self):
        """Test that metric names are correctly resolved in rules."""
        # Create API scorers
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        
        # Create conditions with these scorers
        condition1 = Condition(metric=faithfulness_scorer, operator=Operator.GTE, threshold=0.7)
        condition2 = Condition(metric=relevancy_scorer, operator=Operator.GTE, threshold=0.8)
        
        # Check metric_name property returns the expected value
        assert condition1.metric_name == "faithfulness"
        assert condition2.metric_name == "answer_relevancy"
        
        # Create scores with these exact keys
        scores = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9
        }
        
        # Test direct condition evaluation to ensure we're getting values correctly
        value1 = scores.get(condition1.metric_name)
        value2 = scores.get(condition2.metric_name)
        
        assert value1 is not None
        assert value2 is not None
        assert value1 == 0.8
        assert value2 == 0.9
    
    @patch('judgeval.judgment_client.run_eval')
    def test_judgment_client_with_rules(self, mock_run_eval):
        """Test that JudgmentClient works with API scorers in rules."""
        # Mock the run_eval function
        mock_result = MagicMock()
        mock_result.alert_results = {
            "rule_0": {
                "status": "triggered",
                "rule_name": "Quality Check",
                "conditions_result": [
                    {"metric": "faithfulness", "value": 0.8, "threshold": 0.7, "passed": True, "skipped": False},
                    {"metric": "answer_relevancy", "value": 0.9, "threshold": 0.8, "passed": True, "skipped": False}
                ]
            }
        }
        mock_run_eval.return_value = [mock_result]
        
        # Create client with patched _validate_api_key method
        with patch.object(JudgmentClient, '_validate_api_key', return_value=(True, {"detail": {"user_name": "test_user"}})):
            client = JudgmentClient(judgment_api_key="test_key")
        
            # Create example
            example = Example(
                input="Test input",
                actual_output="Test output",
                expected_output="Expected output"
            )
        
            # Create scorers
            scorers = [
                FaithfulnessScorer(threshold=0.7),
                AnswerRelevancyScorer(threshold=0.8)
            ]
        
            # Create rules
            rules = [
                Rule(
                    name="Quality Check",
                    conditions=[
                        Condition(metric=FaithfulnessScorer(threshold=0.7), operator=Operator.GTE, threshold=0.7),
                        Condition(metric=AnswerRelevancyScorer(threshold=0.8), operator=Operator.GTE, threshold=0.8)
                    ],
                    combine_type="all"
                )
            ]
        
            # Run evaluation
            result = client.run_evaluation(
                examples=[example],
                scorers=scorers,
                model="gpt-3.5-turbo",
                rules=rules
            )
        
            # Verify run_eval was called with the expected arguments
            # This checks that rules were passed correctly
            assert mock_run_eval.called
            call_args = mock_run_eval.call_args[0][0]
            assert hasattr(call_args, 'rules')
            assert call_args.rules is not None 