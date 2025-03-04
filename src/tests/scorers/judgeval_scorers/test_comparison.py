import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.comparison.comparison_scorer import ComparisonScorer
from judgeval.constants import APIScorer


class TestComparisonScorer:
    def test_init(self):
        threshold = 1
        scorer = ComparisonScorer(threshold=threshold, criteria="criteria", description="description")
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.COMPARISON

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            ComparisonScorer(threshold=-0.1, criteria="criteria", description="description")
        
        with pytest.raises(ValueError):
            ComparisonScorer(threshold=1.1, criteria="criteria", description="description")

    def test_name_property(self):
        # Test the __name__ property
        scorer = ComparisonScorer(threshold=0.5, criteria="Use of Evidence and Details", description="description")
        assert scorer.__name__ == "Comparison - Use of Evidence and Details"
