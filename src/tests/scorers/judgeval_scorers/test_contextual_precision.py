import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.contextual_precision import ContextualPrecisionScorer
from judgeval.constants import APIScorer


class TestContextualPrecisionScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = ContextualPrecisionScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.CONTEXTUAL_PRECISION

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            ContextualPrecisionScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            ContextualPrecisionScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = ContextualPrecisionScorer(threshold=0.5)
        assert scorer.__name__ == "Contextual Precision"
