import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.summarization import SummarizationScorer
from judgeval.constants import APIScorer


class TestSummarizationScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = SummarizationScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.SUMMARIZATION

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            SummarizationScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            SummarizationScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = SummarizationScorer(threshold=0.5)
        assert scorer.__name__ == "Summarization"
        