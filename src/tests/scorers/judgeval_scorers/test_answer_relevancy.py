import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.answer_relevancy import AnswerRelevancyScorer
from judgeval.constants import APIScorer


class TestAnswerRelevancyScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = AnswerRelevancyScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.ANSWER_RELEVANCY

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            AnswerRelevancyScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            AnswerRelevancyScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = AnswerRelevancyScorer(threshold=0.5)
        assert scorer.__name__ == "Answer Relevancy"