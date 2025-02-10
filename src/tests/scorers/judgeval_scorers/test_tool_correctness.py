import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.tool_correctness import ToolCorrectnessScorer
from judgeval.constants import APIScorer


class TestToolCorrectnessScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = ToolCorrectnessScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.TOOL_CORRECTNESS

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            ToolCorrectnessScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            ToolCorrectnessScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = ToolCorrectnessScorer(threshold=0.5)
        assert scorer.__name__ == "Tool Correctness"
