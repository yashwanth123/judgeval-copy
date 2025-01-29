import pytest
from judgeval.data import ScoringResult, ScorerData
from judgeval.run_evaluation import assert_test

def test_assert_test_all_passing():
    """Test when all results are successful"""
    scorer_data = ScorerData(
        name="test_scorer",
        success=True,
        score=0.9,
        threshold=0.8,
        reason="Test passed",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        evaluation_cost=0.1,
        verbose_logs="test logs",
        additional_metadata={}
    )
    
    result = ScoringResult(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context="test context",
        retrieval_context="test retrieval",
        eval_run_name="test_run",
        success=True,
        scorers_data=[scorer_data]
    )
    
    # Should not raise any exception
    assert_test([result])

def test_assert_test_failed_scorer():
    """Test when a scorer fails"""
    failed_scorer = ScorerData(
        name="failed_scorer",
        success=False,
        score=0.5,
        threshold=0.8,
        reason="Score below threshold",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        evaluation_cost=0.1,
        verbose_logs="test logs",
        additional_metadata={}
    )
    
    result = ScoringResult(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context="test context",
        retrieval_context="test retrieval",
        eval_run_name="test_run",
        success=False,
        scorers_data=[failed_scorer]
    )
    
    with pytest.raises(AssertionError) as exc_info:
        assert_test([result])
    
    # Verify error message contains relevant information
    error_msg = str(exc_info.value)
    assert "test input" in error_msg
    assert "test output" in error_msg
    assert "failed_scorer" in error_msg
    assert "Score below threshold" in error_msg

def test_assert_test_multiple_failed_scorers():
    """Test when multiple scorers fail"""
    failed_scorer1 = ScorerData(
        name="scorer1",
        success=False,
        score=0.5,
        threshold=0.8,
        reason="First failure",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        evaluation_cost=0.1,
        verbose_logs="test logs",
        additional_metadata={}
    )
    
    failed_scorer2 = ScorerData(
        name="scorer2",
        success=False,
        score=0.6,
        threshold=0.8,
        reason="Second failure",
        strict_mode=True,
        evaluation_model="gpt-4",
        error=None,
        evaluation_cost=0.1,
        verbose_logs="test logs",
        additional_metadata={}
    )
    
    result = ScoringResult(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context="test context",
        retrieval_context="test retrieval",
        eval_run_name="test_run",
        success=False,
        scorers_data=[failed_scorer1, failed_scorer2]
    )
    
    with pytest.raises(AssertionError) as exc_info:
        assert_test([result])
    
    error_msg = str(exc_info.value)
    assert "scorer1" in error_msg
    assert "scorer2" in error_msg
    assert "First failure" in error_msg
    assert "Second failure" in error_msg

def test_assert_test_no_scorer_data():
    """Test when result has no scorer data"""
    result = ScoringResult(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context="test context",
        retrieval_context="test retrieval",
        eval_run_name="test_run",
        success=False,
        scorers_data=None
    )
    
    with pytest.raises(AssertionError) as exc_info:
        assert_test([result])
    
    error_msg = str(exc_info.value)
    assert "test input" in error_msg
    assert "test output" in error_msg

def test_assert_test_empty_results():
    """Test with empty results list"""
    # Should not raise any exception
    assert_test([])

