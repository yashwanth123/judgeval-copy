import pytest
from unittest.mock import AsyncMock, Mock, patch
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

from judgeval.scorers.score import safe_a_score_example, score_task
from judgeval.scorers import CustomScorer
from judgeval.data import Example
from judgeval.common.exceptions import MissingTestCaseParamsError


class MockCustomScorer(CustomScorer):
    def score_example(self, example, *args, **kwargs):
        pass

    async def a_score_example(self, example, *args, **kwargs):
        pass

    def success_check(self):
        return True


@pytest.fixture
def example():
    return Example(
        input="test input",
        actual_output="test output",
        example_id="test_id"
    )


@pytest.fixture
def basic_scorer():
    return MockCustomScorer(
        score_type="test_scorer",
        threshold=0.5
    )


@pytest.fixture
def progress():
    return Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )


@pytest.mark.asyncio
async def test_successful_scoring(example, basic_scorer):
    """Test basic successful scoring case"""
    basic_scorer.a_score_example = AsyncMock()
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    basic_scorer.a_score_example.assert_called_once_with(example, _show_indicator=False)
    assert basic_scorer.error is None
    assert not hasattr(basic_scorer, 'skipped') or not basic_scorer.skipped


@pytest.mark.asyncio
async def test_missing_params_with_skip(example, basic_scorer):
    """Test handling of MissingTestCaseParamsError when skip_on_missing_params is True"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.skipped is True
    assert basic_scorer.error is None


@pytest.mark.asyncio
async def test_missing_params_with_ignore_errors(example, basic_scorer):
    """Test handling of MissingTestCaseParamsError when ignore_errors is True but not skipping"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=False
    )
    
    assert basic_scorer.error == "Missing required params"
    assert basic_scorer.success is False


@pytest.mark.asyncio
async def test_missing_params_raises_error(example, basic_scorer):
    """Test that MissingTestCaseParamsError is raised when appropriate"""
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(MissingTestCaseParamsError):
        await safe_a_score_example(
            scorer=basic_scorer,
            example=example,
            ignore_errors=False,
            skip_on_missing_params=False
        )


@pytest.mark.asyncio
async def test_type_error_handling(example, basic_scorer):
    """Test handling of TypeError when _show_indicator is not accepted"""
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert len(calls) == 2  # Should try twice - once with _show_indicator, once without
    assert '_show_indicator' in calls[0]  # First attempt includes _show_indicator
    assert '_show_indicator' not in calls[1]  # Second attempt doesn't include _show_indicator


@pytest.mark.asyncio
async def test_general_exception_with_ignore(example, basic_scorer):
    """Test handling of general exceptions when ignore_errors is True"""
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.error == "Test error"
    assert basic_scorer.success is False


@pytest.mark.asyncio
async def test_general_exception_raises(example, basic_scorer):
    """Test that general exceptions are raised when ignore_errors is False"""
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(ValueError):
        await safe_a_score_example(
            scorer=basic_scorer,
            example=example,
            ignore_errors=False,
            skip_on_missing_params=True
        )


@pytest.mark.asyncio
async def test_error_with_missing_params(example, basic_scorer):
    """Test handling of TypeError followed by MissingTestCaseParamsError"""
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        raise MissingTestCaseParamsError("Missing params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    await safe_a_score_example(
        scorer=basic_scorer,
        example=example,
        ignore_errors=True,
        skip_on_missing_params=True
    )
    
    assert basic_scorer.skipped is True
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_task_successful_scoring(example, basic_scorer, progress):
    """Test basic successful scoring case with progress tracking"""
    task_id = progress.add_task(description="Test Task", total=100)
    basic_scorer.a_score_example = AsyncMock()
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    basic_scorer.a_score_example.assert_called_once_with(example, _show_indicator=False)
    assert progress.tasks[task_id].completed == 100
    assert "Completed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_missing_params_with_skip(example, basic_scorer, progress):
    """Test handling of MissingTestCaseParamsError when skip_on_missing_params is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            skip_on_missing_params=True
        )
    
    assert basic_scorer.skipped is True
    assert not progress.tasks[task_id].completed  # Task should not be marked as complete


@pytest.mark.asyncio
async def test_task_missing_params_with_ignore_errors(example, basic_scorer, progress):
    """Test handling of MissingTestCaseParamsError when ignore_errors is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            skip_on_missing_params=False,
            ignore_errors=True
        )
    
    assert basic_scorer.error == "Missing required params"
    assert basic_scorer.success is False
    assert progress.tasks[task_id].completed == 100
    assert "Failed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_missing_params_raises_error(example, basic_scorer, progress):
    """Test that MissingTestCaseParamsError is raised when appropriate"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise MissingTestCaseParamsError("Missing required params")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(MissingTestCaseParamsError):
        with progress:
            await score_task(
                task_id=task_id,
                progress=progress,
                scorer=basic_scorer,
                example=example,
                skip_on_missing_params=False,
                ignore_errors=False
            )


@pytest.mark.asyncio
async def test_task_type_error_handling(example, basic_scorer, progress):
    """Test handling of TypeError when _show_indicator is not accepted"""
    task_id = progress.add_task(description="Test Task", total=100)
    calls = []
    
    async def mock_score(*args, **kwargs):
        calls.append(kwargs)
        if '_show_indicator' in kwargs:
            raise TypeError("_show_indicator not accepted")
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    assert len(calls) == 2  # Should try twice - once with _show_indicator, once without
    assert progress.tasks[task_id].completed == 100
    assert "Completed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_general_exception_with_ignore(example, basic_scorer, progress):
    """Test handling of general exceptions when ignore_errors is True"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example,
            ignore_errors=True
        )
    
    assert basic_scorer.error == "Test error"
    assert basic_scorer.success is False
    assert progress.tasks[task_id].completed == 100
    assert "Failed" in progress.tasks[task_id].description


@pytest.mark.asyncio
async def test_task_general_exception_raises(example, basic_scorer, progress):
    """Test that general exceptions are raised when ignore_errors is False"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        raise ValueError("Test error")
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with pytest.raises(ValueError):
        with progress:
            await score_task(
                task_id=task_id,
                progress=progress,
                scorer=basic_scorer,
                example=example,
                ignore_errors=False
            )


@pytest.mark.asyncio
async def test_task_progress_timing(example, basic_scorer, progress):
    """Test that timing information is correctly added to progress description"""
    task_id = progress.add_task(description="Test Task", total=100)
    
    async def mock_score(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate some work
        return True
    
    basic_scorer.a_score_example = AsyncMock(side_effect=mock_score)
    
    with progress:
        await score_task(
            task_id=task_id,
            progress=progress,
            scorer=basic_scorer,
            example=example
        )
    
    assert "(" in progress.tasks[task_id].description
    assert "s)" in progress.tasks[task_id].description  # Should show timing
