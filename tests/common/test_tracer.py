import pytest
import time
from unittest.mock import Mock, patch
from judgeval.common.tracer import tracer, TraceEntry, TraceClient

@pytest.fixture
def reset_tracer():
    """Reset tracer state between tests"""
    tracer.depth = 0
    tracer._current_trace = None
    tracer.api_key = None
    yield
    
@pytest.fixture
def configured_tracer(reset_tracer):
    """Provide a configured tracer"""
    tracer.configure("test_api_key")
    return tracer

def test_tracer_singleton():
    """Test that Tracer maintains singleton pattern"""
    from judgeval.common.tracer import Tracer
    tracer1 = Tracer()
    tracer2 = Tracer()
    assert tracer1 is tracer2

def test_tracer_configuration(reset_tracer):
    """Test tracer configuration"""
    assert tracer.api_key is None
    tracer.configure("test_api_key")
    assert tracer.api_key == "test_api_key"

def test_trace_entry_print(capsys):
    """Test TraceEntry print formatting"""
    # Test each type of entry
    entries = [
        TraceEntry(type="enter", function="test_func", depth=1, message="", timestamp=0),
        TraceEntry(type="exit", function="test_func", depth=1, message="", timestamp=0, duration=0.5),
        TraceEntry(type="output", function="test_func", depth=1, message="", timestamp=0, output="result"),
        TraceEntry(type="input", function="test_func", depth=1, message="", timestamp=0, inputs={"arg": 1}),
    ]
    
    expected_outputs = [
        "  → test_func\n",
        "  ← test_func (0.500s)\n",
        "  Output: result\n",
        "  Input: {'arg': 1}\n",
    ]
    
    for entry, expected in zip(entries, expected_outputs):
        entry.print_entry()
        captured = capsys.readouterr()
        assert captured.out == expected

@pytest.mark.asyncio
async def test_trace_client_operations(configured_tracer):
    """Test TraceClient basic operations"""
    trace = configured_tracer.start_trace("test_trace")
    
    # Test adding entries
    entry = TraceEntry(
        type="enter",
        function="test_func",
        depth=0,
        message="test",
        timestamp=time.time()
    )
    trace.add_entry(entry)
    assert len(trace.entries) == 1
    assert trace.entries[0] == entry
    
    # Test duration calculation
    time.sleep(0.1)
    duration = trace.get_duration()
    assert duration > 0

def test_observe_decorator(configured_tracer):
    """Test the @tracer.observe decorator"""
    results = []
    
    @tracer.observe
    def test_function(x, y):
        results.append(f"Function called with {x}, {y}")
        return x + y
    
    with patch.object(TraceClient, 'add_entry') as mock_add_entry:
        trace = configured_tracer.start_trace("test_trace")
        result = test_function(1, 2)
        
        # Verify function execution
        assert result == 3
        assert results == ["Function called with 1, 2"]
        
        # Verify trace entries
        assert mock_add_entry.call_count == 4  # enter, input, output, exit
        
        # Verify entry types
        entry_types = [call.args[0].type for call in mock_add_entry.call_args_list]
        assert entry_types == ["enter", "input", "output", "exit"]

@pytest.mark.asyncio
async def test_save_trace(configured_tracer, mocker):
    """Test saving trace data to API"""
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.raise_for_status = Mock()
    
    trace = configured_tracer.start_trace("test_trace")
    
    # Add some test entries
    @tracer.observe
    def test_function():
        return "test_result"
    
    test_function()
    
    # Save trace
    trace_id, trace_data = trace.save_trace()
    
    # Verify API call
    mock_post.assert_called_once()
    assert mock_post.call_args[1]['json']['trace_id'] == trace_id
    assert mock_post.call_args[1]['json']['name'] == "test_trace"
    assert len(mock_post.call_args[1]['json']['entries']) > 0

def test_condense_trace(configured_tracer):
    """Test trace condensing functionality"""
    trace = configured_tracer.start_trace("test_trace")
    
    # Create sample entries
    entries = [
        {"type": "enter", "function": "test_func", "depth": 0, "timestamp": 1.0},
        {"type": "input", "function": "test_func", "depth": 0, "timestamp": 1.1, "inputs": {"x": 1}},
        {"type": "output", "function": "test_func", "depth": 0, "timestamp": 1.2, "output": "result"},
        {"type": "exit", "function": "test_func", "depth": 0, "timestamp": 1.3},
    ]
    
    condensed = trace.condense_trace(entries)
    
    assert len(condensed) == 1
    assert condensed[0]["function"] == "test_func"
    assert condensed[0]["inputs"] == {"x": 1}
    assert condensed[0]["output"] == "result"
    assert condensed[0]["duration"] == pytest.approx(0.3)

def test_nested_function_depth(configured_tracer):
    """Test depth tracking for nested function calls"""
    @tracer.observe
    def outer():
        @tracer.observe
        def inner():
            pass
        inner()
    
    trace = configured_tracer.start_trace("test_trace")
    outer()
    
    # Verify depths
    print(f"{trace.entries=}")
    depths = [entry.depth for entry in trace.entries]
    assert depths == [0, 1, 1, 2, 2, 1, 1, 0]  # outer(enter), inner(enter), inner(input), inner(output), inner(exit), outer(exit)

def test_error_handling(configured_tracer):
    """Test error handling in traced functions"""
    @tracer.observe
    def failing_function():
        raise ValueError("Test error")
    
    trace = configured_tracer.start_trace("test_trace")
    
    with pytest.raises(ValueError):
        failing_function()
    
    # Verify that exit entry was still recorded
    assert trace.entries[-1].type == "exit"
    assert tracer.depth == 0  # Depth should be reset even after error
