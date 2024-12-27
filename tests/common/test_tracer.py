import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from openai import OpenAI
from together import Together
from anthropic import Anthropic

from judgeval.common.tracer import Tracer, TraceEntry, TraceClient, wrap

@pytest.fixture
def tracer():
    """Provide a configured tracer instance"""
    return Tracer(api_key="test_api_key")

@pytest.fixture
def trace_client(tracer):
    """Provide a trace client instance"""
    with tracer.trace("test_trace") as client:
        yield client

def test_tracer_singleton():
    """Test that Tracer maintains singleton pattern"""
    tracer1 = Tracer(api_key="test1")
    tracer2 = Tracer(api_key="test2")
    assert tracer1 is tracer2
    assert tracer1.api_key == "test2"  # Should have new api_key

def test_tracer_requires_api_key():
    """Test that Tracer requires an API key"""
    # Clear any existing singleton instance first
    Tracer._instance = None
    
    with pytest.raises(ValueError):
        tracer = Tracer(api_key=None)
        print(tracer.api_key)

def test_trace_entry_print(capsys):
    """Test TraceEntry print formatting"""
    entries = [
        TraceEntry(type="enter", function="test_func", depth=1, message="test", timestamp=0),
        TraceEntry(type="exit", function="test_func", depth=1, message="test", timestamp=0, duration=0.5),
        TraceEntry(type="output", function="test_func", depth=1, message="test", timestamp=0, output="result"),
        TraceEntry(type="input", function="test_func", depth=1, message="test", timestamp=0, inputs={"arg": 1}),
    ]
    
    expected_outputs = [
        "  → test_func (trace: test)\n",
        "  ← test_func (0.500s)\n",
        "  Output: result\n",
        "  Input: {'arg': 1}\n",
    ]
    
    for entry, expected in zip(entries, expected_outputs):
        entry.print_entry()
        captured = capsys.readouterr()
        assert captured.out == expected

def test_trace_entry_to_dict():
    """Test TraceEntry serialization"""
    # Test basic serialization
    entry = TraceEntry(
        type="enter",
        function="test_func",
        depth=1,
        message="test",
        timestamp=0
    )
    data = entry.to_dict()
    assert data["type"] == "enter"
    assert data["function"] == "test_func"

    # Test with non-serializable output
    class NonSerializable:
        pass
    
    entry = TraceEntry(
        type="output",
        function="test_func",
        depth=1,
        message="test",
        timestamp=0,
        output=NonSerializable()
    )
    
    with pytest.warns(UserWarning):
        data = entry.to_dict()
        assert data["output"] is None

def test_trace_client_span(trace_client):
    """Test span context manager"""
    initial_entries = len(trace_client.entries)  # Get initial count
    
    with trace_client.span("test_span") as span:
        assert trace_client._current_span == "test_span"
        assert len(trace_client.entries) == initial_entries + 1  # Compare to initial count
    
    assert len(trace_client.entries) == initial_entries + 2  # Account for both enter and exit
    assert trace_client.entries[-1].type == "exit"
    assert trace_client._current_span == "test_trace"

def test_trace_client_nested_spans(trace_client):
    """Test nested spans maintain proper depth"""
    with trace_client.span("outer"):
        assert trace_client.tracer.depth == 2  # 1 for trace + 1 for span
        with trace_client.span("inner"):
            assert trace_client.tracer.depth == 3
        assert trace_client.tracer.depth == 2
    assert trace_client.tracer.depth == 1

def test_record_input_output(trace_client):
    """Test recording inputs and outputs"""
    with trace_client.span("test_span"):
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")
    
    # Filter entries to only include those for the current span
    entries = [e.type for e in trace_client.entries if e.function == "test_span"]
    assert entries == ["enter", "input", "output", "exit"]

def test_condense_trace(trace_client):
    """Test trace condensing functionality"""
    entries = [
        {"type": "enter", "function": "test_func", "depth": 0, "timestamp": 1.0},
        {"type": "input", "function": "test_func", "depth": 1, "timestamp": 1.1, "inputs": {"x": 1}},
        {"type": "output", "function": "test_func", "depth": 1, "timestamp": 1.2, "output": "result"},
        {"type": "exit", "function": "test_func", "depth": 0, "timestamp": 2.0},
    ]
    
    condensed = trace_client.condense_trace(entries)
    assert len(condensed) == 1
    assert condensed[0]["function"] == "test_func"
    assert condensed[0]["depth"] == 1
    assert condensed[0]["inputs"] == {"x": 1}
    assert condensed[0]["output"] == "result"
    assert condensed[0]["duration"] == 1.0

@patch('requests.post')
def test_save_trace(mock_post, trace_client):
    """Test saving trace data"""
    mock_post.return_value.raise_for_status = Mock()
    
    with trace_client.span("test_span"):
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")
    
    trace_id, data = trace_client.save()
    
    assert mock_post.called
    assert data["trace_id"] == trace_client.trace_id
    assert data["name"] == "test_trace"
    assert len(data["entries"]) > 0
    assert isinstance(data["created_at"], str)
    assert isinstance(data["duration"], float)

def test_observe_decorator(tracer):
    """Test the @tracer.observe decorator"""
    @tracer.observe
    def test_function(x, y):
        return x + y
    
    with tracer.trace("test_trace"):
        result = test_function(1, 2)
    
    assert result == 3

def test_observe_decorator_with_error(tracer):
    """Test decorator error handling"""
    @tracer.observe
    def failing_function():
        raise ValueError("Test error")
    
    with tracer.trace("test_trace"):
        with pytest.raises(ValueError):
            failing_function()

@patch('requests.post')
def test_wrap_openai(mock_post, tracer):
    """Test wrapping OpenAI client"""
    client = OpenAI()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    client.chat.completions.create = MagicMock(return_value=mock_response)
    
    wrapped_client = wrap(client)
    
    with tracer.trace("test_trace"):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
    
    assert response == mock_response

@patch('requests.post')
def test_wrap_anthropic(mock_post, tracer):
    """Test wrapping Anthropic client"""
    client = Anthropic()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
    client.messages.create = MagicMock(return_value=mock_response)
    
    wrapped_client = wrap(client)
    
    with tracer.trace("test_trace"):
        response = wrapped_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "test"}]
        )
    
    assert response == mock_response

def test_wrap_unsupported_client(tracer):
    """Test wrapping unsupported client type"""
    class UnsupportedClient:
        pass
    
    with pytest.raises(ValueError):
        wrap(UnsupportedClient())
