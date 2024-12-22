import pytest
import os
from judgeval.common.telemetry import (
    get_unique_id,
    telemetry_opt_out,
    blocked_by_firewall,
    capture_evaluation_run,
    capture_metric_type,
    capture_synthesizer_run,
    capture_red_teamer_run,
)
from opentelemetry import trace
from opentelemetry.trace import SpanContext

@pytest.fixture
def clean_env():
    """Clear relevant environment variables before and after tests"""
    original_env = {
        'judgeval_UNIQUE_ID': os.environ.get('judgeval_UNIQUE_ID'),
        'judgeval_TELEMETRY_OPT_OUT': os.environ.get('judgeval_TELEMETRY_OPT_OUT'),
        'ERROR_REPORTING': os.environ.get('ERROR_REPORTING')
    }
    
    for key in original_env:
        if key in os.environ:
            del os.environ[key]
            
    yield
    
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

def test_get_unique_id(clean_env):
    """Test unique ID generation and persistence"""
    id1 = get_unique_id()
    assert id1 is not None
    assert os.environ['judgeval_UNIQUE_ID'] == id1
    
    id2 = get_unique_id()
    assert id2 == id1

def test_telemetry_opt_out(clean_env):
    """Test telemetry opt-out functionality"""
    assert not telemetry_opt_out()
    
    os.environ['judgeval_TELEMETRY_OPT_OUT'] = 'YES'
    assert telemetry_opt_out()
    
    os.environ['judgeval_TELEMETRY_OPT_OUT'] = 'INVALID'
    assert not telemetry_opt_out()

def test_blocked_by_firewall(mocker):
    """Test firewall blocking detection"""
    mock_connection = mocker.patch('socket.create_connection')
    
    # Test successful connection
    mock_connection.return_value = True
    assert not blocked_by_firewall()
    
    # Test failed connection
    mock_connection.side_effect = OSError()
    assert blocked_by_firewall()

@pytest.mark.parametrize("context_manager,args", [
    (capture_evaluation_run, ("benchmark",)),
    (capture_metric_type, ("accuracy",)),
    (capture_synthesizer_run, (10, "method1")),
    (capture_red_teamer_run, ("task1",))
])
def test_telemetry_context_managers_opt_out(clean_env, context_manager, args):
    """Test that context managers respect opt-out setting"""
    os.environ['judgeval_TELEMETRY_OPT_OUT'] = 'YES'
    
    with context_manager(*args) as span:
        assert span is None

def test_capture_synthesizer_run(mocker, clean_env):
    """Test synthesizer run capture with proper span context"""
    # Mock the span context
    mock_span_context = mocker.MagicMock(spec=SpanContext)
    mock_span_context.trace_id = 1234567890123456789
    mock_span_context.span_id = 9876543210
    
    # Mock the span with proper get_span_context method
    mock_span = mocker.MagicMock()
    mock_span.get_span_context.return_value = mock_span_context
    
    # Mock the trace module directly
    mock_trace = mocker.patch('judgeval.common.telemetry.trace')
    mock_tracer = mocker.MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    mock_trace.get_tracer.return_value = mock_tracer
    
    # Use the context manager
    with capture_synthesizer_run(max_generations=5, method="test_method") as span:
        assert span is not None
        assert span == mock_span  # Verify we got our mock span
        mock_span.set_attribute.assert_called_once_with("user.unique_id", mocker.ANY)

    # No need to verify span creation details since we're using the real tracer
