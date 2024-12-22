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

@pytest.fixture
def mock_telemetry(mocker):
    """Fixture to mock OpenTelemetry setup and provide mock span for testing"""
    # Create the mock chain
    mock_span = mocker.MagicMock()
    mock_context_manager = mocker.MagicMock()
    mock_context_manager.__enter__.return_value = mock_span
    mock_tracer = mocker.MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_context_manager
    
    # Mock OpenTelemetry setup
    mocker.patch('opentelemetry.trace.get_tracer', return_value=mock_tracer)
    mocker.patch('opentelemetry.trace.set_tracer_provider')
    mocker.patch('opentelemetry.trace.get_tracer_provider')
    
    # Reload telemetry module to recreate tracer with our mock
    import importlib
    import judgeval.common.telemetry
    importlib.reload(judgeval.common.telemetry)
    
    # Return the mock objects for test assertions
    return mocker.MagicMock(
        span=mock_span,
        context_manager=mock_context_manager,
        tracer=mock_tracer
    )

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

def test_capture_synthesizer_run(mock_telemetry, mocker, clean_env):
    """Test synthesizer run capture"""
    with capture_synthesizer_run(max_generations=5, method="test_method") as span:
        # Verify we got the mock span
        assert span is mock_telemetry.span
        
        # Verify the span was created with correct parameters
        mock_telemetry.tracer.start_as_current_span.assert_called_once_with(
            "Invoked synthesizer (5) | Method: test_method"
        )
        
        # Verify attributes were set
        mock_telemetry.span.set_attribute.assert_called_once_with("user.unique_id", mocker.ANY)

def test_capture_red_teamer_run(mock_telemetry, mocker, clean_env):
    """Test red teamer run capture"""
    with capture_red_teamer_run(task="test_task") as span:
        # Verify we got the mock span
        assert span is mock_telemetry.span
        
        # Verify the span was created with correct parameters
        mock_telemetry.tracer.start_as_current_span.assert_called_once_with(
            "Invoked red teamer: (test_task)"
        )
        
        # Verify attributes were set
        mock_telemetry.span.set_attribute.assert_called_once_with("user.unique_id", mocker.ANY)

def test_capture_metric_type(mock_telemetry, mocker, clean_env):
    """Test metric type capture"""
    with capture_metric_type(metric_name="test_metric") as span:
        # Verify we got the mock span
        assert span is mock_telemetry.span
        
        # Verify the span was created with correct parameters
        mock_telemetry.tracer.start_as_current_span.assert_called_once_with(
            "test_metric"
        )
        
        # Verify attributes were set
        mock_telemetry.span.set_attribute.assert_called_once_with("user.unique_id", mocker.ANY)

def test_capture_evaluation_run(mock_telemetry, mocker, clean_env):
    """Test evaluation run capture"""
    with capture_evaluation_run(type="test_type") as span:
        # Verify we got the mock span
        assert span is mock_telemetry.span
        
        # Verify the span was created with correct parameters
        mock_telemetry.tracer.start_as_current_span.assert_called_once_with(
            "Evaluation run: test_type"
        )
        
        # Verify attributes were set
        mock_telemetry.span.set_attribute.assert_called_once_with("user.unique_id", mocker.ANY)

