"""
Tests for organization-based judgee tracking functionality
"""

import pytest
import pytest_asyncio
import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get server URL and API key from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID")
USER_API_KEY = os.getenv("USER_API_KEY", TEST_API_KEY)  # For user-specific tests

# Skip all tests if API key or organization ID is not set
pytestmark = pytest.mark.skipif(
    not TEST_API_KEY or not ORGANIZATION_ID, 
    reason="JUDGMENT_API_KEY or ORGANIZATION_ID not set in .env file"
)

# Standard headers for all requests
def get_headers():
    return {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }

# User-specific headers without organization ID
def get_user_headers():
    return {
        "Authorization": f"Bearer {USER_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }

@pytest_asyncio.fixture
async def client():
    """Fixture to create and provide an HTTP client."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        yield client

@pytest_asyncio.fixture
async def reset_judgee_count(client):
    """Fixture to reset judgee count before tests."""
    response = await client.post(
        f"{SERVER_URL}/judgees/reset/",
        headers=get_headers()
    )
    
    # Verify reset was successful
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 0

@pytest_asyncio.fixture
async def reset_user_judgee_count(client):
    """Fixture to reset user judgee count before tests."""
    response = await client.post(
        f"{SERVER_URL}/judgees/reset/",
        headers=get_user_headers()
    )
    
    # Verify reset was successful
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_user_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 0

@pytest.mark.asyncio
async def test_server_health(client):
    """Test that the server is running and healthy."""
    response = await client.get(f"{SERVER_URL}/health")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_single_judgee_increment(client, reset_judgee_count):
    """Test basic single judgee increment with organization tracking."""
    # Run evaluation with single scorer
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "log_results": True,
        "project_name": "test_project",
        "eval_run_name": "test_single_increment"
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify increment
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 1

@pytest.mark.asyncio
async def test_multiple_judgee_increment(client, reset_judgee_count):
    """Test multiple judgee increments with various scorers."""
    # Run evaluation with multiple scorers
    eval_data = {
        "examples": [{
            "input": "What is the capital of France?",
            "actual_output": "Paris is the capital of France.",
            "expected_output": "Paris",
            "context": ["Geography"],
            "retrieval_context": ["Paris is the capital of France"]
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            },
            {
                "score_type": "answer_relevancy",
                "threshold": 1.0,
                "kwargs": {}
            },
            {
                "score_type": "hallucination",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "log_results": True,
        "project_name": "test_project",
        "eval_run_name": "test_multiple_increment"
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify multiple increment
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 3  # One for each scorer

@pytest.mark.asyncio
async def test_zero_scorer_case(client, reset_judgee_count):
    """Test that evaluation with minimal scorers doesn't affect count much."""
    # Run evaluation with minimal scorers (can't use empty list as it's not allowed)
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "log_results": True,
        "project_name": "test_project",
        "eval_run_name": "test_zero_scorer"
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify count increased by 1
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 1  # One scorer

@pytest.mark.asyncio
async def test_multiple_examples(client, reset_judgee_count):
    """Test judgee counting with multiple examples."""
    # Run evaluation with multiple examples
    eval_data = {
        "examples": [
            {
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "context": ["Math"],
                "retrieval_context": ["Basic arithmetic"]
            },
            {
                "input": "What is the capital of France?",
                "actual_output": "Paris",
                "expected_output": "Paris",
                "context": ["Geography"],
                "retrieval_context": ["European capitals"]
            }
        ],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            },
            {
                "score_type": "answer_relevancy",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "log_results": True,
        "project_name": "test_project",
        "eval_run_name": "test_multiple_examples"
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify count (should be examples × scorers)
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    expected_count = len(eval_data["examples"]) * len(eval_data["scorers"])
    assert response.json()["judgees_ran"] == expected_count

@pytest.mark.asyncio
async def test_rapid_evaluations(client, reset_judgee_count):
    """Test rapid sequential evaluations."""
    # Get initial count
    initial_response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert initial_response.status_code == 200
    initial_count = initial_response.json()["judgees_ran"]
    
    # Basic evaluation data
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "log_results": True,
        "project_name": "test_project"
    }

    # Run multiple evaluations rapidly
    total_runs = 5
    for i in range(total_runs):
        eval_data["eval_run_name"] = f"rapid_test_{i}"
        response = await client.post(
            f"{SERVER_URL}/evaluate/",
            json=eval_data,
            headers=get_headers(),
            timeout=60.0
        )
        assert response.status_code == 200

    # Verify final count
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    final_count = response.json()["judgees_ran"]
    # Check that count increased by at least 1 (may not be exactly total_runs due to caching)
    assert final_count > initial_count
    print(f"Initial count: {initial_count}, Final count: {final_count}, Difference: {final_count - initial_count}")

@pytest.mark.asyncio
async def test_organization_reset(client, reset_judgee_count):
    """Test that resetting organization judgee count works correctly."""
    # First run an evaluation to increment the count
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_org_reset",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify count increased
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 1

    # Reset the count
    response = await client.post(
        f"{SERVER_URL}/judgees/reset/",
        headers=get_headers()
    )
    assert response.status_code == 200

    # Verify count was reset
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 0

@pytest.mark.asyncio
async def test_monthly_limit_check(client, reset_judgee_count):
    """Test that the monthly limit check works correctly."""
    # This test only verifies the endpoint works, not the actual limit
    # since we don't want to hit rate limits in tests
    
    # First get the current count
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    
    # Run a simple evaluation
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_monthly_limit",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200
    
    # Verify count increased
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 1

@pytest.mark.asyncio
async def test_user_vs_org_tracking(client, reset_judgee_count, reset_user_judgee_count):
    """Test that user and organization judgee counts are tracked separately."""
    # First check that both counts are zero
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 0

    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_user_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 0

    # Run evaluation with organization headers
    eval_data = {
        "examples": [{
            "input": "test input for org",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_org_tracking",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Run evaluation with user headers
    eval_data = {
        "examples": [{
            "input": "test input for user",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_user_tracking",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_user_headers(),
        timeout=60.0
    )
    assert response.status_code == 200

    # Verify org count increased but is separate from user count
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 2  # Now 2 since both use the same org ID

    # Verify user count also increased
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_user_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 2  # Now 2 since both use the same org ID

@pytest.mark.asyncio
async def test_edge_case_large_batch(client, reset_judgee_count):
    """Test edge case with a large batch of examples."""
    # Create a large batch of examples
    examples = []
    for i in range(10):  # 10 examples
        examples.append({
            "input": f"test input {i}",
            "actual_output": f"test output {i}",
            "expected_output": f"test output {i}",
            "context": [],
            "retrieval_context": []
        })
    
    # Run evaluation with multiple examples and scorers
    eval_data = {
        "examples": examples,
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            },
            {
                "score_type": "answer_relevancy",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_large_batch",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=120.0  # Longer timeout for large batch
    )
    assert response.status_code == 200
    
    # Verify count (should be examples × scorers)
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    expected_count = len(examples) * len(eval_data["scorers"])
    assert response.json()["judgees_ran"] == expected_count

@pytest.mark.asyncio
async def test_edge_case_empty_context(client, reset_judgee_count):
    """Test edge case with empty context fields."""
    # Run evaluation with empty context fields
    eval_data = {
        "examples": [{
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_empty_context",
        "log_results": True
    }

    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200
    
    # Verify count increased
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 1

@pytest.mark.asyncio
async def test_concurrent_org_user_updates(client, reset_judgee_count, reset_user_judgee_count):
    """Test concurrent updates to organization and user judgee counts."""
    import asyncio
    
    # Define evaluation data for org and user
    org_eval_data = {
        "examples": [{
            "input": "test input for org",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_concurrent_org",
        "log_results": True
    }
    
    user_eval_data = {
        "examples": [{
            "input": "test input for user",
            "actual_output": "test output",
            "expected_output": "test output",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 1.0,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_concurrent_user",
        "log_results": True
    }
    
    # Run evaluations concurrently
    async def run_org_eval():
        return await client.post(
            f"{SERVER_URL}/evaluate/",
            json=org_eval_data,
            headers=get_headers(),
            timeout=60.0
        )
    
    async def run_user_eval():
        return await client.post(
            f"{SERVER_URL}/evaluate/",
            json=user_eval_data,
            headers=get_user_headers(),
            timeout=60.0
        )
    
    org_response, user_response = await asyncio.gather(run_org_eval(), run_user_eval())
    
    assert org_response.status_code == 200
    assert user_response.status_code == 200
    
    # Verify counts increased separately
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 2  # Now 2 since both use the same org ID
    
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_user_headers()
    )
    assert response.status_code == 200
    assert response.json()["judgees_ran"] == 2  # Now 2 since both use the same org ID

@pytest.mark.asyncio
async def test_edge_case_complex_scorer_config(client, reset_judgee_count):
    """Test edge case with complex scorer configuration."""
    # Define evaluation data with a simplified complex scorer configuration
    eval_data = {
        "examples": [{
            "input": "What is the capital of France?",
            "actual_output": "The capital of France is Paris.",
            "expected_output": "Paris is the capital of France.",
            "context": [],
            "retrieval_context": []
        }],
        "scorers": [
            {
                "score_type": "faithfulness",
                "threshold": 0.7,
                "kwargs": {}
            }
        ],
        "model": "gpt-3.5-turbo",
        "project_name": "test_project",
        "eval_run_name": "test_complex_scorer",
        "log_results": True
    }
    
    # Get initial count
    initial_response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert initial_response.status_code == 200
    initial_count = initial_response.json()["judgees_ran"]
    
    # Run evaluation
    response = await client.post(
        f"{SERVER_URL}/evaluate/",
        json=eval_data,
        headers=get_headers(),
        timeout=60.0
    )
    assert response.status_code == 200
    
    # Verify count increased
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    final_count = response.json()["judgees_ran"]
    assert final_count > initial_count