"""
Shared fixtures and configuration for E2E tests.
"""

import os
import pytest
import random
import string
import logging
from dotenv import load_dotenv

from judgeval.judgment_client import JudgmentClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")

if not API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set", allow_module_level=True)

@pytest.fixture(scope="session")
def client() -> JudgmentClient:
    """Create a single JudgmentClient instance for all tests."""
    return JudgmentClient(judgment_api_key=API_KEY, organization_id=ORGANIZATION_ID)

@pytest.fixture
def random_name() -> str:
    """Generate a random name for test resources."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

def pytest_configure(config):
    """Add markers for test categories."""
    config.addinivalue_line("markers", "basic: mark test as testing basic functionality")
    config.addinivalue_line("markers", "advanced: mark test as testing advanced features")
    config.addinivalue_line("markers", "custom: mark test as testing custom components")
    config.addinivalue_line("markers", "traces: mark test as testing trace operations") 