"""
Tests for trace operations in the JudgmentClient.
"""

import pytest
import requests
import os

from judgeval.judgment_client import JudgmentClient

# Get constants from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")

@pytest.mark.traces
class TestTraceOperations:
    """Tests for trace-related operations."""
    
    def test_fetch_traces_by_time_period(self, client: JudgmentClient):
        """Test successful cases with different time periods."""
        for hours in [1, 3, 6, 12, 24, 72, 168]:
            response = requests.post(
                f"{SERVER_URL}/traces/fetch_by_time_period/",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Organization-Id": ORGANIZATION_ID
                },
                json={"hours": hours}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_fetch_traces_invalid_period(self, client: JudgmentClient):
        """Test invalid time periods."""
        for hours in [0, 2, 4]:
            response = requests.post(
                f"{SERVER_URL}/traces/fetch_by_time_period/",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Organization-Id": ORGANIZATION_ID
                },
                json={"hours": hours}
            )
            assert response.status_code == 400

    def test_fetch_traces_missing_api_key(self, client: JudgmentClient):
        """Test missing API key scenario."""
        response = requests.post(
            f"{SERVER_URL}/traces/fetch_by_time_period/",
            headers={
                "Content-Type": "application/json",
                "X-Organization-Id": ORGANIZATION_ID
            },
            json={"hours": 12}
        )
        assert response.status_code in [401, 403] 