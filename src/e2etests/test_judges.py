"""
Test runner for custom judge implementations.
Uses pytest fixtures from conftest.py
"""

import pytest
import os
from test_custom_judges import TestCustomJudges

@pytest.mark.skipif(not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="VertexAI credentials not configured")
def test_custom_judge_vertexai(client):
    """Run the VertexAI custom judge test."""
    TestCustomJudges().test_custom_judge_vertexai(client) 