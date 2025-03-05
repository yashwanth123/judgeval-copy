"""
Test runner for dataset operations.
Uses pytest fixtures from conftest.py
"""

import pytest
from test_dataset_operations import TestDatasetOperations

# Direct test functions that use the fixture from conftest.py
def test_dataset(client):
    """Run the dataset creation and manipulation test."""
    TestDatasetOperations().test_dataset(client)

def test_pull_all_user_dataset_stats(client):
    """Run the test for pulling dataset statistics."""
    TestDatasetOperations().test_pull_all_user_dataset_stats(client)

def test_edit_dataset(client):
    """Run the test for editing datasets."""
    TestDatasetOperations().test_edit_dataset(client)

def test_export_jsonl(client, random_name):
    """Run the test for JSONL export."""
    TestDatasetOperations().test_export_jsonl(client, random_name) 