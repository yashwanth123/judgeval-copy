import pytest
from pydantic import BaseModel, Field
from judgeval.scorers.judgeval_scorers.local_implementations.json_correctness import JsonCorrectnessScorer
from judgeval.constants import APIScorer


class SampleSchema(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    email: str = Field(description="The email address")


class TestJSONCorrectnessScorer:
    def test_init(self):
        # Test initialization with valid threshold and schema
        threshold = 0.7
        schema = SampleSchema
        scorer = JsonCorrectnessScorer(threshold=threshold, json_schema=schema)
        
        assert scorer.threshold == threshold
        assert scorer.json_schema == schema
        assert scorer.score_type == APIScorer.JSON_CORRECTNESS

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        schema = SampleSchema
        with pytest.raises(ValueError):
            JsonCorrectnessScorer(threshold=-0.1, json_schema=schema)
        
        with pytest.raises(ValueError):
            JsonCorrectnessScorer(threshold=1.1, json_schema=schema)

    def test_name_property(self):
        # Test the __name__ property
        schema = SampleSchema
        scorer = JsonCorrectnessScorer(threshold=0.5, json_schema=schema)
        assert scorer.__name__ == "JSON Correctness"
