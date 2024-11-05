"""
Custom Scorer class
"""
from typing import Generic, TypeVar, Optional, Any, Dict, Union, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum

from judgeval.data.example import Example


class CustomScorer(BaseModel, ABC):
    """
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class. This is best used for special use cases
    where none of Judgment's scorers are suitable.
    """
    score_type: str
    
    @abstractmethod
    def score(self, example: Example, *args, **kwargs) -> float:
        """Method that must be implemented to measure test results"""
        raise NotImplementedError("You must implement the `score` method in your custom scorer")

    @abstractmethod
    async def a_score(self, example: Example, *args, **kwargs) -> float:
        raise NotImplementedError("You must implement the `a_score` method in your custom scorer") 
    
    @abstractmethod
    def success_check(self) -> bool:
        raise NotImplementedError("You must implement the `passes` method in your custom scorer")
