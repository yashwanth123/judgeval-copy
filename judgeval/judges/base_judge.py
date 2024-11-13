from abc import ABC, abstractmethod
from typing import Optional, List


class judgevalJudge(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = model_name
        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

    @abstractmethod
    async def a_generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

    def batch_generate(self, *args, **kwargs) -> List[str]:
        """Runs the model to output LLM responses.

        Returns:
            A list of strings.
        """
        raise AttributeError

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        pass
    