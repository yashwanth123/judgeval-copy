from typing import List, Optional, Union
from dataclasses import dataclass, field

from judgeval.data.datasets.ground_truth import GroundTruthExample
from judgeval.data.datasets.utils import ground_truths_to_examples, examples_to_ground_truths
from judgeval.data import Example


@dataclass
class EvalDataset:
    ground_truths: List[GroundTruthExample]
    examples: List[Example]
    _alias: Union[str, None] = field(default=None)
    _id: Union[str, None] = field(default=None)

    def __init__(self, 
                 ground_truths: List[GroundTruthExample] = [], 
                 examples: List[Example] = [],
                 ):
        self.ground_truths = ground_truths
        self.examples = examples
        self._alias = None
        self._id = None

    def push(self):
        """
        Pushes the dataset to Judgment platform
        """
        raise NotImplementedError

    def pull(self):
        """
        Pulls the dataset from Judgment platform
        """
        raise NotImplementedError

    def add_examples_from(self):
        """
        Load more examples from a file
        """
        # support csv, json
        raise NotImplementedError

    def add_example(self, e: Example) -> None:
        self.examples.extend(e)
        # TODO if we need to add rank, then we need to do it here

    def add_ground_truth(self, g: GroundTruthExample) -> None:
        self.ground_truths.extend(g)
    
    def save_as(self):
        """
        Saves the dataset as a file
        """
        raise NotImplementedError

    def __iter__(self):
        return iter(self.examples)
    
    def __len__(self):
        return len(self.examples)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ground_truths={self.ground_truths}, "
            f"examples={self.examples}, "
            f"_alias={self._alias}, "
            f"_id={self._id}"
            f")"
        )
    

if __name__ == "__main__":

    dataset = EvalDataset()
    print(dataset)
