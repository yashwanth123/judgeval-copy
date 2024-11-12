import ast
import csv
import datetime 
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
import uuid
from dataclasses import dataclass, field
import os
from typing import List, Optional, Union, Literal

from judgeval.constants import JUDGMENT_DATASETS_API_URL
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

    def push(self, alias: str, overwrite: Optional[bool] = None) -> None:
        """
        Pushes the dataset to Judgment platform

        Mock request:
        {
            "alias": alias,
            "ground_truths": [...],
            "examples": [...],
            "overwrite": overwrite
        } ==>
        {
            "_alias": alias,
            "_id": "..."  # ID of the dataset
        }
        """
        # Make a POST request to the Judgment API to create a new dataset

        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Pushing [rgb(106,0,255)]'{alias}' to Judgment...",
                total=100,
            )
            content = {
                    "alias": alias,
                    "ground_truths": [g.to_dict() for g in self.ground_truths],
                    "examples": [e.to_dict() for e in self.examples],
                    "overwrite": overwrite,
                    "user_id": str(uuid.uuid4())  # TODO fix
                }
            try:
                response = requests.post(
                    JUDGMENT_DATASETS_API_URL, 
                    json=content
                ) 
                if response.status_code == 500:
                    content = response.json()
                    print("Error details:", content.get("message"))
                    return False
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                if response.status_code == 422:
                    print("Validation error details:", err.response.json())
                else:
                    print("HTTP error occurred:", err)
            
            payload = response.json()
            self._alias = payload.get("_alias")
            self._id = payload.get("_id")
            progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
                )
            return True
        
    def pull(self, alias: str):
        """
        Pulls the dataset from Judgment platform

        Mock request:
        {
            "alias": alias,
            "user_id": user_id
        } 
        ==>
        {
            "ground_truths": [...],
            "examples": [...],
            "_alias": alias,
            "_id": "..."  # ID of the dataset
        }
        """
        # Make a GET request to the Judgment API to get the dataset

        with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                TextColumn("[progress.description]{task.description}"),
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Pulling [rgb(106,0,255)]'{alias}'[/rgb(106,0,255)] from Judgment...",
                    total=100,
                )
                response = requests.get(
                    JUDGMENT_DATASETS_API_URL, 
                    params={"alias": alias}  # TODO add user id
                ) 

                response.raise_for_status()
                
                payload = response.json()
                self.ground_truths = [GroundTruthExample(**g) for g in payload.get("ground_truths", [])]
                self.examples = [Example(**e) for e in payload.get("examples", [])]
                self._alias = payload.get("_alias")
                self._id = payload.get("_id")
                progress.update(
                    task_id,
                    description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done!)",
                )

    def add_from_json(
        self,
        file_path: str,
        ) -> None:
        """
        Adds examples and ground truths from a JSON file.

        The format of the JSON file is expected to be a dictionary with two keys: "examples" and "ground_truths". 
        The value of each key is a list of dictionaries, where each dictionary represents an example or ground truth. 
        """
        try:
            with open(file_path, "r") as file:
                payload = json.load(file)
                examples = payload.get("examples", [])
                ground_truths = payload.get("ground_truths", [])
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON file.")

        new_examples = [Example(**e) for e in examples]
        for e in new_examples:
            self.add_example(e)

        new_ground_truths = [GroundTruthExample(**g) for g in ground_truths]
        for g in new_ground_truths:
            self.add_ground_truth(g)
  
    def add_from_csv(
        self, 
        file_path: str,
        ) -> None:
        """
        Add Examples and GroundTruthExamples from a CSV file.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pandas to use this method. 'pip install pandas'"
            )
        
        df = pd.read_csv(file_path)
        """
        Expect the CSV to have headers

        "input", "actual_output", "expected_output", "context", \
        "retrieval_context", "additional_metadata", "tools_called", \
        "expected_tools", "name", "comments", "source_file", "example"

        We want to collect the examples and ground truths separately which can
        be determined by the "example" column. If the value is True, then it is an
        example, otherwise it is a ground truth.
        """
        examples, ground_truths = [], []

        for _, row in df.iterrows():
            data = {
                "input": row["input"],
                "actual_output": row["actual_output"] if pd.notna(row["actual_output"]) else None,
                "expected_output": row["expected_output"] if pd.notna(row["expected_output"]) else None,
                "context": row["context"].split(";") if pd.notna(row["context"]) else [],
                "retrieval_context": row["retrieval_context"].split(";") if pd.notna(row["retrieval_context"]) else [],
                "additional_metadata": ast.literal_eval(row["additional_metadata"]) if pd.notna(row["additional_metadata"]) else dict(),
                "tools_called": row["tools_called"].split(";") if pd.notna(row["tools_called"]) else [],
                "expected_tools": row["expected_tools"].split(";") if pd.notna(row["expected_tools"]) else [],
            }

            if row["example"]:
                data["name"] = row["name"] if pd.notna(row["name"]) else None
                # every Example has `input` and `actual_output` fields
                if data["input"] is not None and data["actual_output"] is not None:
                    e = Example(**data)
                    examples.append(e)
                else:
                    raise ValueError("Every example must have an 'input' and 'actual_output' field.")
            else:
                data["comments"] = row["comments"] if pd.notna(row["comments"]) else None
                data["source_file"] = row["source_file"] if pd.notna(row["source_file"]) else None
                # every GroundTruthExample has `input` field
                if data["input"] is not None:
                    g = GroundTruthExample(**data)
                    ground_truths.append(g)
                else:
                    raise ValueError("Every ground truth must have an 'input' field.")

        for e in examples:
            self.add_example(e)

        for g in ground_truths:
            self.add_ground_truth(g)

    def add_example(self, e: Example) -> None:
        self.examples = self.examples + [e]
        # TODO if we need to add rank, then we need to do it here

    def add_ground_truth(self, g: GroundTruthExample) -> None:
        self.ground_truths = self.ground_truths + [g]
    
    def save_as(self, file_type: Literal["json", "csv"], dir_path: str, save_name: str = None) -> None:
        """
        Saves the dataset as a file. Save both the ground truths and examples.

        Args:
            file_type (Literal["json", "csv"]): The file type to save the dataset as.
            dir_path (str): The directory path to save the file to.
            save_name (str, optional): The name of the file to save. Defaults to None.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if save_name is None else save_name
        complete_path = os.path.join(dir_path, f"{file_name}.{file_type}")
        if file_type == "json":
            with open(complete_path, "w") as file:
                json.dump(
                    {
                        "ground_truths": [g.to_dict() for g in self.ground_truths],
                        "examples": [e.to_dict() for e in self.examples],
                    },
                    file,
                    indent=4,
                )
        elif file_type == "csv":
            with open(complete_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "input", "actual_output", "expected_output", "context", \
                    "retrieval_context", "additional_metadata", "tools_called", \
                    "expected_tools", "name", "comments", "source_file", "example"
                ])
                for e in self.examples:
                    writer.writerow(
                        [
                            e.input,
                            e.actual_output,
                            e.expected_output,
                            ";".join(e.context),
                            ";".join(e.retrieval_context),
                            e.additional_metadata,
                            ";".join(e.tools_called),
                            ";".join(e.expected_tools),
                            e.name,
                            None,  # Example does not have comments
                            None,  # Example does not have source file
                            True,  # Adding an Example
                        ]
                    )
                
                for g in self.ground_truths:
                    writer.writerow(
                        [
                            g.input,
                            g.actual_output,
                            g.expected_output,
                            ";".join(g.context),
                            ";".join(g.retrieval_context),
                            g.additional_metadata,
                            ";".join(g.tools_called),
                            ";".join(g.expected_tools),
                            None,  # GroundTruthExample does not have name
                            g.comments,
                            g.source_file,
                            False,  # Adding a GroundTruthExample, not an Example
                        ]
                    )
        else:
            ACCEPTABLE_FILE_TYPES = ["json", "csv"]
            raise TypeError(f"Invalid file type: {file_type}. Please choose from {ACCEPTABLE_FILE_TYPES}")
        
    def __iter__(self):
        return iter(self.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __str__(self):
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
    dataset.add_example(Example(input="input 1", actual_output="output 1"))
    print(dataset)

    file_path = "/Users/alexshan/Desktop/judgment_labs/judgeval/judgeval/data/datasets/20241111_175859.csv"
    dataset.add_from_csv(file_path)

    dataset.push(alias="test_dataset_1", overwrite=True)
