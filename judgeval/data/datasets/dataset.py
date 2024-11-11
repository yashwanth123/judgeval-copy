import ast
import json
from dataclasses import dataclass, field
import os
from typing import List, Optional, Union, Literal

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

    def add_examples_from_json(
        self,
        file_path: str,
        input_key_name: str,
        actual_output_key_name: str,
        expected_output_key_name: Optional[str] = None,
        context_key_name: Optional[str] = None,
        retrieval_context_key_name: Optional[str] = None,
        ) -> None:
        """
        Load examples from a JSON file.

        This method reads a JSON file containing a list of objects, each representing an example. 
        It extracts the necessary information based on specified key names and creates `Example` 
        objects to add to the `EvalDataset`.

        Args:
            file_path (str): Path to the JSON file containing the examples.
            input_key_name (str): The key name in the JSON objects corresponding to the input for the example.
            actual_output_key_name (str): The key name in the JSON objects corresponding to the actual output for the example.
            expected_output_key_name (str, optional): The key name in the JSON objects corresponding to the expected output for the example. Defaults to None.
            context_key_name (str, optional): The key name in the JSON objects corresponding to the context for the example. Defaults to None.
            retrieval_context_key_name (str, optional): The key name in the JSON objects corresponding to the retrieval context for the example. Defaults to None.

        Returns:
            None: The method adds examples to the `EvalDataset` instance but does not return anything.

        Raises:
            FileNotFoundError: If the JSON file specified by `file_path` cannot be found.
            ValueError: If the JSON file is not valid or if required keys (input and actual output) are missing in one or more JSON objects.

        Note:
            The JSON file should be structured as a list of objects, with each object containing the required keys. The method assumes the file format and keys are correctly defined and present.
        """
        try:
            with open(file_path, "r") as file:
                json_list = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON file.")

        for json_obj in json_list:
            if input_key_name not in json_obj:
                raise ValueError(f"Required field '{input_key_name}' is missing in one or more JSON objects")
            if actual_output_key_name not in json_obj:
                raise ValueError(f"Required field '{actual_output_key_name}' is missing in one or more JSON objects")

            input = json_obj[input_key_name]
            actual_output = json_obj[actual_output_key_name]
            expected_output = json_obj.get(expected_output_key_name)
            context = json_obj.get(context_key_name)
            retrieval_context = json_obj.get(retrieval_context_key_name)

            self.add_example(
                Example(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                )
            )

    def add_examples_from_csv(
        self, 
        file_path: str,
        input_col_name: str,
        actual_output_col_name: str,
        expected_output_col_name: Optional[str] = None,
        context_col_name: Optional[str] = None,
        context_col_delimiter: str = ";",
        retrieval_context_col_name: Optional[str] = None,
        retrieval_context_col_delimiter: str = ";",
        additional_metadata_col_name: Optional[str] = None,
        ) -> None:
        """
        Load examples from a CSV file.

        This method reads a CSV file, extracting example data based on specified column names. It creates Example objects for each row in the CSV and adds them to the Dataset instance. The context data, if provided, is expected to be a delimited string in the CSV, which this method will parse into a list.

        Args:
            file_path (str): Path to the CSV file containing the examples.
            input_col_name (str): The column name in the CSV corresponding to the input for the example.
            actual_output_col_name (str): The column name in the CSV corresponding to the actual output for the example.
            expected_output_col_name (str, optional): The column name in the CSV corresponding to the expected output for the example. Defaults to None.
            context_col_name (str, optional): The column name in the CSV corresponding to the context for the example. Defaults to None.
            context_delimiter (str, optional): The delimiter used to separate items in the context list within the CSV file. Defaults to ';'.
            retrieval_context_col_name (str, optional): The column name in the CSV corresponding to the retrieval context for the example. Defaults to None.
            retrieval_context_delimiter (str, optional): The delimiter used to separate items in the retrieval context list within the CSV file. Defaults to ';'.
            additional_metadata_col_name (str, optional): The column name in the CSV corresponding to additional metadata for the example. Defaults to None.

        Returns:
            None: The method adds examples to the Dataset instance but does not return anything.

        Raises:
            FileNotFoundError: If the CSV file specified by `file_path` cannot be found.
            pd.errors.EmptyDataError: If the CSV file is empty.
            KeyError: If one or more specified columns are not found in the CSV file.

        Note:
            The CSV file is expected to contain columns as specified in the arguments. Each row in the file represents a single example. The method assumes the file is properly formatted and the specified columns exist. For context data represented as lists in the CSV, ensure the correct delimiter is specified.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install pandas to use this method. 'pip install pandas'")

        def fetch_col_data(df: pd.DataFrame, col_name: str, default=None):
            """
            Fetches column data from a DataFrame.

            If the column name is not found, the method returns a list of default values.
            """
            return df[col_name].values if col_name in df.columns else [default] * len(df)
        
        df = pd.read_csv(file_path)

        inputs = fetch_col_data(df, input_col_name)
        actual_outputs = fetch_col_data(df, actual_output_col_name)
        expected_outputs = fetch_col_data(df, expected_output_col_name, default=None)
        contexts = [context.split(context_col_delimiter) if context else [] \
                    for context in fetch_col_data(df, context_col_name, default="")]
        retrieval_contexts = [
            retrieval_context.split(retrieval_context_col_delimiter) if retrieval_context else []
            for retrieval_context in fetch_col_data(df, retrieval_context_col_name, default="")
        ]
        additional_metadatas = [
            ast.literal_eval(metadata) if metadata else None
            for metadata in fetch_col_data(df, additional_metadata_col_name, default="")
        ]

        for input, actual_output, expected_output, \
            context, retrieval_context, additional_metadata,\
        in zip(
            inputs,
            actual_outputs,
            expected_outputs,
            contexts,
            retrieval_contexts,
            additional_metadatas,
        ):
            self.add_example(
                Example(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    additional_metadata=additional_metadata,
                )
            )

    def add_example(self, e: Example) -> None:
        self.examples.extend(e)
        # TODO if we need to add rank, then we need to do it here

    def add_ground_truth(self, g: GroundTruthExample) -> None:
        self.ground_truths.extend(g)
    
    def save_as(self, file_type: Literal["json", "csv"], dir_path: str):
        """
        Saves the dataset as a file. Save both the ground truths and examples.

        Args:
            file_type (Literal["json", "csv"]): The file type to save the dataset as.
            dir_path (str): The directory path to save the file to.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if file_type == "json":
            pass 
        elif file_type == "csv":
            pass
        else:
            ACCEPTABLE_FILE_TYPES = ["json", "csv"]
            raise TypeError(f"Invalid file type: {file_type}. Please choose from {ACCEPTABLE_FILE_TYPES}")
        
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
