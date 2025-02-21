import csv
import os
from openai import OpenAI


if __name__ == "__main__":

    PATH_TO_CSV = os.path.join(os.path.dirname(__file__), "cstone_data.csv")
    client = OpenAI()


    with open(PATH_TO_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip the header row since we know the structure
        examples = []
        for row in reader:
            docket_id, excerpts, raw_response, quote, is_class_action, note = row
