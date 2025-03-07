import csv
from openai import OpenAI


def extract_incorrect_judgments(file_path, 
                               docket_id_col='docket_id', 
                               excerpts_col='excerpts', 
                               llm_response_col='LLM_raw_response', 
                               label_col='correct', 
                               note_col='RZ_note'):
    """
    Extracts data from a CSV file and prints notes for rows where correct is FALSE.
    
    Args:
        file_path (str): Path to the CSV file
        docket_id_col (str): Column name for docket ID
        excerpts_col (str): Column name for excerpts
        llm_response_col (str): Column name for LLM response
        correct_col (str): Column name for correctness indicator
        note_col (str): Column name for notes
    
    Returns:
        list: List of dictionaries containing the extracted data

        {
        "docket_id": "1234567890",
        "excerpts": "excerpts",
        "LLM_raw_response": "LLM_raw_response",
        "correct": "correct",
        "RZ_note": "RZ_note"
        }
    """
    extracted_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            print(f"CSV Headers: {reader.fieldnames}")
            
            for row in reader:
                # Extract the required columns
                data = {
                    'docket_id': row.get(docket_id_col, ''),
                    'excerpts': row.get(excerpts_col, ''),
                    'LLM_raw_response': row.get(llm_response_col, ''),
                    'correct': row.get(label_col, ''),
                    'RZ_note': row.get(note_col, '')
                }
                
                extracted_data.append(data)
                
                # Print note if correct is FALSE
                if data['correct'].upper() == 'FALSE':
                    print(f"Incorrect judgment for docket {data['docket_id']}:")
                    print(f"Note: {data['RZ_note']}")
                    print("-" * 80)
        
        return extracted_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []





if __name__ == "__main__":
    file_path = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/clh-ma-supplemental-disclosure.csv"
    incorrect_row_data = extract_incorrect_judgments(
        file_path,
        excerpts_col="content",
        llm_response_col="raw_response",
        label_col="supplemental_disclosure_correct",
        note_col="RZ_note"
    )
