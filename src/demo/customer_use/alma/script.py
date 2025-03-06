import os
import csv

# Define the paths to the folders
draft_folder = 'alma_anonymized_draft'
final_folder = 'alma_anonymized_final'
output_file = 'data.csv'

# Get the list of files in the draft folder
draft_files = os.listdir(draft_folder)

# Open the CSV file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['ID', 'draft text', 'final text'])
    
    # Iterate over the files in the draft folder
    for draft_filename in draft_files:
        # Construct the full paths to the draft and final text files
        draft_path = os.path.join(draft_folder, draft_filename)
        final_path = os.path.join(final_folder, draft_filename)
        
        # Read the content of the draft text file
        with open(draft_path, 'r') as draft_file:
            draft_text = draft_file.read()
        
        # Read the content of the final text file
        with open(final_path, 'r') as final_file:
            final_text = final_file.read()
        
        # Write the ID, draft text, and final text to the CSV file
        writer.writerow([draft_filename, draft_text, final_text])
