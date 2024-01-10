import os
import docx
import shutil
import random


import os
import random

import os
import random

"""
This code fragment is repsonsible for converting data into the appropriate format such that the
hololens can display it
"""

def merge_ply_files(file_path, output_file='merged.ply'):
    ply_files = sorted([f for f in os.listdir(file_path) if f.endswith('.ply')])

    # Open the output file for writing (creating if it doesn't exist)
    with open(output_file, 'wb') as merged_file:
        # Iterate through each PLY file
        counter = 0
        for ply_file in ply_files:
            counter += 1
            ply_file_path = os.path.join(file_path, ply_file)

            # Read the PLY file
            with open(ply_file_path, 'rb') as f:
                # Skip the original header
                while True:
                    line = f.readline()
                    if line == b'end_header\n':
                        break

                # Read all data lines into a list
                data_lines = f.readlines()

            # Shuffle the data lines
            random.shuffle(data_lines)

            # Write a random subset (e.g., 6000 lines) of the shuffled data lines to the new file
            subset_size = min(6000, len(data_lines))
            merged_file.writelines(data_lines[:subset_size])

            # Add the "end_cloud" string to the merged file
            merged_file.write(b'end_cloud\n')

        print(counter*6000)

            
def convert_docx_to_txt(docx_path, txt_path):
    # Load the Docx file
    doc = docx.Document(docx_path)

    # Extract text from paragraphs
    text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    # Write the text content to the TXT file
    with open(txt_path, 'w') as txt_file:
        txt_file.write(text_content)

def process_meta_folder(meta_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each Docx file in the meta folder
    for index, docx_file in enumerate(sorted(os.listdir(meta_folder)), start=1):
        if docx_file.endswith('.docx'):
            # Generate the corresponding TXT file name
            txt_filename = f"meta_{index:06d}.txt"

            # Construct the full paths for source and destination
            docx_path = os.path.join(meta_folder, docx_file)
            txt_path = os.path.join(output_folder, txt_filename)

            # Convert the Docx file to TXT
            convert_docx_to_txt(docx_path, txt_path)


if __name__ == "__main__":
    file_path = "/path_to_ply_file"
    merge_ply_files(file_path)
    #meta_folder_path = "/Users/lukasschuepp/framework/hand_data/data/predicted_data/meta"
    #output_folder_path = "/Users/lukasschuepp/framework/hand_data/data/predicted_data/final"

    #process_meta_folder(meta_folder_path, output_folder_path)


    
