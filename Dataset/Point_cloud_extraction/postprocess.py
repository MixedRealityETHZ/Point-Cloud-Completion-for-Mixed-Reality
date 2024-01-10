import os
import shutil
import json

# Define the root directory containing the four folders
root_directory = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/raw_data'
category_id = '10'
category = "hand"

# Function to rename files with '.pcd' extension in the 'complete' folder
def rename_complete_files(folder_path, day_number):
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.pcd'):
                new_filename = f'{day_number}{filename}'
                os.rename(os.path.join(foldername, filename), os.path.join(foldername, new_filename))

# Function to rename folders in the 'partial' folder
def rename_partial_folders(folder_path, day_number):
    for foldername in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, foldername)):
            new_foldername = f'{day_number}{foldername}'
            os.rename(os.path.join(folder_path, foldername), os.path.join(folder_path, new_foldername))

# Function to create a subfolder "10" and move contents to it
def create_and_move_to_10(folder_path):
    folder_10_path = os.path.join(folder_path, category_id)
    os.makedirs(folder_10_path, exist_ok=True)
    for item in os.listdir(folder_path):
        if item != category_id:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                shutil.move(item_path, os.path.join(folder_10_path, item))
            elif os.path.isdir(item_path):
                shutil.move(item_path, os.path.join(folder_10_path, item))

# Loop through the four folders
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        if folder_name.startswith("Hand_Data"):
            day_number = folder_name.split('-')[1]  # Extract day number from folder name
            complete_folder = os.path.join(folder_path, 'complete')
            partial_folder = os.path.join(folder_path, 'partial')

            # Rename files in the 'complete' folder
            rename_complete_files(complete_folder, day_number)

            # Rename folders in the 'partial' folder
            rename_partial_folders(partial_folder, day_number)
            # Create a subfolder "10" and move contents in 'complete' and 'partial' folders
            create_and_move_to_10(complete_folder)
            create_and_move_to_10(partial_folder)

# Define the destination directory
destination_folder = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/our_data/PCN/'

# Create "train," "valid," and "test" folders within the destination directory
train_folder = os.path.join(destination_folder, "train")
valid_folder = os.path.join(destination_folder, "valid")
test_folder = os.path.join(destination_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Move the contents of "Hand_Data_7-14-1-2" to "train"
if os.path.exists(os.path.join(root_directory, "Hand_Data_7-14-1-2")):
    src_dir = os.path.join(root_directory, "Hand_Data_7-14-1-2")
    dest_dir = train_folder
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        shutil.move(item_path, os.path.join(dest_dir, item))

# Move the contents of "Hand_Data_9-10-1-2" to "train"
if os.path.exists(os.path.join(root_directory, "Hand_Data_9-10-1-2")):
    src_dir = os.path.join(root_directory, "Hand_Data_9-10-1-2")
    dest_dir = train_folder
    for complete_partial in ["complete", "partial"]:
        for item in os.listdir(os.path.join(src_dir, complete_partial, category_id)):
            item_path = os.path.join(src_dir, complete_partial, category_id, item)
            shutil.move(item_path, os.path.join(dest_dir, complete_partial, category_id, item))
        

# Move the contents of "Hand_Data_9-17-1-2" to "valid"
if os.path.exists(os.path.join(root_directory, "Hand_Data_9-17-1-2")):
    src_dir = os.path.join(root_directory, "Hand_Data_9-17-1-2")
    dest_dir = valid_folder
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        shutil.move(item_path, os.path.join(dest_dir, item))

# Move the contents of "Hand_Data_9-25-1-2" to "test"
if os.path.exists(os.path.join(root_directory, "Hand_Data_9-25-1-2")):
    src_dir = os.path.join(root_directory, "Hand_Data_9-25-1-2")
    dest_dir = test_folder
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        shutil.move(item_path, os.path.join(dest_dir, item))

# List of folder names to rename
folders_to_rename = ['Complete', 'Partial']

# Iterate through the train, test, and valid folders
for split_folder in ['train', 'test', 'valid']:
    for folder_name in folders_to_rename:
        old_folder_path = os.path.join(destination_folder, split_folder, folder_name)
        new_folder_path = os.path.join(destination_folder, split_folder, folder_name.lower())
        
        # Rename the folder to lowercase
        os.rename(old_folder_path, new_folder_path)

train_files = os.listdir(os.path.join(train_folder, 'complete', category_id))
valid_files = os.listdir(os.path.join(valid_folder, 'complete', category_id))
test_files = os.listdir(os.path.join(test_folder, 'complete', category_id))

# Create a JSON file with the split information
split_info = [{
    "taxonomy_id": category_id,
    "taxonomy_name": category,
    "train": train_files,
    "valid": valid_files,
    "test": test_files
}]

json_file_path = os.path.join(destination_folder, 'PCN.json')

with open(json_file_path, 'w') as json_file:
    json.dump(split_info, json_file)

category_file_path = os.path.join(destination_folder, 'category.txt')

with open(category_file_path, 'w') as category_file:
    category_file.write(category)

