import os
import json
import random
import shutil

# Set the path to your data folder
data_folder = '/Users/lukasschuepp/framework/hand_data/multiviewDataset/Hand_Data_3500'
destination_folder = '/Users/lukasschuepp/framework/hand_data/multiviewDataset/Hand_Data_split'
category_id = '10'
category = "hand"

# create destination folder
os.makedirs(destination_folder, exist_ok=True)

# create subfolders
os.makedirs(os.path.join(destination_folder, 'train', 'complete', category_id), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'train', 'partial', category_id), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'valid', 'complete', category_id), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'valid', 'partial', category_id), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'test', 'complete', category_id), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'test', 'partial', category_id), exist_ok=True)

# Create directories for train, validation, and test splits
train_folder = os.path.join(destination_folder, 'train')
valid_folder = os.path.join(destination_folder, 'valid')
test_folder = os.path.join(destination_folder, 'test')


# List all the files in the data folder
file_list = os.listdir(os.path.join(data_folder, 'Complete'))

# Shuffle the file list to ensure randomness
random.shuffle(file_list)

# Define the split percentages
train_percent = 0.7
valid_percent = 0.2
test_percent = 0.1

# Calculate the number of files for each split
num_files = len(file_list)
num_train = int(num_files * train_percent)
num_valid = int(num_files * valid_percent)

# Split the files into train, validation, and test sets
train_files = file_list[:num_train]
valid_files = file_list[num_train:num_train + num_valid]
test_files = file_list[num_train + num_valid:]

# Function to copy files to the specified folder
def copy_files(src_files, dest_folder):
    for file in src_files:
        shutil.copy(os.path.join(data_folder, 'complete', file), os.path.join(dest_folder, 'complete', category_id, file))
        for partial_pc in os.listdir(os.path.join(data_folder, 'partial', file.split('.')[0])):
            # create destination folder
            os.makedirs(os.path.join(dest_folder, 'partial', category_id, file.split('.')[0]), exist_ok=True)
            shutil.copy(os.path.join(data_folder, 'partial', file.split('.')[0], partial_pc), os.path.join(dest_folder, 'partial', category_id, file.split('.')[0], partial_pc))


# Copy files to the respective split folders
copy_files(train_files, train_folder)
copy_files(valid_files, valid_folder)
copy_files(test_files, test_folder)

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

print("Data splitting and JSON file creation complete.")
