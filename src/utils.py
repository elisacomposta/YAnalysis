import glob
import json
import os
import random

SAVE = True
SHOW = False
MISINFO_LEVELS = [0.0, 0.05, 0.10, 0.5]
COALITIONS = ["Centre-Left", "M5S", "Third Pole", "Right"]
TOPICS = ["nuclear energy", "immigration", "reddito di cittadinanza", "civil rights"]

 
def get_files(dir_path, file_extension=".db"):
    return sorted(glob.glob(os.path.join(dir_path, f"*{file_extension}")))


def get_random_file_from_dir(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    selected_file = random.choice(files)
    selected_file_path = os.path.join(dir_path, selected_file)
    return selected_file_path


def add_to_json(file_path, data, print_message=False):
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
    except FileNotFoundError:
        content = {}

    content.update(data)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(content, file, indent=4)
    
    if print_message:
        print(f"  - Saved data to {file_path}")
