import os
import json
from collections import defaultdict

def process_person_attendance_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count occurrences of each student ID
    id_count = defaultdict(int)
    # Container for new data
    new_data = {}

    # Count all best_match occurrences first
    for person, info in data.items():
        best_match = str(info.get('best_match'))
        id_count[best_match] += 1

    # Index per student ID for suffixing
    id_index = defaultdict(int)

    for person, info in data.items():
        best_match = str(info.get('best_match'))
        if id_count[best_match] > 1:
            id_index[best_match] += 1
            new_person_key = f"{best_match}-{id_index[best_match]}"
        else:
            new_person_key = best_match
        new_data[new_person_key] = info

    # Save (overwrite the original file)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def walk_and_process(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'person_attendance.json':
                json_path = os.path.join(dirpath, filename)
                print(f"Processing: {json_path}")
                process_person_attendance_json(json_path)

if __name__ == "__main__":
    # Change to your dataset root directory
    root_dir = "../dataset/NB116_person_spatial"
    walk_and_process(root_dir)