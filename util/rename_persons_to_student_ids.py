#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename person folders in the NB116_person_spatial dataset to corresponding student IDs.
Based on matches in person_attendance.json, if multiple persons map to the same student ID,
append suffixes like -1, -2, etc.
"""

import os
import json
import glob
from collections import defaultdict, Counter


def process_class_folder(class_folder_path):
    """
    Process a single class folder and rename person folders to corresponding student IDs.
    
    Args:
        class_folder_path (str): Path to the class folder
    
    Returns:
        bool: Whether the processing succeeded
    """
    print(f"Processing class folder: {class_folder_path}")
    
    # Path to person_attendance.json
    attendance_file = os.path.join(class_folder_path, 'person_attendance_proxyfusion_quality.json')
    
    if not os.path.exists(attendance_file):
        print(f"Warning: {attendance_file} not found, skipping")
        return False
    
    try:
        # Read JSON file
        with open(attendance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if 'successful_students' not in data:
            print(f"Warning: unexpected structure in {attendance_file}, skipping")
            return False
        
        # Collect all persons with their matched student IDs and similarity
        person_to_student = {}
        
        # Handle 'matched_students' structure
        if 'matched_students' in data:
            # New structure: matched_students
            for student_id, matches in data['matched_students'].items():
                if matches and isinstance(matches, list):
                    # Select the match with highest similarity
                    for match in matches:
                        if isinstance(match, dict) and 'person_id' in match:
                            person_id = match['person_id']
                            similarity = match.get('similarity_score', 0)
                            
                            # Update if not seen or higher similarity for this student
                            if (person_id not in person_to_student or 
                                similarity > person_to_student[person_id]['similarity']):
                                person_to_student[person_id] = {
                                    'student_id': student_id,
                                    'similarity': similarity
                                }
        else:
            # Old structure: direct key-value
            for key, value in data.items():
                # Skip non-student-ID keys
                if key in ['class_folder', 'faces_json_source', 'successful_students', 'fusion_method', 'stats']:
                    continue
                
                if isinstance(value, dict) and 'person_id' in value:
                    person_id = value['person_id']
                    student_id = key
                    similarity = value.get('similarity_score', 0)
                    
                    # Update if not seen or higher similarity for this student
                    if (person_id not in person_to_student or 
                        similarity > person_to_student[person_id]['similarity']):
                        person_to_student[person_id] = {
                            'student_id': student_id,
                            'similarity': similarity
                        }
        
        print(f"Found {len(person_to_student)} person match entries")
        
        # Sort by similarity (desc) so higher-similarity persons get suffix-free names first
        sorted_persons = sorted(
            person_to_student.items(),
            key=lambda x: x[1]['similarity'],
            reverse=True
        )
        
        # Count how many persons map to each student
        student_counter = Counter([info['student_id'] for _, info in sorted_persons])
        
        # Assign suffix for duplicated student IDs
        student_suffix_counter = defaultdict(int)
        rename_plan = {}
        
        for person_id, info in sorted_persons:
            student_id = info['student_id']
            
            # If multiple persons map to the same student, add suffix to all but the first
            if student_counter[student_id] > 1:
                if student_suffix_counter[student_id] == 0:
                    # First (highest similarity) has no suffix
                    new_name = student_id
                else:
                    # Subsequent ones get suffix
                    new_name = f"{student_id}-{student_suffix_counter[student_id]}"
                student_suffix_counter[student_id] += 1
            else:
                # Unique mapping, no suffix
                new_name = student_id
            
            rename_plan[person_id] = new_name
            print(f"  {person_id} -> {new_name} (student: {student_id}, similarity: {info['similarity']:.4f})")
        
        # Execute folder rename
        success_count = 0
        for person_id, new_name in rename_plan.items():
            old_path = os.path.join(class_folder_path, person_id)
            new_path = os.path.join(class_folder_path, new_name)
            
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    print(f"    Warning: target folder {new_path} already exists, skipping {person_id}")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"    ✓ Renamed: {person_id} -> {new_name}")
                        success_count += 1
                    except Exception as e:
                        print(f"    ✗ Rename failed: {person_id} -> {new_name}, error: {str(e)}")
            else:
                print(f"    Warning: person folder {old_path} not found")
        
        print(f"✓ Done: {class_folder_path}, renamed {success_count} folders")
        return True
        
    except Exception as e:
        print(f"✗ Failed to process class folder {class_folder_path}: {str(e)}")
        return False


def main():
    """
    Main: iterate all class folders under NB116_person_spatial and rename person folders
    """
    print("Start renaming person folders in NB116_person_spatial dataset...")
    

    base_path = r"C:\Project\Classroom-Reid\dataset\NB116_person_spatial"
    pattern = os.path.join(base_path, "*", "*")
    class_folders = []
    
    for folder_path in glob.glob(pattern):
        if os.path.isdir(folder_path):
            attendance_file = os.path.join(folder_path, "person_attendance_proxyfusion_quality.json")
            if os.path.exists(attendance_file):
                class_folders.append(folder_path)
    
    if not class_folders:
        print(f"No class folders with person_attendance.json found in {base_path}")
        return
    
    print(f"Found {len(class_folders)} class folders")
    
    success_count = 0
    for folder_path in class_folders:
        if process_class_folder(folder_path):
            success_count += 1
        print("-" * 50)
    
    print(f"\nDone! Successfully processed {success_count}/{len(class_folders)} class folders")


if __name__ == "__main__":
    main()