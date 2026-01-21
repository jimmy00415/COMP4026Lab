"""
Download and prepare LFW (Labeled Faces in the Wild) dataset for Assignment 2.
This script will:
1. Download LFW dataset using sklearn
2. Select 3 people with at least 50 images each
3. Split into training (30 images) and test (20 images)
4. Save to the appropriate folders
"""

import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_lfw_people
import cv2

def download_and_prepare_dataset():
    """Download LFW dataset and prepare for assignment"""
    
    script_dir = Path(__file__).resolve().parent
    
    print("=" * 70)
    print("Downloading LFW Dataset...")
    print("=" * 70)
    
    # Download LFW dataset with at least 50 images per person
    # This ensures we have enough for 30 training + 20 test
    lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=1.0, 
                                   color=True, slice_=None)
    
    print(f"\nDataset downloaded!")
    print(f"Total people with 50+ images: {len(lfw_people.target_names)}")
    print(f"Image shape: {lfw_people.images[0].shape}")
    
    # Get the first 3 people (those with most images)
    selected_people = []
    for idx, name in enumerate(lfw_people.target_names[:3]):
        person_images = lfw_people.images[lfw_people.target == idx]
        selected_people.append({
            'name': name,
            'images': person_images,
            'count': len(person_images)
        })
        print(f"Person {idx+1}: {name} - {len(person_images)} images")
    
    print("\n" + "=" * 70)
    print("Preparing dataset folders...")
    print("=" * 70)
    
    # Prepare folders
    for person_idx, person_data in enumerate(selected_people, start=1):
        person_name = person_data['name']
        images = person_data['images']
        
        # Shuffle images
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(images))
        shuffled_images = images[indices]
        
        # Split: first 30 for training, next 20 for test
        train_images = shuffled_images[:30]
        test_images = shuffled_images[30:50]
        
        # Save training images
        train_dir = script_dir / "training-data" / f"s{person_idx}"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove README.txt if exists
        readme_file = train_dir / "README.txt"
        if readme_file.exists():
            readme_file.unlink()
        
        print(f"\nSaving Person {person_idx} ({person_name}) training images...")
        for img_idx, img in enumerate(train_images, start=1):
            # Convert from RGB float [0, 1] to BGR uint8 [0, 255]
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_path = train_dir / f"{img_idx}.jpg"
            cv2.imwrite(str(img_path), img_bgr)
        
        print(f"  Saved {len(train_images)} training images to {train_dir}")
        
        # Save test images
        test_dir = script_dir / "test-data" / f"s{person_idx}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove README.txt if exists
        readme_file = test_dir / "README.txt"
        if readme_file.exists():
            readme_file.unlink()
        
        print(f"Saving Person {person_idx} ({person_name}) test images...")
        for img_idx, img in enumerate(test_images, start=1):
            # Convert from RGB float [0, 1] to BGR uint8 [0, 255]
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_path = test_dir / f"{img_idx}.jpg"
            cv2.imwrite(str(img_path), img_bgr)
        
        print(f"  Saved {len(test_images)} test images to {test_dir}")
    
    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print("\nPeople selected:")
    for idx, person in enumerate(selected_people, start=1):
        print(f"  Person {idx} (s{idx}): {person['name']}")
    
    print("\nNext steps:")
    print("1. Update subject names in assignment2_face_recognition.py")
    print("2. Run: python assignment2_face_recognition.py")
    
    # Update the subject names in the assignment script
    update_subject_names(script_dir, selected_people)

def update_subject_names(script_dir, selected_people):
    """Update the subject names in the assignment script"""
    assignment_file = script_dir / "assignment2_face_recognition.py"
    
    with open(assignment_file, 'r') as f:
        content = f.read()
    
    # Create new subject line
    names = [person['name'] for person in selected_people]
    new_subjects_line = f'subjects = ["", "{names[0]}", "{names[1]}", "{names[2]}"]'
    
    # Replace the subjects line
    import re
    content = re.sub(
        r'subjects = \["", ".*?", ".*?", ".*?"\]',
        new_subjects_line,
        content
    )
    
    with open(assignment_file, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Updated subject names in {assignment_file.name}")

if __name__ == "__main__":
    download_and_prepare_dataset()
