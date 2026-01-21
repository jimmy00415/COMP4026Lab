import cv2
import os
import numpy as np
from pathlib import Path
import csv
from datetime import datetime

# Configuration
script_dir = Path(__file__).resolve().parent
cascade_path = script_dir.parent / "lab4" / "Face Recognition" / "lab4 code" / "haarcascade_frontalface_default.xml"

# Subject names (Update with actual names)
subjects = ["", "Ariel Sharon", "Colin Powell", "Donald Rumsfeld"]

# Training set sizes to test
training_sizes = [10, 20, 30]

def detect_face(img):
    """Detect face using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], faces[0]

def prepare_training_data(data_folder_path, max_images_per_person=None):
    """Load and prepare training data with optional limit per person"""
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in sorted(dirs):
        if not dir_name.startswith("s"):
            continue
            
        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = sorted(os.listdir(subject_dir_path))
        
        # Limit number of images if specified
        if max_images_per_person:
            subject_images_names = subject_images_names[:max_images_per_person]
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            face, rect = detect_face(image)
            
            if face is not None:
                face = cv2.resize(face, (200, 300))
                faces.append(face)
                labels.append(label)
    
    return faces, labels

def test_recognizer(face_recognizer, test_data_path):
    """Test the face recognizer and return accuracy metrics"""
    dirs = os.listdir(test_data_path)
    
    total_tests = 0
    correct_predictions = 0
    predictions_per_person = {1: {'correct': 0, 'total': 0},
                              2: {'correct': 0, 'total': 0},
                              3: {'correct': 0, 'total': 0}}
    
    for dir_name in sorted(dirs):
        if not dir_name.startswith("s"):
            continue
            
        true_label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(test_data_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            face, rect = detect_face(image)
            
            if face is not None:
                face = cv2.resize(face, (200, 300))
                predicted_label, confidence = face_recognizer.predict(face)
                
                total_tests += 1
                predictions_per_person[true_label]['total'] += 1
                
                if predicted_label == true_label:
                    correct_predictions += 1
                    predictions_per_person[true_label]['correct'] += 1
    
    overall_accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_tests': total_tests,
        'correct': correct_predictions,
        'per_person': predictions_per_person
    }

def main():
    """Main function to run experiments with different training set sizes"""
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    csv_file = results_dir / "results.csv"
    
    # Prepare CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Training Size', 'Overall Accuracy (%)', 'Total Tests', 
                        'Correct Predictions', 'Person 1 Accuracy (%)', 
                        'Person 2 Accuracy (%)', 'Person 3 Accuracy (%)'])
    
    print("=" * 70)
    print("Lab Assignment 2: Face Recognition with Eigenface")
    print("=" * 70)
    print()
    
    for train_size in training_sizes:
        print(f"\n{'='*70}")
        print(f"Training with {train_size} images per person")
        print(f"{'='*70}\n")
        
        # Prepare training data
        print(f"Preparing training data ({train_size} images/person)...")
        faces, labels = prepare_training_data(
            str(script_dir / "training-data"),
            max_images_per_person=train_size
        )
        
        print(f"Total training faces: {len(faces)}")
        print(f"Total training labels: {len(labels)}")
        
        # Create and train recognizer
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))
        print("Training complete!\n")
        
        # Test recognizer
        print("Testing recognizer...")
        results = test_recognizer(face_recognizer, str(script_dir / "test-data"))
        
        # Display results
        print(f"\nResults for {train_size} training images/person:")
        print(f"  Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"  Total Tests: {results['total_tests']}")
        print(f"  Correct: {results['correct']}")
        print(f"\n  Per Person Results:")
        
        person_accuracies = []
        for person_id in [1, 2, 3]:
            stats = results['per_person'][person_id]
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            person_accuracies.append(acc)
            print(f"    {subjects[person_id]}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        # Save to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                train_size,
                f"{results['overall_accuracy']:.2f}",
                results['total_tests'],
                results['correct'],
                f"{person_accuracies[0]:.2f}",
                f"{person_accuracies[1]:.2f}",
                f"{person_accuracies[2]:.2f}"
            ])
    
    print(f"\n{'='*70}")
    print(f"All experiments complete!")
    print(f"Results saved to: {csv_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
