# Lab Assignment 2: Face Recognition with Eigenface

## Objective
Train and evaluate a face recognition model with 3 individuals using different training set sizes (10, 20, and 30 images per person).

## Assignment Requirements
- Train face recognition model with 3 different individuals
- Test with 3 training set sizes: 10, 20, and 30 images per person
- Use 20 different test images per person (no overlap with training)
- Record and analyze recognition results
- Submit 2-page report by February 27, 2026

## Folder Structure
```
Lab assignment2/
  ├── README.md (this file)
  ├── assignment2_face_recognition.py
  ├── results/
  │   └── results.csv (generated after running)
  ├── training-data/
  │   ├── s1/  (30 images for Person 1)
  │   ├── s2/  (30 images for Person 2)
  │   └── s3/  (30 images for Person 3)
  └── test-data/
      ├── s1/  (20 images for Person 1)
      ├── s2/  (20 images for Person 2)
      └── s3/  (20 images for Person 3)
```

## Setup Instructions

### 1. Collect Training Images
- Collect **30 images** for each of the 3 individuals
- Create folders `s1`, `s2`, `s3` for the three people
- Place images in `training-data/s1/`, `training-data/s2/`, `training-data/s3/`
- Number images sequentially: `1.jpg`, `2.jpg`, ..., `30.jpg`
- Tips for good training data:
  - Include variety in expressions (smiling, neutral, serious)
  - Different lighting conditions
  - Different angles (frontal, slight left/right)
  - With/without glasses (if applicable)

### 2. Collect Test Images
- Collect **20 different images** for each person (NO overlap with training)
- Place in `test-data/s1/`, `test-data/s2/`, `test-data/s3/`
- Number images: `1.jpg`, `2.jpg`, ..., `20.jpg`
- Ensure test images are completely separate from training images

### 3. Update Subject Names
Edit `assignment2_face_recognition.py` line 13:
```python
subjects = ["", "Your Name", "Friend Name", "Celebrity Name"]
```
Replace with actual names of the 3 people.

### 4. Run the Experiment
```bash
python assignment2_face_recognition.py
```

The script will:
1. Train 3 separate models (10, 20, 30 training images)
2. Test each model with 20 test images per person
3. Calculate overall and per-person accuracy
4. Save results to `results/results.csv`
5. Display comprehensive results in the console

## Results Format
The CSV file contains:
- **Training Size**: Number of images used per person (10, 20, 30)
- **Overall Accuracy**: Percentage of correct predictions across all people
- **Total Tests**: Total number of test images evaluated
- **Correct Predictions**: Number of correctly classified images
- **Person 1/2/3 Accuracy**: Individual accuracy for each person

## Image Sources
You can collect images from:

### Option 1: Public Datasets
- **LFW (Labeled Faces in the Wild)**: http://vis-www.cs.umass.edu/lfw/
- **CelebA**: Celebrity faces dataset
- **VGGFace2**: Large-scale face recognition dataset

### Option 2: Personal Images
- Your own photos and friends/family (with permission)
- Extract frames from personal videos
- Use smartphone camera to capture images

### Option 3: Web Sources
- Google Images (ensure you have rights to use)
- Social media profile pictures (with permission)
- Public figure images from news websites

### Best Practices
1. **Face Visibility**: Ensure face is clearly visible in all images
2. **Image Quality**: Use high-resolution images (at least 640x480)
3. **Consistency**: Use similar image formats (all JPG or all PNG)
4. **Diversity**: Include variety in poses, expressions, and conditions
5. **No Overlap**: Absolutely no shared images between training and testing

## Expected Results
- **10 images/person**: Lower accuracy (60-80%), baseline performance
- **20 images/person**: Improved accuracy (70-85%), better generalization
- **30 images/person**: Best accuracy (80-95%), optimal training

Results may vary based on:
- Image quality
- Lighting consistency
- Facial expression variation
- Pose diversity

## Report Guidelines
Your 2-page report should include:

### Section 1: Introduction
- Brief description of the task
- Names of the 3 individuals chosen
- Source of images (dataset/personal/web)

### Section 2: Methodology
- Training data: 30 images per person
- Test data: 20 images per person  
- Experiments: 10, 20, 30 training sizes
- Tool: EigenFace with OpenCV

### Section 3: Results
- Table with accuracy for each training size
- Per-person accuracy breakdown
- Include the CSV results or recreate as table

Example table:
| Training Size | Overall Accuracy | Person 1 | Person 2 | Person 3 |
|--------------|------------------|----------|----------|----------|
| 10           | 75.00%          | 80.00%   | 70.00%   | 75.00%   |
| 20           | 85.00%          | 85.00%   | 85.00%   | 85.00%   |
| 30           | 91.67%          | 90.00%   | 95.00%   | 90.00%   |

### Section 4: Discussion
- Analysis of how training size affects accuracy
- Which person was easiest/hardest to recognize and why
- Any challenges encountered
- Possible improvements

### Section 5: Conclusion
- Summary of findings
- Key takeaways about face recognition

### Section 6: References (if applicable)
- Image sources
- Any papers or resources consulted

## Image Storage Options
If images are too large for direct submission:
- **Google Drive**: Create shared folder link
- **OneDrive**: Create shared folder link
- **GitHub**: Create private repo (if images have rights)
- **Dropbox**: Create shared folder link

Include the storage link in your report.

## Troubleshooting

### Issue: "No faces detected"
- **Solution**: Ensure faces are clearly visible and frontal
- Use images with good lighting
- Check if `haarcascade_frontalface_default.xml` path is correct

### Issue: "Low accuracy even with 30 images"
- **Solution**: Check image quality and consistency
- Ensure no corrupted images
- Verify test images are different from training
- Consider using more varied training images

### Issue: "One person has much lower accuracy"
- **Solution**: Review images for that person
- May need better quality or more diverse training images
- Check if lighting/pose is very different in test vs training

## Submission Checklist
- [ ] Collected 30 training images for 3 people
- [ ] Collected 20 test images for 3 people (no overlap)
- [ ] Updated subject names in script
- [ ] Ran experiment successfully
- [ ] Generated results.csv
- [ ] Wrote 2-page report
- [ ] Included/linked all images
- [ ] Named file as StudentID_Lab_Assignment2.pdf
- [ ] Submitted to Moodle before Feb 27, 2026

## Notes
- **Late Submission**: 10% deduction per day
- **File Naming**: Use format `12345678_Lab_Assignment2.pdf`
- **Page Limit**: Strictly 2 pages (excluding references if needed)
- **Academic Integrity**: Use your own work; cite any sources

## Contact
For questions or issues:
- Check course Moodle forum
- Email instructor
- Attend office hours

---

**Good luck with your assignment!**

*Created: January 21, 2026*
*Course: COMP4026 Computer Vision and Pattern Recognition*
