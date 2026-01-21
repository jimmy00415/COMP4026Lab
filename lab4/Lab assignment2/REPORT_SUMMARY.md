# Lab Assignment 2: Face Recognition with Eigenface - Report Summary

## Student Information
- **Course**: COMP4026 Computer Vision and Pattern Recognition
- **Assignment**: Lab Assignment 2
- **Date**: January 21, 2026
- **Dataset**: LFW (Labeled Faces in the Wild)

## 1. Introduction

This assignment implements a face recognition system using the Eigenface method with OpenCV. The objective is to evaluate how training set size affects recognition accuracy by testing with 10, 20, and 30 training images per person.

### Individuals Selected
Three public figures from the LFW dataset were selected:
1. **Person 1 (s1)**: Ariel Sharon
2. **Person 2 (s2)**: Colin Powell
3. **Person 3 (s3)**: Donald Rumsfeld

### Dataset Source
- **Source**: Labeled Faces in the Wild (LFW) - http://vis-www.cs.umass.edu/lfw/
- **Reason for Selection**: LFW is a widely used face recognition benchmark with multiple images per person, ensuring sufficient data for training and testing
- **Images per Person**: 77, 236, and 121 images respectively
- **Image Resolution**: 250x250 pixels, RGB color

## 2. Methodology

### Data Collection
- **Training Data**: 30 images per person (stored in training-data/s1, s2, s3)
- **Test Data**: 20 images per person (stored in test-data/s1, s2, s3)
- **Total Images**: 90 training + 60 testing = 150 images
- **No Overlap**: Training and test sets are completely disjoint

### Feature Extraction
- **Method**: EigenFace (Principal Component Analysis on face images)
- **Preprocessing**: 
  - Face detection using Haar Cascade classifier
  - Face cropping and resizing to 200x300 pixels
  - Grayscale conversion for eigenface computation

### Training Configuration
Three experiments with different training set sizes:
- **Experiment 1**: 10 images per person (30 total)
- **Experiment 2**: 20 images per person (60 total)
- **Experiment 3**: 30 images per person (90 total)

### Testing
- Each model tested on the same 20 images per person (60 total)
- Metrics: Overall accuracy and per-person accuracy

## 3. Results

### Accuracy Table

| Training Size | Overall Accuracy | Person 1 (Ariel Sharon) | Person 2 (Colin Powell) | Person 3 (Donald Rumsfeld) |
|---------------|------------------|-------------------------|-------------------------|----------------------------|
| 10 images     | 35.00%          | 35.00% (7/20)          | 20.00% (4/20)          | 50.00% (10/20)            |
| 20 images     | 51.67%          | 55.00% (11/20)         | 45.00% (9/20)          | 55.00% (11/20)            |
| 30 images     | 45.00%          | 30.00% (6/20)          | 40.00% (8/20)          | 65.00% (13/20)            |

### Observations

1. **Training Size vs. Accuracy**: 
   - 20 images per person achieved the highest overall accuracy (51.67%)
   - 10 images showed lowest performance (35.00%)
   - Surprisingly, 30 images (45.00%) performed worse than 20 images

2. **Per-Person Performance**:
   - **Donald Rumsfeld** (Person 3): Most consistent, improving from 50% → 55% → 65%
   - **Colin Powell** (Person 2): Lowest accuracy, ranging 20-45%
   - **Ariel Sharon** (Person 1): Most variable, 30-55%

3. **Accuracy Trend**: 
   - Expected improvement from 10 to 20 images
   - Unexpected decline from 20 to 30 images suggests potential overfitting or dataset bias

## 4. Discussion

### Analysis of Results

**Why 20 images performed better than 30:**
1. **Overfitting**: The 30-image model may have overfitted to training data peculiarities
2. **PCA Dimensionality**: More training data doesn't always help if eigenface components capture noise
3. **Data Distribution**: The additional 10 images may have introduced more variability rather than better representation

**Per-Person Analysis:**

**Donald Rumsfeld** (Best Performance):
- Steady improvement across all training sizes
- Clear facial features and consistent expressions in dataset
- Well-represented by eigenface features

**Colin Powell** (Poorest Performance):
- Lowest accuracy in all experiments
- Possible causes:
  - High variability in lighting conditions in his images
  - More diverse poses or expressions
  - Dataset contains more challenging test cases

**Ariel Sharon** (Variable Performance):
- Improved from 10 to 20 images but dropped with 30
- Suggests optimal training size around 20 images for this individual
- May have specific visual characteristics that benefit from moderate training

### Challenges Encountered

1. **Face Detection Failures**: Some images failed face detection due to extreme angles or lighting
2. **Computational Time**: Training with 30 images per person took notably longer
3. **Dataset Imbalance**: LFW has varying numbers of images per person (77-236)

### Possible Improvements

1. **Data Augmentation**: Apply transformations to training images for better generalization
2. **Feature Enhancement**: 
   - Use deeper features (e.g., FisherFace, LBPH)
   - Combine multiple feature descriptors
3. **Preprocessing**: 
   - Histogram equalization for lighting normalization
   - Better face alignment techniques
4. **Model Tuning**:
   - Optimize number of eigenfaces retained
   - Experiment with different distance metrics
5. **Cross-Validation**: Use k-fold cross-validation for more robust evaluation

## 5. Conclusion

This assignment demonstrated the practical application of Eigenface-based face recognition. Key findings:

1. **Training Size Impact**: More training data doesn't always guarantee better performance; 20 images per person achieved optimal results
2. **Individual Variation**: Recognition accuracy varies significantly across individuals due to dataset characteristics
3. **Eigenface Limitations**: Simple PCA-based features may not capture all discriminative facial characteristics

The experiment successfully showed that:
- Face recognition requires careful balance between training data quantity and quality
- Individual facial characteristics greatly affect recognition accuracy  
- 20 training images per person provides a good balance for this dataset

**Recommendation**: For practical face recognition systems, use 15-25 training images per person with proper preprocessing and consider hybrid approaches combining multiple feature extraction methods.

## 6. References

1. **Dataset**: Gary B. Huang, et al. "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments." University of Massachusetts, Amherst, Technical Report 07-49, October 2007.
   - http://vis-www.cs.umass.edu/lfw/

2. **Eigenface Method**: Turk, M., & Pentland, A. (1991). "Eigenfaces for recognition." Journal of cognitive neuroscience, 3(1), 71-86.

3. **OpenCV Documentation**: OpenCV Face Recognition. https://docs.opencv.org/

4. **Tools Used**:
   - Python 3.13.7
   - OpenCV 4.13.0
   - scikit-learn for LFW dataset access
   - NumPy for numerical operations

---

## Appendix: Dataset Information

### Training Data Distribution
- **Person 1 (Ariel Sharon)**: 30 images (1.jpg - 30.jpg)
- **Person 2 (Colin Powell)**: 30 images (1.jpg - 30.jpg)
- **Person 3 (Donald Rumsfeld)**: 30 images (1.jpg - 30.jpg)

### Test Data Distribution  
- **Person 1 (Ariel Sharon)**: 20 images (1.jpg - 20.jpg)
- **Person 2 (Colin Powell)**: 20 images (1.jpg - 20.jpg)
- **Person 3 (Donald Rumsfeld)**: 20 images (1.jpg - 20.jpg)

### Data Access
All images are stored locally in the assignment folder structure:
- `Lab assignment2/training-data/s{1,2,3}/`
- `Lab assignment2/test-data/s{1,2,3}/`

### Experimental Setup
- Random seed: 42 (for reproducibility)
- Face detection: Haar Cascade frontal face detector
- Face size: 200x300 pixels (resized after detection)
- No data augmentation applied
- Training-test split: Chronologically random with fixed seed

---

**End of Report**

*This report summarizes the complete Lab Assignment 2 implementation and results. Full code and dataset available in the submission package.*
