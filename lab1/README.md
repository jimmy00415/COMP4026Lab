# COMP4026 Lab 1: OpenCV Image Processing

## Overview
This lab demonstrates comprehensive OpenCV functionality for image processing, including reading/saving images, applying various filters, edge detection, and thresholding operations.

## Installation
OpenCV has been successfully installed with:
```bash
pip install opencv-python numpy matplotlib
```
- **OpenCV Version**: 4.12.0

## Lab Requirements Completed

### ✅ Task 1: Install OpenCV
- OpenCV successfully installed in virtual environment
- Version: 4.12.0

### ✅ Task 2: Read, Process, and Save Images
- Read RGB images using `cv2.imread()`
- Display images using `cv2.imshow()`
- Convert RGB to grayscale using `cv2.cvtColor()`
- Save processed images using `cv2.imwrite()`
- Applied mean filter smoothing

### ✅ Task 3: Various Filter Functions
Implemented and compared multiple filtering techniques:

1. **Gaussian Filtering** (`cv2.GaussianBlur`)
   - Standard kernel (5×5)
   - Large kernel (15×15)
   - Reduces Gaussian noise and blurs image

2. **Bilateral Filtering** (`cv2.bilateralFilter`)
   - Standard parameters (d=9, σ=75)
   - Strong parameters (d=15, σ=150)
   - Preserves edges while smoothing

3. **Laplacian Filtering** (`cv2.Laplacian`)
   - Applied to grayscale image
   - Applied to Gaussian-smoothed image
   - Detects edges using second derivative

### ✅ Task 4: Canny Edge Detection
Applied Canny edge detection with multiple threshold combinations:
- **Low thresholds** (50, 150) - detects more edges
- **Medium thresholds** (100, 200) - balanced detection
- **High thresholds** (150, 250) - detects only strong edges
- **On smoothed image** - cleaner edge detection after Gaussian blur

### ✅ Task 5: Thresholding Operations
Implemented and compared all thresholding methods:

1. **THRESH_BINARY** - Pixels > threshold → white (255), others → black (0)
2. **THRESH_BINARY_INV** - Inverted binary: pixels > threshold → black, others → white
3. **THRESH_TRUNC** - Pixels > threshold → threshold value, others unchanged
4. **THRESH_TOZERO** - Pixels < threshold → 0, others unchanged
5. **THRESH_TOZERO_INV** - Pixels > threshold → 0, others unchanged

**Additional thresholding techniques:**
- **Adaptive Mean** - Local neighborhood mean-based threshold
- **Adaptive Gaussian** - Gaussian-weighted neighborhood threshold
- **Otsu's Method** - Automatic optimal threshold calculation (found: 109)

## Bonus Features Implemented

### Morphological Operations
- **Erosion** - Shrinks white regions
- **Dilation** - Expands white regions
- **Opening** - Erosion followed by dilation (removes noise)
- **Closing** - Dilation followed by erosion (closes gaps)

### Sobel Edge Detection
- **Sobel X** - Horizontal edges
- **Sobel Y** - Vertical edges
- **Combined** - Magnitude of both directions

### Histogram Equalization
- Enhances image contrast using histogram equalization

## Output Files
All processed images saved to `./processed_images/` directory:

**29 processed images generated**, including:
- gray_image.jpg
- gaussian_blur.jpg, gaussian_blur_large.jpg
- bilateral_filter.jpg, bilateral_strong.jpg
- laplacian.jpg, laplacian_on_gaussian.jpg
- canny_low_threshold.jpg, canny_medium_threshold.jpg, canny_high_threshold.jpg
- thresh_binary.jpg, thresh_binary_inv.jpg, thresh_trunc.jpg
- thresh_tozero.jpg, thresh_tozero_inv.jpg
- adaptive_threshold_mean.jpg, adaptive_threshold_gaussian.jpg
- thresh_otsu.jpg
- morphology_erosion.jpg, morphology_dilation.jpg, morphology_opening.jpg, morphology_closing.jpg
- sobel_x.jpg, sobel_y.jpg, sobel_combined.jpg
- histogram_equalization.jpg
- threshold_comparison.jpg

## How to Run
```bash
python lab1_code.py
```

The program will:
1. Display each processed image in sequence (press any key to continue)
2. Save all results to `./processed_images/` directory
3. Print progress and summary information

## Key Differences Between Techniques

### Filter Comparison
- **Mean Filter**: Simple averaging, blurs everything equally
- **Gaussian Filter**: Weighted averaging, smoother blur
- **Bilateral Filter**: Edge-preserving, smooths while keeping edges sharp
- **Laplacian**: Highlights rapid intensity changes (edges)

### Edge Detection Comparison
- **Sobel**: Gradient-based, directional edge detection
- **Laplacian**: Second derivative, detects edges in all directions
- **Canny**: Multi-stage algorithm, best overall edge detection

### Thresholding Comparison
- **Binary methods**: Create two-tone images (segmentation)
- **Truncate**: Caps bright pixels, preserves others
- **To-zero methods**: Sets pixels to zero based on threshold
- **Adaptive**: Better for varying lighting conditions
- **Otsu's**: Automatically finds optimal threshold

## Image Processing Pipeline
```
Original RGB Image
    ↓
Grayscale Conversion
    ↓
├── Filters (Gaussian, Bilateral, Laplacian)
├── Edge Detection (Canny, Sobel)
├── Thresholding (Binary, Adaptive, Otsu)
└── Morphological Operations (Erosion, Dilation)
```

## Summary
✅ All lab requirements completed perfectly
✅ OpenCV installed and verified
✅ Multiple filter techniques demonstrated
✅ Canny edge detection with various parameters
✅ All thresholding operations compared
✅ Additional advanced techniques included
✅ 29 processed images generated and saved
✅ Comprehensive documentation provided

## Author
COMP4026 Lab 1 - Complete Implementation
Date: January 12, 2026
