"""
COMP4026 Lab 1: OpenCV Image Processing
Complete implementation of all required tasks:
1. Install OpenCV
2. Read, process, and save images
3. Apply various filters (Laplacian, Gaussian, Bilateral)
4. Canny edge detection
5. Different thresholding operations
"""

import cv2
import numpy as np
import os

# Create output directory for processed images
output_dir = './processed_images'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("COMP4026 Lab 1: OpenCV Image Processing")
print("=" * 70)

# ============================================================================
# TASK 1 & 2: Read, Process, and Save Images
# ============================================================================
print("\n[1] Reading and Processing Images...")

# Read RGB image
rgb_img = cv2.imread('./lab1 data/ET_1.jpg')
if rgb_img is None:
    print("Error: Could not read image!")
    exit()

print(f"   - Image shape: {rgb_img.shape}")
print(f"   - Image dtype: {rgb_img.dtype}")

# Display RGB image
cv2.imshow('1. Original RGB Image', rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('2. Grayscale Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save grayscale image
cv2.imwrite(f'{output_dir}/gray_image.jpg', gray_img)
print(f"   ✓ Saved: grayscale image")

# Basic smoothing using mean filter
smooth_img = cv2.blur(rgb_img, (5, 5))
cv2.imshow('3. Mean Filter Smoothing', smooth_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/smooth_mean.jpg', smooth_img)
print(f"   ✓ Saved: mean filtered image")

# ============================================================================
# TASK 3: Apply Various Filters (Laplacian, Gaussian, Bilateral)
# ============================================================================
print("\n[2] Applying Various Filters...")

# Gaussian Filter - reduces Gaussian noise and details
print("   - Gaussian Filtering...")
gaussian_blur = cv2.GaussianBlur(rgb_img, (5, 5), 0)
cv2.imshow('4. Gaussian Blur', gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/gaussian_blur.jpg', gaussian_blur)
print(f"   ✓ Gaussian filter applied and saved")

# Try different kernel sizes for Gaussian
gaussian_blur_large = cv2.GaussianBlur(rgb_img, (15, 15), 0)
cv2.imshow('5. Gaussian Blur (Large Kernel)', gaussian_blur_large)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/gaussian_blur_large.jpg', gaussian_blur_large)

# Bilateral Filter - preserves edges while smoothing
print("   - Bilateral Filtering...")
bilateral = cv2.bilateralFilter(rgb_img, 9, 75, 75)
cv2.imshow('6. Bilateral Filter', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/bilateral_filter.jpg', bilateral)
print(f"   ✓ Bilateral filter applied and saved")

# Try different parameters for bilateral filter
bilateral_strong = cv2.bilateralFilter(rgb_img, 15, 150, 150)
cv2.imshow('7. Bilateral Filter (Strong)', bilateral_strong)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/bilateral_strong.jpg', bilateral_strong)

# Laplacian Filter - edge detection based on second derivative
print("   - Laplacian Filtering...")
# Convert to grayscale first for better results
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
laplacian_abs = np.uint8(np.absolute(laplacian))
cv2.imshow('8. Laplacian Filter', laplacian_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/laplacian.jpg', laplacian_abs)
print(f"   ✓ Laplacian filter applied and saved")

# Apply Laplacian on Gaussian smoothed image
laplacian_on_gaussian = cv2.Laplacian(cv2.GaussianBlur(gray_img, (5, 5), 0), cv2.CV_64F)
laplacian_on_gaussian_abs = np.uint8(np.absolute(laplacian_on_gaussian))
cv2.imshow('9. Laplacian on Gaussian Smoothed', laplacian_on_gaussian_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/laplacian_on_gaussian.jpg', laplacian_on_gaussian_abs)

# ============================================================================
# TASK 4: Canny Edge Detection on Multiple Images
# ============================================================================
print("\n[3] Applying Canny Edge Detection...")

# Canny edge detection with different parameters
print("   - Canny with low thresholds...")
canny_low = cv2.Canny(gray_img, 50, 150)
cv2.imshow('10. Canny Edge Detection (Low Threshold)', canny_low)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/canny_low_threshold.jpg', canny_low)
print(f"   ✓ Canny (50, 150) saved")

print("   - Canny with medium thresholds...")
canny_medium = cv2.Canny(gray_img, 100, 200)
cv2.imshow('11. Canny Edge Detection (Medium Threshold)', canny_medium)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/canny_medium_threshold.jpg', canny_medium)
print(f"   ✓ Canny (100, 200) saved")

print("   - Canny with high thresholds...")
canny_high = cv2.Canny(gray_img, 150, 250)
cv2.imshow('12. Canny Edge Detection (High Threshold)', canny_high)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/canny_high_threshold.jpg', canny_high)
print(f"   ✓ Canny (150, 250) saved")

# Canny on Gaussian blurred image
print("   - Canny on Gaussian smoothed image...")
canny_on_blur = cv2.Canny(cv2.GaussianBlur(gray_img, (5, 5), 0), 100, 200)
cv2.imshow('13. Canny on Gaussian Smoothed', canny_on_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/canny_on_blur.jpg', canny_on_blur)
print(f"   ✓ Canny on smoothed image saved")

# ============================================================================
# TASK 5: Different Thresholding Operations
# ============================================================================
print("\n[4] Applying Different Thresholding Operations...")

# Use a threshold value of 127 (middle value)
threshold_value = 127
max_value = 255

# THRESH_BINARY - creates binary image (pixels above threshold become max_value)
print("   - THRESH_BINARY...")
ret, thresh_binary = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)
cv2.imshow('14. THRESH_BINARY', thresh_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_binary.jpg', thresh_binary)
print(f"   ✓ THRESH_BINARY: pixels > {threshold_value} → white, others → black")

# THRESH_BINARY_INV - inverse of binary threshold
print("   - THRESH_BINARY_INV...")
ret, thresh_binary_inv = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY_INV)
cv2.imshow('15. THRESH_BINARY_INV', thresh_binary_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_binary_inv.jpg', thresh_binary_inv)
print(f"   ✓ THRESH_BINARY_INV: pixels > {threshold_value} → black, others → white")

# THRESH_TRUNC - truncates pixel values at threshold
print("   - THRESH_TRUNC...")
ret, thresh_trunc = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_TRUNC)
cv2.imshow('16. THRESH_TRUNC', thresh_trunc)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_trunc.jpg', thresh_trunc)
print(f"   ✓ THRESH_TRUNC: pixels > {threshold_value} → {threshold_value}, others unchanged")

# THRESH_TOZERO - sets pixels below threshold to zero
print("   - THRESH_TOZERO...")
ret, thresh_tozero = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_TOZERO)
cv2.imshow('17. THRESH_TOZERO', thresh_tozero)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_tozero.jpg', thresh_tozero)
print(f"   ✓ THRESH_TOZERO: pixels < {threshold_value} → 0, others unchanged")

# THRESH_TOZERO_INV - inverse of THRESH_TOZERO
print("   - THRESH_TOZERO_INV...")
ret, thresh_tozero_inv = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_TOZERO_INV)
cv2.imshow('18. THRESH_TOZERO_INV', thresh_tozero_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_tozero_inv.jpg', thresh_tozero_inv)
print(f"   ✓ THRESH_TOZERO_INV: pixels > {threshold_value} → 0, others unchanged")

# Compare thresholding methods side by side
print("\n   - Creating comparison image...")
comparison = np.hstack([
    cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(thresh_binary, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(thresh_binary_inv, cv2.COLOR_GRAY2BGR)
])
cv2.imshow('19. Comparison: Original | BINARY | BINARY_INV', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/threshold_comparison.jpg', comparison)

# Adaptive thresholding for better results
print("   - Adaptive Thresholding...")
adaptive_mean = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
cv2.imshow('20. Adaptive Threshold (Mean)', adaptive_mean)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/adaptive_threshold_mean.jpg', adaptive_mean)

adaptive_gaussian = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
cv2.imshow('21. Adaptive Threshold (Gaussian)', adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/adaptive_threshold_gaussian.jpg', adaptive_gaussian)
print(f"   ✓ Adaptive thresholding methods saved")

# Otsu's thresholding - automatic threshold selection
print("   - Otsu's Thresholding...")
ret_otsu, thresh_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"   ✓ Otsu's optimal threshold: {ret_otsu:.2f}")
cv2.imshow('22. Otsu\'s Thresholding', thresh_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/thresh_otsu.jpg', thresh_otsu)

# ============================================================================
# Additional Processing Examples
# ============================================================================
print("\n[5] Additional Image Processing Examples...")

# Morphological operations
print("   - Morphological Operations...")
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(thresh_binary, kernel, iterations=1)
dilation = cv2.dilate(thresh_binary, kernel, iterations=1)
opening = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow('23. Erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/morphology_erosion.jpg', erosion)

cv2.imshow('24. Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/morphology_dilation.jpg', dilation)

cv2.imshow('25. Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/morphology_opening.jpg', opening)

cv2.imshow('26. Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/morphology_closing.jpg', closing)
print(f"   ✓ Morphological operations saved")

# Sobel edge detection
print("   - Sobel Edge Detection...")
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = np.uint8(np.sqrt(sobelx**2 + sobely**2))

cv2.imshow('27. Sobel X', np.uint8(np.absolute(sobelx)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/sobel_x.jpg', np.uint8(np.absolute(sobelx)))

cv2.imshow('28. Sobel Y', np.uint8(np.absolute(sobely)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/sobel_y.jpg', np.uint8(np.absolute(sobely)))

cv2.imshow('29. Sobel Combined', sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/sobel_combined.jpg', sobel_combined)
print(f"   ✓ Sobel edge detection saved")

# Histogram equalization
print("   - Histogram Equalization...")
hist_eq = cv2.equalizeHist(gray_img)
cv2.imshow('30. Histogram Equalization', hist_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'{output_dir}/histogram_equalization.jpg', hist_eq)
print(f"   ✓ Histogram equalization saved")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("PROCESSING COMPLETE!")
print("=" * 70)
print(f"\n✓ All processed images saved to: {output_dir}/")
print("\nSummary of operations performed:")
print("  1. Read, display, and save images")
print("  2. Applied filters:")
print("     - Mean filter")
print("     - Gaussian blur (multiple kernel sizes)")
print("     - Bilateral filter (edge-preserving)")
print("     - Laplacian filter (edge detection)")
print("  3. Canny edge detection (multiple threshold values)")
print("  4. Thresholding operations:")
print("     - THRESH_BINARY")
print("     - THRESH_BINARY_INV")
print("     - THRESH_TRUNC")
print("     - THRESH_TOZERO")
print("     - THRESH_TOZERO_INV")
print("     - Adaptive thresholding (Mean & Gaussian)")
print("     - Otsu's automatic thresholding")
print("  5. Additional operations:")
print("     - Morphological operations (erosion, dilation, opening, closing)")
print("     - Sobel edge detection (x, y, combined)")
print("     - Histogram equalization")
print("\n" + "=" * 70) 







