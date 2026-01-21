import cv2
import numpy as np

# Read the input image
img = cv2.imread('../lab2 data/assignment/ori_img.jpg')

if img is None:
    print("Error: Could not load image")
    exit()

# Display the original image
cv2.imshow('(a) Input Image', img)
cv2.waitKey(0)

# Convert BGR to HSV color space (better for color-based segmentation)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for blue color in HSV
# Blue typically has H value around 100-130 in OpenCV's HSV
# Adjust these values based on the specific blue in the image
lower_blue = np.array([90, 50, 50])    # Lower bound for blue
upper_blue = np.array([130, 255, 255])  # Upper bound for blue

# Create a mask for blue regions
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Apply Gaussian blur to smooth the mask edges
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Invert the mask (we want to keep the person, remove the background)
mask_inv = cv2.bitwise_not(mask)

# Create a black background
black_background = np.zeros_like(img)

# Extract the foreground (person) using the inverted mask
foreground = cv2.bitwise_and(img, img, mask=mask_inv)

# Combine foreground with black background
result = cv2.add(foreground, black_background)

# Display the result
cv2.imshow('(b) Result - Background Removed', result)
cv2.waitKey(0)

# Save the result
output_path = '../lab2 data/assignment/result_img.jpg'
cv2.imwrite(output_path, result)
print(f"Result saved to: {output_path}")

# Optional: Show the mask for debugging
cv2.imshow('Mask (Blue regions)', mask)
cv2.waitKey(0)
cv2.imshow('Inverted Mask (Person regions)', mask_inv)
cv2.waitKey(0)

cv2.destroyAllWindows()
