"""
Advanced Background Removal Algorithm
This script removes blue/green screen backgrounds from portrait images
and produces a result with a black background.

Algorithm steps:
1. Convert image to HSV color space for better color segmentation
2. Create a mask for the background color range
3. Apply morphological operations to clean up the mask
4. Use edge refinement for smoother transitions
5. Apply the mask to create the final result
"""

import cv2
import numpy as np


def remove_background(image_path, output_path, show_steps=False):
    """
    Remove blue/green background from an image
    
    Parameters:
    - image_path: Path to the input image
    - output_path: Path to save the result
    - show_steps: Whether to display intermediate processing steps
    """
    
    # Read the input image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Display the original image
    cv2.imshow('(a) Input Image', img)
    cv2.waitKey(0)
    
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Method 1: HSV-based segmentation for blue background
    # Define range for blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue regions
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Method 2: Also try detecting bright blue (like in studio photos)
    lower_bright_blue = np.array([100, 100, 100])
    upper_bright_blue = np.array([130, 255, 255])
    mask_bright_blue = cv2.inRange(hsv, lower_bright_blue, upper_bright_blue)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask_blue, mask_bright_blue)
    
    if show_steps:
        cv2.imshow('Initial Mask', mask)
        cv2.waitKey(0)
    
    # Apply morphological operations to clean up the mask
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    if show_steps:
        cv2.imshow('After Morphological Operations', mask)
        cv2.waitKey(0)
    
    # Apply Gaussian blur to smooth mask edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    # Threshold to create a binary mask again
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    if show_steps:
        cv2.imshow('Final Mask (Blue regions)', mask)
        cv2.waitKey(0)
    
    # Invert the mask (we want to keep the person, remove the background)
    mask_inv = cv2.bitwise_not(mask)
    
    if show_steps:
        cv2.imshow('Inverted Mask (Person regions)', mask_inv)
        cv2.waitKey(0)
    
    # Create a black background
    black_background = np.zeros_like(img)
    
    # Extract the foreground (person) using the inverted mask
    foreground = cv2.bitwise_and(img, img, mask=mask_inv)
    
    # Combine foreground with black background
    result = cv2.add(foreground, black_background)
    
    # Display the result
    cv2.imshow('(b) Result - Background Removed', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
    
    return result


def main():
    # Set paths
    input_path = '../lab2 data/assignment/ori_img.jpg'
    output_path = '../lab2 data/assignment/result_img.jpg'
    
    # Remove background (set show_steps=True to see intermediate results)
    result = remove_background(input_path, output_path, show_steps=True)
    
    if result is not None:
        print("Background removal completed successfully!")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
