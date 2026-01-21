from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
image_path = script_dir.parent / 'lab3 data' / 'face.jpg'

# Load image
image = cv2.imread(str(image_path))
if image is None:
	raise FileNotFoundError(f"Image not found at {image_path}")

cv2.imshow('input', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# RGB is converted into gray image using the cvtColor function
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Try different parameters of LBP extraction
param_sets = [
	{"radius": 1, "n_points": 8, "method": "default"},
	{"radius": 2, "n_points": 8, "method": "default"},
	{"radius": 3, "n_points": 16, "method": "default"},
	{"radius": 2, "n_points": 8, "method": "uniform"},
	{"radius": 2, "n_points": 16, "method": "uniform"},
	{"radius": 2, "n_points": 8, "method": "ror"},
	{"radius": 3, "n_points": 24, "method": "uniform"},
]

for params in param_sets:
	radius = params["radius"]
	n_points = params["n_points"]
	method = params["method"]

	lbp = local_binary_pattern(grayimg, n_points, radius, method=method)

	# Show LBP image
	plt.figure(figsize=(5, 4))
	plt.imshow(lbp, cmap='gray')
	plt.title(f"LBP: radius={radius}, n_points={n_points}, method={method}")
	plt.axis('off')
	plt.show()

	# Histogram of LBP
	n_bins = int(lbp.max() + 1)
	hist, bins = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

	fig, ax = plt.subplots(figsize=(6, 3))
	ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
	ax.set_title(f"Histogram: radius={radius}, n_points={n_points}, method={method}")
	ax.set_xlabel("LBP code")
	ax.set_ylabel("Frequency")
	plt.tight_layout()
	plt.show()




