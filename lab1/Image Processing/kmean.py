import numpy as np
import cv2

img = cv2.imread('../lab1 data/ET_1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshaping the image into a 2D array of pixels and convert to float type 
reshapedImage = np.float32(img.reshape(-1, 3))

# define stop criteria, number of clusters(K) 
K = 8
StopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

#apply kmeans()
ret, labels, clusters = cv2.kmeans(reshapedImage,K,None,StopCriteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
clusters = np.uint8(clusters)
intermediateImage = clusters[labels.flatten()]
clusteredImage = intermediateImage.reshape((img.shape))

cv2.imshow('segimg with K='+str(K),clusteredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

