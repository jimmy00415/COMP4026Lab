import cv2
import numpy as np

img = cv2.imread('../lab1 data/ET_1.jpg')

# using numpy package to design a 5x5 kernel matrix based low-pass filter
kernel = np.ones((5,5),np.float32)/25  

# using cv2.filter2D function to filter the image with the designed low-pass filter
img = cv2.filter2D(img,-1,kernel) 

cv2.imshow('SMOOTHimg',img)   
cv2.waitKey(0)                 
cv2.destroyAllWindows()      







