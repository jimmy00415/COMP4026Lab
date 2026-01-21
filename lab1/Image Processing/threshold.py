import cv2
img = cv2.imread('../lab1 data/ET_1.jpg')

# RGB is converted into gray image using the cvtColor function
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# using cv2.threshold function to segment image with threshold
_,threshimg = cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY)

cv2.imshow('BINARY THRESHimg',threshimg)   
cv2.waitKey(0)                  
cv2.destroyAllWindows()  




threshimg = cv2.adaptiveThreshold(grayimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# using cv2.adaptiveThreshold function to segment image with threshold value which 
# is the weighted sum of neighbourhood values where weights are a gaussian window

cv2.imshow('Adaptive Gaussian THRESHimg',threshimg)   
cv2.waitKey(0)                  
cv2.destroyAllWindows() 




