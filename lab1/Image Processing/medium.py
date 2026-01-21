import cv2

img = cv2.imread('../lab1 data/ET_1.jpg')

medianimg = cv2.medianBlur(img,5)
# using cv2.medianBlur function to filter the image via the median filter 
# with kernel size 5x5

cv2.imshow('MEDIANimg',medianimg)   
cv2.waitKey(0)                  
cv2.destroyAllWindows()       




