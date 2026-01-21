import cv2

img = cv2.imread('../lab1 data/ET_1.jpg')

# using cv2.Canny function to detect the edges of image
edges = cv2.Canny(img,100,200) 

cv2.imshow('EDGEimg',edges)   
cv2.waitKey(0)                  
cv2.destroyAllWindows()       











