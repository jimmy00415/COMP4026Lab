import cv2
from skimage import feature, exposure

img = cv2.imread('../lab2 data/person/crop001208.png')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Input Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ****** HOG map of input image ******
_, hog_image = feature.hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=2)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

cv2.imshow("HOG", hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ****** Person detection ******

# HOG feature description
hog = cv2.HOGDescriptor()
# build SVM detector 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# detect pedestrian
(rects, weights) = hog.detectMultiScale(grayimg,
                                        winStride=(4, 4),
                                        padding=(8, 8),
                                        scale=1.25,
                                        useMeanshiftGrouping=False)
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Person Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


