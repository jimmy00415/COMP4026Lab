import cv2
import os
from skimage.feature import local_binary_pattern
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "lab4 data"
cascade_path = script_dir / "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(str(cascade_path))
#  OpenCV already contains many pre-trained classifiers for face. 
#  We need to load the required pre-trained XML classifiers.

# Function of face detection based viola jones 
def FaceDetction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # if face detector fail to detect the face
    if (len(faces) == 0):
        return None, None
    # if face detector successfully detect the face
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
        return roi_gray, faces[0]
    
# Function of LBP extraction 
def LBPExt(grayimg):

    # LBP extraction based on gray image
    # settings for LBP
    radius = 2 
    n_points = 8
    lbp = local_binary_pattern(grayimg, n_points, radius, 'default')
    # Histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, bins = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    return hist


# Training data of LBP feature and label
FeatureList = []
LabelList = []

TrainDir = str(data_dir / 'training')
Seclist = os.listdir(TrainDir)

for SecNum in range(len(Seclist)):
    SecName = Seclist[SecNum]
    SecDir = os.path.join(TrainDir, SecName)
    
    Imglist = os.listdir(SecDir)
    for ImgNum in range(len(Imglist)):
        if Imglist[ImgNum].startswith('.'):
            continue
        image = cv2.imread(os.path.join(TrainDir, SecName, Imglist[ImgNum]))
        if image is None:
            continue
        DetectedFace,_ = FaceDetction(image)
        
        # if face detector successfully detect the face
        if DetectedFace is not None:
            # LBP extraction
            LBPfeature = LBPExt(DetectedFace)
            FeatureList.append(LBPfeature)
            
            if SecName == 'fake':
                label = 0
            elif SecName == 'real':
                label = 1
            else:
                error = "unknown class"
                raise NotImplementedError(error)
            
            LabelList.append(label)
        
# KNN training        
TrainFeature = np.array(FeatureList).astype(np.float32)
TrainLabel = np.array(LabelList)
knn = cv2.ml.KNearest_create()
knn.train(TrainFeature, cv2.ml.ROW_SAMPLE, TrainLabel) 


# KNN Testing  
test_way = 'img'  # Change to 'cam' for camera testing
if test_way == 'img':
    # Testing data of LBP feature and label
    FeatureList = []
    LabelList = []
    
    TestDir = str(data_dir / 'testing')
    Seclist = os.listdir(TestDir)
    
    for SecNum in range(len(Seclist)):
        SecName = Seclist[SecNum]
        SecDir = os.path.join(TestDir, SecName)
        
        Imglist = os.listdir(SecDir)
        for ImgNum in range(len(Imglist)):
            if Imglist[ImgNum].startswith('.'):
                continue
            image = cv2.imread(os.path.join(TestDir, SecName, Imglist[ImgNum]))
            if image is None:
                continue
            DetectedFace,_ = FaceDetction(image)
            # if face detector successfully detect the face
            if DetectedFace is not None:
                # LBP extraction
                LBPfeature = LBPExt(DetectedFace)
                FeatureList.append(LBPfeature)
                if SecName == 'fake':
                    label = 0
                elif SecName == 'real':
                    label = 1
                else:
                    error = "unknown class"
                    raise NotImplementedError(error)
                LabelList.append(label)
            
    TestFeature = np.array(FeatureList).astype(np.float32)
    TestLabel = np.array(LabelList)

    ret, results, neighbours, dist = knn.findNearest(TestFeature, k=5)
    
    # Compare the result with test_labels and check the accuracy of prediction
    matches = results.squeeze()==TestLabel
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/len(results) 
    print('The accuracy of face presentation attack detection: {:.2f} %'.format(accuracy))

elif test_way == 'cam':
    #function to draw text on give image starting from
    #passed (x, y) coordinates. 
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    video_capture = cv2.VideoCapture(0)
    while cv2.waitKey(1) != 27:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        DetectedFace, rect = FaceDetction(frame)
        if DetectedFace is not None:
            LBPfeature = LBPExt(DetectedFace)
            LBPfeature = np.array([LBPfeature]).astype(np.float32)
            ret, results, neighbours, dist = knn.findNearest(LBPfeature, k=5)
            if results.squeeze() == 1:
                label_text = 'real face'
            else:
                label_text = 'fake face'
            
            #draw label of predicted face
            draw_text(frame, label_text, rect[0], rect[1]-5) 
            
        cv2.imshow('result', frame)
        
        
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



