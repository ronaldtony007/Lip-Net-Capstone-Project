from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../data_files/shape_predictor_68_face_landmarks.dat')
 
cap = cv2.VideoCapture(0) 
    
while (cap.isOpened()):    
    _, image = cap.read() 
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    clone = image.copy()
    overlay = image.copy()
    output = image.copy()

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "mouth":
                pts = shape[i:j]
                break
                
        print(pts)
        hull = cv2.convexHull(pts)
        print(cv2.contourArea(hull))

        cv2.drawContours(overlay, [hull], -1, (158, 163, 32), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
        break
    
    cv2.imshow("Third Ear", output)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
