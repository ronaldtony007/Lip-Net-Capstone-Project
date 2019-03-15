# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
cap = cv2.VideoCapture('video/video1.3gp') 
    
while (cap.isOpened()):    
    # load the input image, resize it, and convert it to grayscale
    _, image = cap.read() 
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    clone = image.copy()
    overlay = image.copy()
    output = image.copy()

    # loop over the face detections
    for (i, rect) in enumerate(rects):
	    # determine the facial landmarks for the face region, then
	    # convert the landmark (x, y)-coordinates to a NumPy array
	    shape = predictor(gray, rect)
	    shape = face_utils.shape_to_np(shape)
     
	    # loop over the face parts individually
	    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		    
		    if name == "mouth":
			    print(i, j)
			    pts = shape[i:j]
		        
			    # loop over the subset of facial landmarks, drawing the
			    # specific face part
			    # for (x, y) in shape[i:j]:
				#     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
				    
			    # extract the ROI of the face region as a separate image
			    # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			    # roi = image[y:y + h, x:x + w]
			    # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
	     
			    # show the particular face part
			    # cv2.imshow("ROI", roi)
			    # cv2.imshow("Image", clone)
			    # cv2.waitKey(0)
     
	    # visualize all facial landmarks with a transparent overlay
	    print(pts)
	    hull = cv2.convexHull(pts)
	    
	    print(cv2.contourArea(hull))
	    
	    cv2.drawContours(overlay, [hull], -1, (158, 163, 32), -1)
	    
	    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
	    break
    
    cv2.imshow("Image", output)
    if cv2.waitKey(1) == 27:
        break
	
cv2.destroyAllWindows()
