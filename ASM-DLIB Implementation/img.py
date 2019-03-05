import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in range(3, 4):
	img = cv2.imread('img/image' + str(i) + '.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
	for (x,y,w,h) in mouth_rects:
		# cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
		img = img[x:x+w, y:y+h] 
		break

	lower = np.array([0, 48, 80], dtype = "uint8")
	upper = np.array([20, 255, 255], dtype = "uint8")

	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(img2, lower, upper)

	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	#mask = cv2.dilate(mask, kernel, iterations = 1)

	mask = cv2.GaussianBlur(mask, (11, 11), 0)
	converted = cv2.bitwise_and(img, img, mask = mask)

	cv2.imshow('Original', img)
	cv2.imshow('mask', mask)
	cv2.imshow('HSV', img2)
	cv2.imshow('Converted', converted)
	cv2.waitKey(0)

	cv2.destroyAllWindows()




