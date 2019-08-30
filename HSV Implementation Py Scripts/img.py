import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

	_, image = cap.read()

	image = cv2.resize(image, (600, 500))
	blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	lower_pink = np.array([7, 130, 122])
	upper_pink = np.array([25, 255, 255])

	mask = cv2.inRange(hsv, lower_pink, upper_pink)

	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(image, contours, -1, (0, 255, 0), 1)


	cv2.imshow('Original Video', image)
	cv2.imshow('Masked Image', mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()