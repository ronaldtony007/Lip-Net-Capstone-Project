import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

	_, frame = cap.read()

	#frame = cv2.resize(frame, (500, 500))
	frame_ori = np.copy(frame)
	frame = cv2.GaussianBlur(frame, (15, 15), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_pink = np.array([1, 100, 120])
	upper_pink = np.array([25, 255, 255])

	mask = cv2.inRange(hsv, lower_pink, upper_pink)

	canny = cv2.Canny(mask, 60, 80)

	contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	
	for contour in contours:
		if (cv2.contourArea(contour) > 450) and (cv2.contourArea(contour) < 3000):
			cv2.drawContours(frame_ori, contour, -1, (0, 0, 255), 2)

	cv2.imshow("Original", frame_ori)
	cv2.imshow("Canny", canny)

	#cv2.waitKey(10)
	#break

	key = cv2.waitKey(1)

	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()