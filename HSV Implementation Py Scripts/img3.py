import cv2
import numpy as np

frame = cv2.imread('trial.jpg')

frame = cv2.resize(frame, (400, 400))
frame_ori = np.copy(frame)
frame = cv2.GaussianBlur(frame, (5, 5), 0)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_pink = np.array([7, 130, 122])
upper_pink = np.array([25, 255, 255])

mask = cv2.inRange(hsv, lower_pink, upper_pink)

canny = cv2.Canny(mask, 60, 80)

contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
	if (cv2.contourArea(contour) > 100) and (cv2.contourArea(contour) < 200):
		cv2.drawContours(frame_ori, contour, -1, (0, 255, 0), 1)

cv2.imshow("Original", frame_ori)
cv2.imshow("Mask", mask)
cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()