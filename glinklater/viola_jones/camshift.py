import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
r, h, c, w = 250, 90, 400, 150
track_window = (c, r, w, h)

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
	ret, frame = cap.read()

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

	ret, track_window = cv2.CamShift(dst, track_window, term_crit)

	pts = cv2.boxPoints(ret)
	pts = np.int0(pts)
	img2 = cv2.polylines(frame, [pts], True, 255, 2)
	cv2.imshow('CamShift', img2)

	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()