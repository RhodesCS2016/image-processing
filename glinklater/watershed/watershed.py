import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

def debug_display(image):
	cv.imshow('Debug', image)
	cv.waitKey(0)
	cv.destroyAllWindows()

file = None
try:
	print sys.argv[1]
	file = sys.argv[1]
except:
	print 'Please specify an input file'
	sys.exit(1)

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

debug_display(thresh)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv.dilate(opening, kernel, iterations=3)

dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

debug_display(sure_bg)
debug_display(sure_fg)
debug_display(unknown)

ret, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

debug_display(img)
