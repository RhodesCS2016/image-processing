import cv2
import numpy as np
import sys
import random

def debug(image):
	cv2.imshow('debug', image)
	cv2.waitKey(0)

image = cv2.imread('pikachu.png')
debug(image)

img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
not_mask = cv2.bitwise_not(mask)

debug(not_mask)

random1 = np.full(not_mask.shape, 0, dtype=np.uint8)
for line in random1:
	for i, px in enumerate(line):
		line.itemset(i, random.randrange(0, 2) * 255)

random2 = cv2.bitwise_xor(random1, not_mask)

debug(random1)
debug(random2)
debug(cv2.bitwise_xor(random1, random2))

cv2.imwrite('share1.png', random1)
cv2.imwrite('share2.png', random2)