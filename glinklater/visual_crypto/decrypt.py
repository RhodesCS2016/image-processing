import cv2
import numpy as np
import sys
import random

def debug(image):
	cv2.imshow('debug', image)
	cv2.waitKey(0)

share1 = cv2.imread('share1.png')
share2 = cv2.imread('share2.png')

debug(cv2.bitwise_xor(share1, share2))
