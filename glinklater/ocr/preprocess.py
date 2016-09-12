import cv2
import numpy as np
import sys
from os.path import isfile, basename, dirname
import glob

GRIDSIZE_Y = 8
GRIDSIZE_X = 12
blacklist = None

with open('blacklist.txt', 'r') as file:
	ls = file.readlines()
	blacklist = [line.split(' ') for line in ls]

def isBlacklisted(letter, sample, scan, idx):
	ret = False
	for b in blacklist:
		if (b[0] == letter and 
			b[1] == sample and 
			b[2] == scan and 
			b[3][:-1] == str(idx)):
			ret = True
	return ret

def debug(image):
	cv2.imshow('debug', image)
	cv2.waitKey(0)
	cv2.destroyWindow('debug')

def detect_grid(name):
	raw_image = cv2.imread(name)

	img2gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)

	edges = cv2.Canny(mask, 100, 200, 3, 5)

	lines = cv2.HoughLinesP(edges, 2, np.pi/180, 200, minLineLength=300, 
		maxLineGap=500)

	minX = min(np.concatenate((lines[:, :, 0][:, 0], lines[:, :, 2][:, 0])))
	maxX = max(np.concatenate((lines[:, :, 0][:, 0], lines[:, :, 2][:, 0])))
	minY = min(np.concatenate((lines[:, :, 1][:, 0], lines[:, :, 3][:, 0])))
	maxY = max(np.concatenate((lines[:, :, 1][:, 0], lines[:, :, 3][:, 0])))
	return raw_image[minY:maxY, minX:maxX]

def divide_grid(image, letter, sample, scan, directory):
	xSize, ySize = image.shape[1], image.shape[0]
	xBlockSize, yBlockSize = (xSize // GRIDSIZE_X), (ySize // GRIDSIZE_Y)

	for y in range(GRIDSIZE_Y):
		for x in range(GRIDSIZE_X):
			minX, maxX = (xBlockSize * x) + 40, (xBlockSize * (x + 1)) - 40
			minY, maxY = (yBlockSize * y) + 40, (yBlockSize * (y + 1)) - 40
			divided = image[minY:maxY, minX:maxX]

			rotation_matrix = cv2.getRotationMatrix2D((divided.shape[1] // 2, 
				divided.shape[0] // 2), 90, 1)
			rotated = cv2.warpAffine(divided, rotation_matrix, 
				(divided.shape[0], divided.shape[1]))

			idx = (y * GRIDSIZE_X) + x

			if not isBlacklisted(letter, sample, scan, idx):
				cv2.imwrite(directory + '%s/%s-%s-%s.%d.png' % 
					(letter, letter, sample, scan, idx), rotated)

def process_image(path):
	filename = basename(path)[:-4]
	letter = filename[0]
	sample_number, scan_number = filename[1:].split('-')
	detected = detect_grid(path)
	divide_grid(detected, letter, sample_number, scan_number, processed_folder)

processed_folder = dirname(__file__) + '/samples/processed/'
raw_folder = dirname(__file__) + '/samples/raw/'
samples = glob.glob(raw_folder + 'samples*.jpg')
files = glob.glob(raw_folder + '*.jpg')

for file in files:
	process_image(file)
