import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from os.path import dirname, basename, isfile

SZ = 20
binNum = 16

def debug(image):
	cv2.imshow('debug', image)
	cv2.waitKey(0)

# test_image = cv2.imread('/samples/processed/samples1-1.jpg')

imtraining = []
folders = glob.glob(dirname(__file__) + '/samples/processed/*')
for folder in folders:
	imtraining += glob.glob(folder + '/*')

# image = cv2.imread(imtraining[0])

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
	return img

def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(binNum*ang/(2*np.pi))
	bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
	mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
	hists = [np.bincount(b.ravel(), m.ravel(), binNum) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist

img = cv2.imread('/opt/opencv/samples/data/digits.png', 0)
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

deskewed = [map(deskew, row) for row in train_cells]
hogdata = [map(hog, row) for row in deskewed]
traindata = np.float32(hogdata).reshape(-1,64)
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(traindata, cv2.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

deskewed = [map(deskew, row) for row in test_cells]
hogdata = [map(hog, row) for row in deskewed]
testdata = np.float32(hogdata).reshape(-1, binNum*4)
result = svm.predict_all(testdata)

mask = result == responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size

# __all__ = [basename(f)[:-3] for f in modules if isfile(f)]

# def register(cvstream):
#   for module in __all__:
#     if not module.startswith('__'):
#       operation = __import__(module, globals(), locals(), [module], -1).__dict__[module]()
#       operation.register(cvstream)
