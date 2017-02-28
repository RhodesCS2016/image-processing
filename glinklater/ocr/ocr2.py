#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from '../data/digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   ocr2.py
'''


# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
from numpy.linalg import norm

from os.path import dirname, basename, isfile
import glob
import sys

from matplotlib import pyplot as plt

# local modules
from cv2Common import clock, mosaic

SZ = 64 # size of each digit is SZ x SZ
CLASS_N = 26
DIGITS_FN = '/opt/opencv/samples/data/digits.png'

def debug(img, name='debug'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def debugConcurrent(img, name='debug'):
    cv2.imshow(name, img)

def destroy():
    cv2.destroyAllWindows()

def preprocess_image(img):
    '''letter dimentions'''
    blurred = cv2.medianBlur(img,5)
    
    ret, thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1:]
    
    cv2.drawContours(thresh, contours, -1, 255, -1)
    try:
        minX, minY, maxX, maxY = (
            min([min(x[:, 0, 0]) for x in contours]),
            min([min(y[:, 0, 1]) for y in contours]),
            max([max(x[:, 0, 0]) for x in contours]),
            max([max(y[:, 0, 1]) for y in contours])
        )
        img = cv2.resize(img[minY:maxY, minX:maxX], (SZ, SZ))
    except:
        debug(img)
        sys.exit(1)
    ret, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_data():
    imtraining = {}
    data, labels = [], []
    folders = glob.glob(dirname(__file__) + '/samples/processed/*')
    for folder in folders:
        files = glob.glob(folder + '/*')
        for file in files:
            filename = basename(file)[:-4].replace('.', '-')
            imtraining[filename] = file
    for k in imtraining.keys():
        data.append(preprocess_image(cv2.imread(imtraining[k], 0)))
        labels.append(ord(k[0]) - ord('a'))
    return np.asarray(data), np.asarray(labels)

def watershed(img, thresh_val, kernel_size, dist_ratio, opening_iter=2, dilate_iter=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray,5)
    ret, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    # debugConcurrent(thresh, 'thresh')

    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iter)

    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iter)
    # debug(sure_bg)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_ratio*dist_transform.max(), 255, 0)
    # debugConcurrent(sure_fg, 'fg')

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # debug(unknown, 'unknown')

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    # img[markers == -1] = [0, 255, 0]
    # debug(markers)

    contourImg = np.zeros(thresh.shape, dtype='uint8')
    contourImg[markers == -1] = 255
    return contourImg

def separate_characters(img):
    contourImg = watershed(img, 240, (3, 3), 0.35)
    # debugConcurrent(contourImg, 'watershed')

    im2, contours, hierarchy = cv2.findContours(contourImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1:]
    contours = [c for c in contours if c.size > 99]
    split_chars = []
    
    for contour in contours:
        minX, minY, maxX, maxY = (
            min(contour[:, 0, 0]),
            min(contour[:, 0, 1]),
            max(contour[:, 0, 0]),
            max(contour[:, 0, 1])
        )
        split = img[minY:maxY, minX:maxX]
        # split = cv2.cvtColor(split, cv2.COLOR_BGR2GRAY)
        ret, split = cv2.threshold(split, 240, 255, cv2.THRESH_BINARY_INV)

        # debugConcurrent(split, 'char')
        # debugConcurrent(watershed(split, 240, (3, 3), 0.35), 'contour')
        # # cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)
        # debugConcurrent(img, 'img')

        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     sys.exit(0)
        split_chars.append(split)
        # cv2.rectangle(img, (minX, minY), (maxX, maxY), (0, 255, 0))
        
    # print(contours)
    return split_chars

def preprocess_test(img):
    chars = separate_characters(img)
    # for char in chars:
    #     debug(char)

    # TODO: Try and split the "double characters"

    return np.asarray([cv2.resize(cv2.cvtColor(char, cv2.COLOR_BGR2GRAY), (SZ, SZ)) for char in chars])

def load_test():
    img = cv2.imread('./samples/processed/test/1-3-1.png')
    img = preprocess_test(img)
    labels = []
    for char in img:
        debugConcurrent(char)
        labels.append((cv2.waitKey(0) & 0xFF) - ord('a'))
    return img, np.asarray(labels)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/Itseez/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    results = ''.join([chr(int(ch) + ord('a')) for ch in resp])
    expected = ''.join([chr(int(ch) + ord('a')) for ch in labels])
    for res, exp in zip(results, expected):
        if res == exp:
            print(res, exp, '*')
        else:
            print(res, exp)

    print('exp: ', expected)
    print('res: ', results)

    err = (labels != resp).mean()
    print('error: %.2f %%' % (err*100))

    # confusion = np.zeros((CLASS_N, CLASS_N), np.int32)
    # for i, j in zip(labels, resp):
    #     confusion[i, j] += 1
    # print('confusion matrix:')
    # print(confusion)
    # print()

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    # print(__doc__)
    print('loading...')

    data, labels = load_data()

    data_test, labels_test = load_test()
    destroy()
    # print(labels_test)
    debug(mosaic(20, data[:100]), 'training set')
    debug(mosaic(20, data_test), 'test set')

    # data = np.concatenate((data, data_test), axis=0)
    # print(data)
    # labels = np.concatenate((labels, labels_test), axis=0)
    # print(labels)

    print('preprocessing...')
    # shuffle digits
    data_test = list(map(deskew, data_test))
    samples_test = preprocess_hog(data_test)

    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(data))
    data_train, labels_train = data[shuffle], labels[shuffle]

    data_train = list(map(deskew, data))
    samples_train = preprocess_hog(data_train)

    print('training SVM...')
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, data_test, samples_test, labels_test)
    debug(vis, 'SVM test')
    vis = evaluate_model(model, data_train[100:], samples_train[100:], labels_train[100:])
    debug(vis, 'SVM test')
    print('saving SVM as "digits_svm.dat"...')
    model.save('digits_svm.dat')

    cv2.waitKey(0)
    destroy()
