import cv2
import os
import sys

import config

def load_batch(meta_file):
    batch_0 = []
    batch_1 = []

    lines = open(meta_file).readlines()
    for line in lines:
        line = line.strip()
        filename, classId = line.split('\t')

        img = load_img(filename)

        batch_0.append(img)
        batch_1.append(int(classId))

    return([batch_0, batch_1], len(lines))

def load_img(filename):
    img = cv2.imread(filename, 0)
    img = preprocess(img)
    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return img

def preprocess(img):
    return cv2.equalizeHist(img)

if __name__ == "__main__":
    load_batch(sys.argv[1])
