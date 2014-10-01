#!/usr/bin/env python

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from iccv07 import *
from dataset import *

N_pos = 2500
N_neg = 12000

dataset_name = "SVMs/newest/_svm_"

#train_set = iccv07(seq=3)
train_set = dataset("Mixed_data_set")
test_set = iccv07(seq=1)


pos_tr_imgs = train_set.get_pos_images(N_pos)
neg_tr_imgs = train_set.get_neg_images(N_neg)

samples = np.vstack((np.array(neg_tr_imgs), np.array(pos_tr_imgs)))
print 'Number of positive training images: ', len(pos_tr_imgs)
print 'Number of negative training images: ', len(neg_tr_imgs)

# calculate HOG features

def computeHOG(data):
	hog = cv2.HOGDescriptor()
	X = np.zeros((len(data), 3780))
	i = 0
	for t in data:
		X[i,:] = hog.compute(t).T
		i += 1
	return X

X = computeHOG(samples)

#labeling

Y = np.concatenate((np.repeat("background",len(neg_tr_imgs),0), np.repeat("pedestrian",len(pos_tr_imgs),0))) 

# SVM
print 'training SVM...'
svm = svm.SVC(kernel="rbf", gamma=0.0, C=1.0, probability=True)
svm.fit(np.array(X), Y)
print 'done'

print 'saving classifier...'
joblib.dump(svm, "%s.pkl"%dataset_name)
