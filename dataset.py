
import numpy as np
import cv2
import glob

neg_path = 'negative/'
pos_path = 'positive/'
test_path = 'test/'

class dataset:

	def __init__(self, name):
		print 'dataset:', name
		self.base_path = name
		self.image_path = ""
	
	def image_path(self):
		return self.image_path


	def load_images(self, names, resize=False):
		result = []
		for im in names:
			img = cv2.imread(im, 0)
			if resize:
				img = cv2.resize(img, (64,128))
			result.append(img)
		return result

	def get_pos_images(self, N):
		self.image_path = self.base_path +'/'+ pos_path
		print self.image_path
		images = glob.glob(self.image_path + '*.png')
		return self.load_images(images[:N], resize=True)
		
	def get_neg_images(self, N):
		self.image_path = self.base_path +'/'+ neg_path
		images = glob.glob(self.image_path + '*.png')
		return self.load_images(images[:N], resize=True)

	def get_test_images(self):
		self.image_path = self.base_path +'/'+ test_path
		images = glob.glob(self.image_path + '*.png')
		return self.load_images(images)
