import re
import numpy as np
import cv2
import random
import os

base_path = 'iccv07-data/'
ann_path = base_path+'annotations/'
im_path = base_path+'images/'

class iccv07:

	def __init__(self, seq=1):
		self.neg_coords = []
		self.annotations = 'pedxing-seq'+str(seq)+'-annot.idl.txt'
		f = open(ann_path+self.annotations, "r")

		pattern = re.compile("\([0-9]*, [0-9]*, [0-9]*, [0-9]*\)")
		annotation_dic = {}
		image_list = []

		for line in f:
			r = re.match("\".*\"", line)
			file_name = r.group(0)[1:-1]
			coordinates = pattern.findall(line)
			annotation_dic[file_name] = coordinates
			image_list.append(file_name)
		self.image_list = image_list
		self.annotation_dic = annotation_dic

	def get_image_path(self):
		return im_path
	
	def get_image_list(self):
		return self.image_list

	def get_coordinates(self, im_name):
		coordinates =  self.annotation_dic[im_name]
		return coordinates

	def sort_coordinates(self, c):
		if c[0] > c[2]:
			c = (c[2], c[1], c[0], c[3])
		if c[1] > c[3]:
			c = (c[0], c[3], c[2], c[1])
		return c

	def overlapp(self, x,y, ex_region, patchsize = (64, 128)):
		ranges = [self.sort_coordinates(eval(c)) for c in ex_region]
		if any(x_min-patchsize[0] <= x <= x_max or y_min-patchsize[1] <= y <= y_max for (x_min, y_min, x_max,y_max) in ranges):
			return True
		return False

	def randomPatch(self, im, ex_region, patchsize = (64, 128)):
		sz = im.shape
		x = int(np.floor(np.random.uniform(low = 0, high = sz[1]-patchsize[0])))
		y = int(np.floor(np.random.uniform(low = 0, high = sz[0]-patchsize[1])))
		iterations = 0
		while iterations < 10:
			x = int(np.floor(np.random.uniform(low = 0, high = sz[1]-patchsize[0])))
			y = int(np.floor(np.random.uniform(low = 0, high = sz[0]-patchsize[1])))
			iterations += 1
			if self.overlapp(x,y, ex_region):
				break;
		return (im[y:y+patchsize[1],x:x+patchsize[0]], (int(x),int(y)))

	def get_pos_images(self, N):
		return self.extract_pos_patches()[0:N]

	def extract_pos_patches(self):
		pos_patches = []
		for image in self.image_list:
			im = cv2.imread(im_path + image,0)
			coordinates = self.annotation_dic[image]
			for c in coordinates:
				c_tup = self.sort_coordinates(eval(c))
				patch = im[c_tup[1]:c_tup[3],c_tup[0]:c_tup[2]]
				patch = cv2.resize(patch, (64, 128))
				pos_patches.append(patch)
		return pos_patches

	def get_neg_images(self, N):
		N = int(N/len(self.image_list))
		return self.extract_neg_patches(N)


	def extract_neg_patches(self, N):
		self.neg_coords = []     
		neg_patches = []
		for image in self.image_list:
			for n in range(N):
				im = cv2.imread(im_path + image,0)
				coordinates = self.annotation_dic[image]
				(patch, coord) = self.randomPatch(im, coordinates)
				neg_patches.append(patch)
				self.neg_coords.append(coord)			
		return neg_patches

	def save_pos_patches(self):
		images = self.extract_pos_patches()
		n = 0
		for im in images:
			file = "{}pos_{}.png".format(os.getcwd() + "/training_images/pos/", n)
			cv2.imwrite(file, im)
			n += 1

	def save_neg_patches(self):
		images = self.extract_neg_patches(5)
		n = 0
		for im in images:
			file = "{}neg_{}.png".format(os.getcwd() + "/training_images/neg/", n)
			cv2.imwrite(file, im)
			n += 1
