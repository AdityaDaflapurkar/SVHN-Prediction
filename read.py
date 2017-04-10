import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imread_collection
import csv
import os.path

class DataProcessor:
	def get_images(self, imgpath):
	# Creating a collection with the available images
		print "Collecting images..."	
		imgset = imread_collection(imgpath)
		print len(imgset)," images collected."
		return imgset

	def read_data(self, datapath, n):
	# Read data from csv and convert it to a sequence of digit lists 
		df=pd.read_csv(datapath)
		nums=[]
		print "Generating data... ",n
		for i in xrange(n):
			c=df.loc[df['FileName'] == str(i)+".png"]["DigitLabel"].tolist()
			# Alternative approach :
			# img="'"+str(i)+'.png'+"'"		
			# c=df.query("FileName == "+img)['DigitLabel']
			if (i+1)%100==0:
				print (i+1)*100.0/n," percent done."
			nums.append(c)
		return nums

	def write_to_csv(self, nums):
	# Create target csv to get list of digits for each image record
		datafile = open("svhn.csv", 'wb')
		wr = csv.writer(datafile, quoting=csv.QUOTE_ALL)
		for i in range(1,len(nums)):
			wr.writerow(nums[i])


	def k_fold_split(self, k):
	# Splitting dataset and image set into k subsets 
		dlist = np.array_split(df, k)
		ilist = np.array_split(df, k)
		return dlist, ilist

	
	def cross_validate(self, x, y):
	# Cross-validation
		n=len(x)
		for i in xrange(n):
			test_x = x[i]
			test_y = y[i]
			train_x = x[:i]+x[i+1:]
			train_y = y[:i]+y[i+1:]
			# TODO : Train on train_x,train_y, test on test_x, test_y

if __name__ == '__main__':
	
	dp=DataProcessor()
	images=dp.get_images("svhn/data/train_images/*.png")
	if(not(os.path.isfile("svhn.csv"))):
		datapath = 'svhn/data/train.csv'
		nums = dp.read_data(datapath,3)
		print nums
		dp.write_to_csv(nums)
	
	datapath = 'svhn.csv'
	df=pd.read_csv(datapath)
	dlist, ilist = dp.k_fold_split(5)
	
