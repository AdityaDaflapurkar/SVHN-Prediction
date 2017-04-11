import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imread_collection
import csv
import os.path

class DataProcessing:
# This file should be located inside the parent directory of the svhn directory("release") where the data is stored.  
 
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
		print "Generating data... "
		for i in range(0,n+1):
			c=df.loc[df['FileName'] == str(i)+".png"]["DigitLabel"].tolist()
			# Alternative approach :
			# img="'"+str(i)+'.png'+"'"		
			# c=df.query("FileName == "+img)['DigitLabel']
			if i%100==0:
				print i*100.0/(n+1)," percent done."
			nums.append(c)
		print "Successful."
		return nums


	def write_to_csv(self, nums):
	# Create target csv to get list of digits for each image record
		datafile = open("svhn.csv", 'wb')
		wr = csv.writer(datafile, quoting=csv.QUOTE_ALL)
		dummy_header=[0,0,0,0,0,0]
		wr.writerow(dummy_header)
		wr.writerows(nums[1:])
		

	def k_fold_split(self, dataset, imageset,k):
	# Splitting dataset and image set into k subsets 
		ilist = []		
		dlist = []
		l=len(imageset)
		b=0
		for i in xrange(k-1):
			ilist.append(imageset[b:(l/k)+b])
			dlist.append(dataset.ix[b:(l/k)+b-1])
			b=(l/k)+b
		ilist.append(imageset[b:])
		dlist.append(dataset.ix[b:])
		return ilist, dlist

	
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
	
	dp=DataProcessing()
	images=dp.get_images("svhn/data/train_images/*.png")
	if(not(os.path.isfile("svhn.csv"))):
		datapath = 'svhn/data/train.csv'
		nums = dp.read_data(datapath,len(images))
		dp.write_to_csv(nums)
	
	datapath = 'svhn.csv'
	df=pd.read_csv(datapath)
	
	ilist, dlist = dp.k_fold_split(df,images,6)
	 
	#for i in xrange(len(ilist)):
	#	print len(ilist[i]),"img",i
	#	print len(dlist[i]),"data",i
	
