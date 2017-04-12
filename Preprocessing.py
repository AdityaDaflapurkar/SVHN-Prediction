import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imread_collection,imshow
import csv
import os.path
from skimage.transform import resize

class DataProcessing:
# This file should be located inside the parent directory of the svhn directory("release") where the data is stored.  
 
	def get_images(self, imgpath):
	# Creating a collection with the available images
		print "Collecting images..."	
		imgset = imread_collection(imgpath)
		print len(imgset)," images collected."
		return imgset


	def generate_data(self, datapath, n):
	# Read data from csv and convert it to a sequence of digit lists 
		df=pd.read_csv(datapath)
		nums=[]
		print "Generating data... "
		for i in range(1,n+1):
			current_df = df.loc[df['FileName'] == str(i)+".png"]
			x_list = current_df["Left"].tolist()
			y_list = current_df["Top"].tolist()
			dx = current_df["Width"].tolist()
			dy = current_df["Height"].tolist()
			
			# Digits list
			digits = current_df["DigitLabel"].tolist()
			
			# Top-left corner coordinates(x1,y1) 
			x1 = min(x_list)
			y1 = min(y_list)
			
			max_x = max(x_list)
			max_y = max(y_list)
			max_id_x = x_list.index(max_x)
			max_id_y = y_list.index(max_y)

			# Bottom-right corner coordinates(x2,y2)
			x2 = max_x + dx[max_id_x]
			y2 = max_y + dy[max_id_y]

			if i%100==0:
				progress = i*100.0/(n+1)
				print "%.2f  percent done." % progress

			current_record = self.get_record(digits, x1, y1, x2, y2)
			nums.append(current_record)

		print "100 percent done."
		print "Data generation successful."
		return nums


	def get_record(self, digits, x1, y1, x2, y2):
	# record in csv = [x1, y1, x2, y2, no_of_digits, digits, padding]	
		record = []
		
		record.append(x1)
		record.append(y1)
		record.append(x2)
		record.append(y2)
		record.append(len(digits))
		record = record + digits
		padding = [-1]*(11-len(record))
		record = record + padding
		
		return record


	def write_to_csv(self, nums):
	# Create target csv to get list of digits for each image record
		datafile = open("svhn.csv", 'wb')
		wr = csv.writer(datafile, quoting=csv.QUOTE_ALL)
		dummy_header=["x1","y1","x2","y2","length","digit1","digit2","digit3","digit4","digit5","digit6"]
		wr.writerow(dummy_header)
		wr.writerows(nums)
		

	def k_fold_split(self, dataset, imageset,k):
	# Splitting dataset and image set into k subsets 
		ilist = []		
		dlist = []
		l=len(imageset)
		b=0
		for i in xrange(k-1):
			ilist.append(imageset[b:(l/k)+b])
			dlist.append(dataset.ix[b:(l/k)+b-1])
			#print dataset.ix[b:(l/k)+b-1]
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
		nums = dp.generate_data(datapath,len(images))
		dp.write_to_csv(nums)
	
	datapath = 'svhn.csv'
	df=pd.read_csv(datapath)
	
	ilist, dlist = dp.k_fold_split(df,images,6)
	#print "x1:",df.get_value(0,"x1")," x2:",df.get_value(0,"x2")," y1:",df.get_value(0,"y1")," y2:",df.get_value(0,"y2")
	#print np.shape(images[0])
	crop=images[0][77:300,246:419,:]
	#print np.shape(crop)
	#print crop	
	#viewer = ImageViewer(images[0])
	#viewer.show()
	#viewer2 = ImageViewer(crop_image)
	reshape=resize(crop, (64,64),mode='constant')
	imshow(reshape,plugin=None)
	plt.show()
