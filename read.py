import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imread_collection

# path 
col_dir = 'svhn/data/train_images/*.png'

# creating a collection with the available images
col = imread_collection(col_dir)
print len(col)

