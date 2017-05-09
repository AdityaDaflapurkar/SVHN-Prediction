import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#from sklearn import tree


x=np.load('train_X.npy')
y=np.load('train_Y.npy')
tx=np.load('test_X.npy')
#print x[1]
pca = PCA(n_components=49)
x=np.reshape(x,(len(x),784))
pca.fit(x)
#x=(x>0.3)*1
#print np.shape(x)

#clf = RandomForestClassifier(n_estimators=10,verbose=True,random_state=0)
#clf = LogisticRegression( multi_class='ovr', random_state=1, verbose=1)
clf =  KMeans(n_clusters=10, random_state=0, verbose=1, tol=1,  n_init=1).fit(x)
clf = clf.fit(x, y)
#tx=(tx>0.3)*1
p=clf.predict(tx)

with open("Output.txt", "w") as text_file:
	
	for i in range(len(p)):
		text_file.write("%s\n" % p[i])


