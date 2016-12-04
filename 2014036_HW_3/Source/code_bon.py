from __future__ import print_function
import struct
import math
import gzip
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier as K_
from scipy import interp
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import KernelPCA as P_
from scipy import linalg

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# Reading the image file
# '>I' makes x into a 4-byte string in big-endian order
def read_image(file_name, idx_image):
   img_file = gzip.open(file_name, 'rb')
   
   image = np.ndarray(shape=(28, 28))
   img_file.seek(16 + 28*28*idx_image)
	
   for row in range(28):
     for col in range(28):
         tmp_d = img_file.read(1)
         tmp_d = struct.unpack('>B',tmp_d)
         image[row, col] = tmp_d[0]

   img_file.close()
   return image

# Reading the label file
def read_label(file_name, idx_label):
   ind_file = gzip.open(file_name, 'rb')
   
   ind_file.seek(8 + idx_label)
   label = ind_file.read(1)
   label = struct.unpack('>B', label)

   ind_file.close()
   return str(label[0])
   
# Preprocessing the data   
def preprocess():
    train_x = [[] for x in xrange(10)]
    train_y = [[] for x in xrange(10)]
    test_x = [[] for x in xrange(10)]
    test_y = [[] for x in xrange(10)]

    for i in range(60000):
        lab = read_label('train-labels-idx1-ubyte.gz', i)
        img = read_image('train-images-idx3-ubyte.gz', i)
        img = img.reshape(1, 784)
        img = flatten(img)
        img = [int(i) for i in img]
        if( len(train_x[ int(lab) ])!=150 ):
            train_x[ int(lab) ].append( img )
            train_y[ int(lab) ].append( int(lab) )
        flag = 0
        for j in range(10):
          if( len( train_y[ j ] )!=150 ):
              flag = 1
        
        if( flag==0 ):
            break

    for i in range(10000):
        lab = read_label('t10k-labels-idx1-ubyte.gz', i)
        img = read_image('t10k-images-idx3-ubyte.gz', i)
        img = img.reshape(1, 784)
        img = flatten(img)
        img = [int(i) for i in img]
        if( len(test_y[ int(lab) ])!=150 ):
            test_x[ int(lab) ].append( img )
            test_y[ int(lab) ].append( int(lab) )
        flag = 0
        for j in range(10):
          if( len( test_y[ j ] )!=150 ):
              flag = 1
        
        if( flag==0 ):
            break   
     
    return train_x, train_y, test_x, test_y    

# finding k-nearest neighbours    
def find_neigh(train_, test_, k):
       opop = len(train_) 
       neighbors = []
       lenn = []
       for x in range(opop):
		dist = euclidean( test_, train_[x] )
		lenn.append( (train_[x], dist) ) 
       lenn.sort( key = operator.itemgetter(1) )
       for i in range(k):
		neighbors.append(lenn[i][0])
       return neighbors 
     
# PErform KPCA
def kp(X):
  num_data, dim = X.shape

  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X
  if( dim>100 ):
      M = DotProduct(X, X.T) 
      e, EV = linalg.eigh(M) # eigenvalues and eigenvectors
      tmp = DotProduct(X.T, EV).T
      V = tmp[::-1]     
      S = math.sqrt(e)[::-1]  
  else:
      U, S, V = linalg.svd(X)
      V = V[:num_data]    

  return V,S,mean_X
  

# To predict       
def pred(neighbors):
	priori = {}
	for i in range(len(neighbors)):
		pre = neighbors[i][-1]
		if pre in priori:
			priori[pre] += 1
		else:
			priori[pre] = 1
	return sorted(priori.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

print('Hi1')
train_x, train_y, test_x, test_y = preprocess()
print(test_y)    
trainlabel = [] 
trainimage = []
trainlabel = np.array( trainlabel ) 
trainimage = np.array( trainimage )
    
for j in range(10):
     for i in range(150):
         trainlabel = np.append( trainlabel, train_y[j][i] )
         trainimage = np.append( trainimage, train_x[j][i] )
            
trainlabel = np.reshape( trainlabel, (1500, 1) )
trainimage = np.reshape( trainimage, (1500, 28*28) )
trainlabel = flatten( trainlabel ) 
print('Hi2')
#param_grid = { 'estimator__gamma': [0.1, 1, 10, 100, 1000] }
#grid = GridSearchCV(neigh, param_grid, cv = 5)
    
kpca = P_( kernel = 'rbf', fit_inverse_transform = True, gamma = 0.1 )
X_kpca = kpca.fit_transform(trainimage)  
neigh = K_(n_neighbors=3)
neigh.fit(X_kpca, trainlabel) 

testlabel = [] 
testimage = []
testlabel = np.array( testlabel ) 
testimage = np.array( testimage )   

for j in range(10):
    for i in range(150):
        testlabel = np.append( testlabel, test_y[j][i] )
        testimage = np.append( testimage, test_x[j][i] )
        
testlabel = np.reshape( testlabel, (1500, 1) )
testimage = np.reshape( testimage, (1500, 28*28) )
testlabel = flatten( testlabel )
#print('Hi3')
X_test = kpca.fit_transform(testimage)  
print( neigh.predict(X_test[0]) )
print( neigh.score(X_test, testlabel) )
'''
n_classes = 10
testscore = neigh.predict_proba(X_kpca)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testlabel[:, i], testscore[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique( np.concatenate([ fpr[i] for i in range(n_classes) ]) )

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["macro"], tpr["macro"], 
         label='macro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
         
plt.plot( [0, 1], [0, 1], 'k--', lw=2 )

plt.xlim( [0.0, 1.0] )
plt.ylim( [0.0, 1.05] )
plt.xlabel( 'False Positive Rate' )
plt.ylabel( 'True Positive Rate' )
plt.title( 'Some extension of Receiver operating characteristic to multi-class' )
plt.show()
'''