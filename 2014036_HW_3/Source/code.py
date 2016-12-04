from __future__ import print_function
import struct
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.externals import joblib


def generate_roc(scoreMatrix,trueLabels,nROCpts =100 ,plotROC = 'false'):

    tpr = np.zeros([1,nROCpts]) 
    fpr = np.zeros([1,nROCpts]) 
    nTrueLabels = np.count_nonzero(trueLabels) 
    nFalseLabels = np.size(trueLabels) - nTrueLabels 
    
    minScore = np.min(scoreMatrix)
    maxScore = np.max(scoreMatrix);
    rangeScore = maxScore - minScore;
  
    
    thdArr = minScore + rangeScore*np.arange(0,1,float(1)/(nROCpts))
    #print thdArr
    for thd_i in range(0,nROCpts):
        thd = thdArr[thd_i]
        ind = np.where(scoreMatrix>=thd) 
        thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
        thisLabel[ind] = 1
        tpr_mat = np.multiply(thisLabel,trueLabels)
        tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
        fpr_mat = np.multiply(thisLabel, 1-trueLabels)
        fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels
        
        #print fpr
       # print tpr  
    if(plotROC == 'true'):
        plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
        plt.plot(fpr[0,:],tpr[0,:], 'b.-')
        
        plt.show()

    return fpr,tpr,thdArr
                             


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

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

def read_label(file_name, idx_label):
   ind_file = gzip.open(file_name, 'rb')
   
   ind_file.seek(8 + idx_label)
   label = ind_file.read(1)
   label = struct.unpack('>B', label)

   ind_file.close()
   return str(label[0])
   
def preprocess():
    train_x = [[] for x in xrange(10)]
    train_y = [[] for x in xrange(10)]
    test_x = [[] for x in xrange(10)]
    test_y = [[] for x in xrange(10)]
    print('hi')
    for i in range(60000):
        lab = read_label('train-labels-idx1-ubyte.gz', i)
        img = read_image('train-images-idx3-ubyte.gz', i)
        img = img.reshape(1, 784)
        img = flatten(img)
        img = [int(i) for i in img]
        if( len(train_x[ int(lab) ])!=2000 ):
            train_x[ int(lab) ].append( img )
            train_y[ int(lab) ].append( int(lab) )
        flag = 0
        for j in range(10):
          if( len( train_y[ j ] )!=2000 ):
              flag = 1
        
        if( flag==0 ):
            break
        
    print('Hi')      
    for i in range(10000):
        lab = read_label('t10k-labels-idx1-ubyte.gz', i)
        img = read_image('t10k-images-idx3-ubyte.gz', i)
        img = img.reshape(1, 784)
        img = flatten(img)
        img = [int(i) for i in img]
        if( len(test_y[ int(lab) ])!=500 ):
            test_x[ int(lab) ].append( img )
            test_y[ int(lab) ].append( int(lab) )
        flag = 0
        for j in range(10):
          if( len( train_y[ j ] )!=500 ):
              flag = 1
        
        if( flag==0 ):
            break           
     
    return train_x, train_y, test_x, test_y    
    
# 49, 3119, 7
########### Part A ##############
train_x, train_y, test_x, test_y = preprocess()

# train of 3 and 8
trainlabel = [] 
trainimage = []
trainlabel = np.array( trainlabel ) 
trainimage = np.array( trainimage )

for i in range(2000):
    trainlabel = np.append( trainlabel, 0 )
    trainimage = np.append( trainimage, train_x[3][i] )
for i in range(2000):
    trainlabel = np.append( trainlabel, 1 )
    trainimage = np.append( trainimage, train_x[8][i] )    
    
trainlabel = np.reshape( trainlabel, (4000, 1) )
trainimage = np.reshape( trainimage, (4000, 28*28) )
trainlabel = flatten( trainlabel )

# test of 3 and 8
testlabel = [] 
testimage = []
testlabel = np.array( testlabel ) 
testimage = np.array( testimage )

for i in range(500):
     testlabel = np.append( testlabel, 0 )
     testimage = np.append( testimage, test_x[3][i] )
for i in range(500):
     testlabel = np.append( testlabel, 1 )
     testimage = np.append( testimage, test_x[8][i] )     
              
testlabel = np.reshape( testlabel, (1000, 1) )
testimage = np.reshape( testimage, (1000, 28*28) )
testlabel = flatten( testlabel )

param_grid = { 'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000] }
svr = SVC()
grid = GridSearchCV(svr, param_grid, cv = 5)
grid.fit(trainimage, trainlabel)

joblib.dump(grid, '../Models/model_linear.pkl', compress=1)

pred = np.ndarray(shape=(1000, 1))
for i in range(1000):
    pred[i] = grid.predict( testimage[i].reshape(1, 784) )
pred = flatten( pred )

#false_positive_rate, true_positive_rate, thresholds = roc_curve(testlabel, pred, pos_label=1)
#roc_auc = auc(false_positive_rate, true_positive_rate)

#plt.title( 'Receiver Operating Characteristic' )  
#plt.plot( false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc )
#plt.legend( loc = 'lower right' )
#plt.plot([0,1], [0,1], 'r--')
#plt.xlim([-0.1, 1.2])
#plt.ylim([-0.1, 1.2])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

'''
############## Part B ###############
train_x, train_y, test_x, test_y = preprocess()

trainlabel = [] 
trainimage = []
trainlabel = np.array( trainlabel ) 
for j in range(10):
     for i in range(2000):
         trainlabel = np.append( trainlabel, train_y[j][i] )
         trainimage = np.append( trainimage, train_x[j][i] )
            
trainlabel = np.reshape( trainlabel, (20000, 1) )
trainimage = np.reshape( trainimage, (20000, 28*28) )
trainlabel = flatten( trainlabel ) 
#trainlabel = label_binarize( trainlabel, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )         
    
param_grid = { 
   'estimator__C': [0.1, 1, 10, 100, 1000] 
}
svr = OneVsRestClassifier( LinearSVC() )
grid = GridSearchCV(svr, param_grid, cv = 5)
grid.fit(trainimage, trainlabel)
joblib.dump(grid, '../Models/multi.pkl', compress=1)

testlabel = [] 
testimage = []
testlabel = np.array( testlabel ) 
testimage = np.array( testimage )

for j in range(10):
    for i in range(500):
        testlabel = np.append( testlabel, test_y[j][i] )
        testimage = np.append( testimage, test_x[j][i] )
        
testlabel = np.reshape( testlabel, (5000, 1) )
testimage = np.reshape( testimage, (5000, 28*28) )
testlabel = flatten( testlabel )

#testlabel = label_binarize( testlabel, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )

#svr = OneVsRestClassifier( LinearSVC(C=0.1) )
#svr.fit(trainimage, trainlabel)
#print(svr.score(testimage, testlabel))

#svr = OneVsRestClassifier( LinearSVC(C=1) )
#svr.fit(trainimage, trainlabel)
#print(svr.score(testimage, testlabel))

#svr = OneVsRestClassifier( LinearSVC(C=10) )
#svr.fit(trainimage, trainlabel)
#print(svr.score(testimage, testlabel))

#svr = OneVsRestClassifier( LinearSVC(C=100) )
#svr.fit(trainimage, trainlabel)
#print(svr.score(testimage, testlabel) )

#svr = OneVsRestClassifier( LinearSVC(C=1000) )
#svr.fit(trainimage, trainlabel)
#print(svr.score(testimage, testlabel))

#n_classes = 10
#testscore = grid.decision_function( testimage )
#print( 'H3' )
    
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(testlabel[:, i], testscore[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
#all_fpr = np.unique( np.concatenate([ fpr[i] for i in range(n_classes) ]) )

# Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
#mean_tpr /= n_classes

#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#plt.figure()
#plt.plot(fpr["macro"], tpr["macro"], 
#         label='macro-average ROC curve (area = {0:0.2f})'
#         ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)
         
#plt.plot( [0, 1], [0, 1], 'k--', lw=2 )

#plt.xlim( [0.0, 1.0] )
#plt.ylim( [0.0, 1.05] )
#plt.xlabel( 'False Positive Rate' )
#plt.ylabel( 'True Positive Rate' )
#plt.title( 'Some extension of Receiver operating characteristic to multi-class' )
#plt.show()
'''

'''
########### Part C #############
train_x, train_y, test_x, test_y = preprocess()

trainlabel = [] 
trainimage = []
trainlabel = np.array( trainlabel ) 
trainimage = np.array( trainimage )
    
for j in range(10):
     for i in range(2000):
         trainlabel = np.append( trainlabel, train_y[j][i] )
         trainimage = np.append( trainimage, train_x[j][i] )
            
trainlabel = np.reshape( trainlabel, (20000, 1) )
trainimage = np.reshape( trainimage, (20000, 28*28) )
trainlabel = flatten( trainlabel ) 
#trainlabel = label_binarize( trainlabel, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )

param_grid = { 'estimator__C': [0.1, 1, 10, 100, 1000], 'estimator__gamma': [0.000001, 0.00001, 0.0001, 1, 0.1, 10, 100] }
svr = OneVsRestClassifier( SVC(kernel='rbf', max_iter = 1500) )
grid = GridSearchCV(svr, param_grid, cv = 5)
grid.fit(trainimage, trainlabel)
joblib.dump(svr, "../Models/rbf.pkl", compress = 1)

testlabel = [] 
testimage = []
testlabel = np.array( testlabel ) 
testimage = np.array( testimage )

for j in range(10):
    for i in range(500):
        testlabel = np.append( testlabel, test_y[j][i] )
        testimage = np.append( testimage, test_x[j][i] )
        
testlabel = np.reshape( testlabel, (5000, 1) )
testimage = np.reshape( testimage, (5000, 28*28) )
testlabel = flatten( testlabel )
#testlabel = label_binarize( testlabel, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )

#n_classes = 10
#testscore = grid.decision_function( testimage )
    
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(testlabel[:, i], testscore[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
#all_fpr = np.unique( np.concatenate([ fpr[i] for i in range(n_classes) ]) )

# Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
#mean_tpr /= n_classes

#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#plt.figure()
#plt.plot(fpr["macro"], tpr["macro"], 
#         label='macro-average ROC curve (area = {0:0.2f})'
#         ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)
         
#plt.plot( [0, 1], [0, 1], 'k--', lw=2 )

#plt.xlim( [0.0, 1.0] )
#plt.ylim( [0.0, 1.05] )
#plt.xlabel( 'False Positive Rate' )
#plt.ylabel( 'True Positive Rate' )
#plt.title( 'Some extension of Receiver operating characteristic to multi-class' )
#plt.show()
'''

'''
##################################### Extra #######################################
image = fir_trainimage[0]
img_plot = plt.imshow(image, 'Greys')
plt.show()
print( fir_trainlabel[0] )
'''