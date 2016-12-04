import numpy as np
import csv
import os
import itertools
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn import metrics
from sklearn.mixture import GMM 
from sklearn.manifold import TSNE

def rand_score( X, labels_true, labels_pred ):
     correct = 0
     total = 0

     arr2 = []     
     opop = len(X)
     for i in range(opop):
         arr2.append( i )
     
     for index_combo in itertools.combinations(arr2, 2):
          index1 = index_combo[0]
          index2 = index_combo[1]
                
          same_class = (labels_true[index1] == labels_true[index2])
          same_cluster = (labels_pred[index1] == labels_pred[index2])
    
          if same_class and same_cluster:
              correct += 1
          elif not same_class and not same_cluster:
              correct += 1
                
          total += 1
            
     return float(correct) / total

#######################################################################
# File to be scanned 
stt = input("Enter file Name: ")
# No of clusters
CLUS = eval( input("Enter no of clusters you want: ") )                                         
#######################################################################

# h -> no of rows
# w -> no of col
h = 0
w = 0
ary = []    
                
fn = os.path.join(os.path.dirname(__file__), stt)
                                          
if( stt=='iris_data.txt' ):
   limiter = ',' 
   g_truth = 3  
elif( stt=='column_3C.dat' ):                                         
   limiter = ' '
   g_truth = 3
elif( stt=='segmentation.txt' ):
   limiter = ','  
   g_truth = 7
else:
   limiter = '\t'  
   g_truth = 3
                                       
with open(fn) as f:          
    for line in f:
        h += 1
        
with open(fn) as f:            
    lolo = f.read()
    lines = lolo.replace('\n', limiter)  
    lines = lines.split(limiter)
    ary.append( lines ) 

with open(fn) as f:      
    reader = csv.reader(f, delimiter = limiter, skipinitialspace=True)                     
    first_row = next(reader)                                       
    w = len(first_row)

kk = 0
matrix = [[0 for x in range(w)] for y in range(h)]                               

for i in range(h):
    for j in range(w):
        matrix[i][j] = ary[0][kk]      
        kk += 1

X = np.array( matrix )
if(stt=='segmentation.txt'):
    temp = np.copy( X[:, 0] )
    for i in range(w-1, -1, -1):
        temp2 = np.copy( X[:, i] )
        X[:, i] = temp
        temp = np.copy( temp2 )

opop = X.shape[1]-1          
label_true = np.copy(X[:, opop])
try:
      label_true = label_true.astype(np.int)            
      label_true = label_true - 1
      
except ValueError:
      qq = set(label_true)
      len1 = len(qq)
      for i in range(len1):
         ele1 = qq.pop()
      for j in range( len(X) ):
         if( label_true[j]==ele1 ):
              label_true[j] = str(i)
    
Y =  np.copy(X[:, range(0, opop)]) 
Y = Y.astype(np.float)

clf = GMM(CLUS, n_iter=500, random_state=3).fit(Y)
label_predicted = clf.fit_predict(Y)

# Normalized Mutual Information(MI) 
nmi = metrics.normalized_mutual_info_score(label_true, label_predicted)
       
# Adjusted Mutual Information(AMI)
ami = metrics.adjusted_mutual_info_score(label_true, label_predicted) 
      
# Rand index(RI)
ri = rand_score(Y, label_true, label_predicted)
       
# Adjusted Rand index(ARI)
ari = metrics.adjusted_rand_score(label_true, label_predicted) 

print( "NMI: ", nmi )
print( "AMI: ", ami )
print( "RI: ", ri )
print( "ARI: ", ari )    

print('Means of Chosen Gaussians')
print( clf.means_ )

print('Covariance of Chosen Gaussians')
print( clf.covars_ )

model = TSNE(n_components=2, random_state=0)
np.set_printoptions( suppress=True )
Z = np.array( model.fit_transform(Y) )

if( CLUS<=8 ):
    colors = [ 'r.', 'b.', 'g.', 'k.', 'm.', 'y.', 'c.', 'w.' ]    
    plt.figure(1)                # the second figure
    plt.title('Mixture of Gaussian: Clustering Output Scatter')
    for i in range( len(Z) ):
         plt.plot( Z[i][0], Z[i][1], colors[ label_predicted[i] ], markersize = 10 )
      
    plt.show()