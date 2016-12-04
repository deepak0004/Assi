import numpy as np
import csv
import os

from my_kmeans import kmeans
from my_kmeans_script import ground_truth_plot
from my_kmeans_script import clustering_output_scatter
from my_kmeans_script import ObjectiveFunction_VS_Cost

#######################################################################
# File to be scanned 
stt = input("Enter file Name: ")
# No of clusters
K_CLUS = eval( input("Enter no of clusters you want: ") )                                         
#######################################################################

nmii = 0
amii = 0
rii = 0
arii = 0

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


centr = [[0 for x in range(w-1)] for y in range(K_CLUS)]   
for i in range(10):
   opop = X.shape[1]-1    
   centr = np.copy( X[np.random.choice(X.shape[0], K_CLUS)] )
   centr = centr[:, range(0, opop)]
   centr = centr.astype(np.float)
   new_centroid, eval_mat = kmeans(X, centr, 10)                
   
   nmii += eval_mat[0]
   amii += eval_mat[1]
   rii += eval_mat[2]
   arii += eval_mat[3]
      
nmii /= 10
amii /= 10
rii /= 10
arii /= 10

print( "NMI: ", nmii )
print( "AMI: ", amii )
print( "RI: ", rii )
print( "ARI: ", arii )

if( K_CLUS==g_truth ):
    ground_truth_plot(X, g_truth)                                                               
    clustering_output_scatter(X, new_centroid)                
    ObjectiveFunction_VS_Cost(X, centr)                