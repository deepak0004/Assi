import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.style.use('ggplot')

from linear_regression import linear_regression

####################################################################### 
# File to be scanned                                                    
stt = input("Enter file you want: ")  
# Kernel to be applied                                     
phi = eval( input("Enter kernel you want: ") )
# Setting up max_itr, and delta.
max_itr = 1500
delta = 0
if( stt=='seeds_dataset.txt' or stt=='AirQualityUCI.txt' ):                                         
   limiter = '\t'   
else:
   limiter = ','  
#######################################################################  

if( stt=='sph.txt' ):
    alpha = 0.000001
else:
    alpha = 0.01 
                                                                  
fn = os.path.join(os.path.dirname(__file__), stt)                       
data = np.genfromtxt(fn, delimiter = limiter, dtype = 'str')   
m = 0

with open(fn) as f:      
    reader = csv.reader(f, delimiter = limiter, skipinitialspace=True)                     
    first_row = next(reader)                                       
    w = len(first_row)
  
with open(fn) as f:                                                       
    for line in f:  
        m += 1
        
if( stt=='iris_data.txt' ):                         
        opop = data.shape[1]-1                     
        label_true = np.copy(data[:, opop])          
        qq = set(label_true)                       
        len1 = len(qq)                                       
        for i in range(len1):                      
            ele1 = qq.pop()                         
            for j in range( len(data) ): 
                 if( label_true[j]==ele1 ):
                      label_true[j] = str(i) 
        label_true = label_true.astype(np.int)
        data[:, opop] = label_true             

X = np.ones(shape=(m, w))
for i in range(w):
    X[:, i] = data[:, i]

X = X.astype(np.float)

final_parameters = linear_regression( X, phi, alpha, max_itr, delta, stt )
print( 'Final_parameters:' )
print( final_parameters )             # the first figure
if( (stt=='lin.txt' or stt=='sph.txt') and phi==0 ):
      plt.figure(2)   
      XY = np.ones(shape=(m, w+1))
      plt.plot(X[:, 0], X[:, 1], 'ro')       
      for i in range(w):
        XY[:, i+1] = X[:, i]
        
      col = XY.shape[1]-1
      yy = XY[:, col]
      XY = np.delete(XY, col, 1)
      
      pred = np.dot(XY, final_parameters)
      plt.plot(X[:, 0], pred, 'xb-', label = 'Plot')     
      plt.show()
      
  