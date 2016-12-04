import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import scipy.spatial

from sklearn.manifold import TSNE

cost = [ 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999 ]        
P = []
Z = []

def cluster_centroids(data, clusters, k):
    return np.array([data[ clusters==i ].mean( axis = 0 ) for i in range(k)])

def ground_truth_plot(X, K_CLUS):
    opop = X.shape[1]-1  
    Y =  np.copy(X[:, range(0, opop)]) 
    Y = Y.astype(np.float)
    
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
    
    label_true = label_true.astype(np.int)           
    
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions( suppress=True )
    Z = np.array( model.fit_transform(Y) )
    
    colors = [ 'r.', 'b.', 'g.', 'k.', 'm.', 'y.', 'c.', 'w.' ]

    plt.figure(1)                # the first figure
    plt.title('Ground Truth Scatter')
    for i in range( len(Z) ):
        plt.plot( Z[i][0], Z[i][1], colors[ label_true[i] ], markersize = 10 )
      
    plt.show()

def clustering_output_scatter(X, new_centroids):
    opop = X.shape[1]-1  
    Y =  np.copy(X[:, range(0, opop)]) 
    Y = Y.astype(np.float)
        
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
    
    label_true = label_true.astype(np.int)   
    
    sqdists = scipy.spatial.distance.cdist(Y, new_centroids, 'sqeuclidean')  # n*k matrix of squared distance, in each row distance of point from each of the centroid is stored        
    # Index of the closest centroid to each data point
    clusters = np.argmin(sqdists, axis = 1)  
    
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions( suppress=True )
    Z = np.array( model.fit_transform(Y) )

    colors = [ 'r.', 'b.', 'g.', 'k.', 'm.', 'y.', 'c.', 'w.' ]    
    plt.figure(2)                # the second figure
    plt.title('Clustering Output Scatter')
    for i in range( len(Z) ):
        plt.plot( Z[i][0], Z[i][1], colors[ clusters[i] ], markersize = 10 )
      
    plt.show()
    
def ObjectiveFunction_VS_Cost(X, centroids):
    opop = X.shape[1]-1  
    Y =  np.copy(X[:, range(0, opop)]) 
    Y = Y.astype(np.float)
            
    li = []
    clusters = []
    k = len(centroids)
    for i in range(10):
       # Squared distances between each point and each centroid
       sqdists = scipy.spatial.distance.cdist(Y, centroids, 'sqeuclidean')  # n*k matrix of squared distance, in each row distance of point from each of the centroid is stored
        
       # Index of the closest centroid to each data point
       clusters = np.argmin(sqdists, axis = 1)                                 # For example np.argmin(a, axis=0) returns the index of the minimum value in each of the columns      
       
       # Finding new centroids
       new_centroids = cluster_centroids(Y, clusters, k)
                    
       opop = Y - centroids[ clusters ]  
       opop2 = np.square( opop )              # element wise square of matrix         
       opop3 = sum( sum( opop2 ) )            # sum of rows + cols

       li.append( opop3 )
       centroids = new_centroids
       
       P.append( i+1 )
       Z.append( opop3 )
    
    global cost, cluss
    if( li[9]<cost[9] ):
        cost = li
    
    sqdists = scipy.spatial.distance.cdist(Y, new_centroids, 'sqeuclidean')  # n*k matrix of squared distance, in each row distance of point from each of the centroid is stored        
    # Index of the closest centroid to each data point
    clusters = np.argmin(sqdists, axis = 1)  
    
    plt.figure(3)                # the third figure
    plt.title('Objective Function VS Cost')
    plt.xlabel( 'Iteration' )
    plt.ylabel( 'Objective Function' )
    
    plt.plot(P, Z, 'xb-')  
      
    plt.show()    