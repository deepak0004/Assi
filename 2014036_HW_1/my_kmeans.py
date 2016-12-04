import numpy as np
import scipy.spatial
import itertools

from sklearn import metrics 

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
         
def cluster_centroids(data, clusters, k):
    return np.array([data[ clusters==i ].mean( axis = 0 ) for i in range(k)]) # Compute the mean of data[ i ] where i is set column wise
   
def kmeans(X, initial_centroids, max_iters):
     eval_mat = []     
     clusters = []
     
     k = len(initial_centroids)
  
     opop = X.shape[1]-1   
     label_true = np.copy(X[:, opop])
     Y =  np.copy(X[:, range(0, opop)])   
     
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
     Y = Y.astype(np.float)
     
     centroids = initial_centroids  
     for i in range(max_iters):
         # Squared distances between each point and each centroid
        sqdists = scipy.spatial.distance.cdist(Y, centroids, 'sqeuclidean')  # n*k matrix of squared distance, in each row distance of point from each of the centroid is stored
        
        # Index of the closest centroid to each data point
        clusters = np.argmin(sqdists, axis = 1)                                 # For example np.argmin(a, axis=0) returns the index of the minimum value in each of the columns      
       
        # Finding new centroids
        new_centroids = cluster_centroids(Y, clusters, k)

        # break early if new_centroids = centroids
        if np.array_equal(new_centroids, centroids):
            break
         
        centroids = new_centroids
         
     # Normalized Mutual Information(MI) 
     nmi = metrics.normalized_mutual_info_score(label_true, clusters)
       
     # Adjusted Mutual Information(AMI)
     ami = metrics.adjusted_mutual_info_score(label_true, clusters) 
        
     # Rand index(RI)
     ri = rand_score(Y, label_true, clusters)
        
     # Adjusted Rand index(ARI)
     ari = metrics.adjusted_rand_score(label_true, clusters)    
     
     eval_mat.append( nmi )    
     eval_mat.append( ami )
     eval_mat.append( ri )
     eval_mat.append( ari )   
     
     return new_centroids, eval_mat   