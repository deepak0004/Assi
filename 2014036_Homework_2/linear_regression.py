import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from scipy import linalg
from sklearn import cross_validation
matplotlib.style.use('ggplot')

# Add Data
def add_data(percent_wise_X, X, data_till, m):   
    ran = data_till + (int)(m/10) 
    for i in range(data_till, ran):  
        percent_wise_X = np.append( percent_wise_X, X[i], axis = 0 )    
        
    return percent_wise_X, ran 

# inv( X'X + ( lambda.^2 * I ) ) * X' * y    
def theta_ridge_reg(lamda, it, y, data_till, w):
    it = it.reshape( data_till, w ) 
    X = np.matrix(it)                   
    w = X.shape[1]    
    m = X.shape[0]
    trans = it.transpose()      
    trans = trans.reshape( w, m )            
    trans_mul_X = np.dot( trans, X )   
    identity = np.identity( trans_mul_X.shape[0] )  
    regu = (lamda * lamda) * identity
    summ = regu + trans_mul_X   
    inver = linalg.inv( summ )
    res = np.dot( inver, trans )
    y = y[:m]        
    y = np.matrix( y )                       
    y = y.reshape( m, 1 )               
    res = np.dot( res, y )              
    
    return res

def g_descent(X, y, alpha, num_iters, w):
    m = y.size
    costt = np.zeros(shape=(num_iters, 1))
    
    percent_wise_X = X[0] 
    data_till = (int)(m/2)
    theta = np.zeros(shape=(2, 1))    
    
    for i in range( 1, data_till ):
        percent_wise_X = np.append( percent_wise_X, X[i], axis = 0 )
    
    for j in range( 5 ):       
       for i in range(num_iters):   
            pred = X.dot(theta).flatten()
            er1 = (pred - y) * X[:, 0]
            er2 = (pred - y) * X[:, 1]
        
            theta[0][0] = theta[0][0] - alpha * (1.0 / m) * er1.sum()
            theta[1][0] = theta[1][0] - alpha * (1.0 / m) * er2.sum()
            costt[i, 0] = cost_calc(X, y, theta)
            
       if( j<=3 ):         
           [percent_wise_X, data_till] = add_data(percent_wise_X, X, data_till, m)                   
    
    return theta
    
# Cost
def cost_calc(X, y, theta):
    m = y.size
    pred = X.dot(theta).flatten()
    y = y.reshape(m, 1)
    pred = pred.reshape(m, 1)
    error_sq = (pred - y) 
    error_sq = np.squeeze( np.asarray( error_sq ) )
    error_sq = error_sq ** 2
    J = (1.0 / (2 * m)) * error_sq.sum()

    return J

def linear_regression( XX, phi, alpha, max_itr, delta, stt ):     
    m = XX.shape[0]                               # no of rows
    w = XX.shape[1]    
    J_history = np.zeros(shape=( 5, 1 ) )
    #alpha = 0.01    
    
    X = np.ones(shape=(m, w+1))
    for i in range(w):
        X[:, i+1] = XX[:, i]
    col = X.shape[1]-1
    y = X[:, col]
    X = np.delete(X, col, 1)
    w = X.shape[1]
    X2 = X[:,1:]
    
    X_temp = X
    w_temp = w
    
    P = [50, 60, 70, 80, 90]
    if( phi==0 ):     # linear
            #theta  = g_descent(X, y, alpha, max_itr, w)
        
            percent_wise_X = X[0] 
            data_till = (int)(m/2)
            
            for i in range( 1, data_till ):
                percent_wise_X = np.append( percent_wise_X, X[i], axis = 0 )
            for i in range( 5 ): 
               theta = theta_ridge_reg( delta, percent_wise_X, y, data_till, w) 
               J_history[i, 0] = cost_calc(X, y, theta)
               if( i<=3 ):         
                   [percent_wise_X, data_till] = add_data(percent_wise_X, X, data_till, m)
            if( stt=='lin.txt' or stt=='sph.txt' ):
                plt.figure(1)                # the first figure
                plt.title('MSE on Data')
                plt.xlabel( 'Percentage of data' )
                plt.ylabel( 'MSE' )             
                plt.plot(P, J_history, 'xb-', label = 'Linear') 
    elif( phi==1 ):  # Polynomial 
            w = w_temp               
            X = X_temp               
            X_2 = (X2*X2)
            X_3 = (X2*X2)*X2
            X = np.concatenate((X, X_2), axis=1)
            X = np.concatenate((X, X_3), axis=1)
            w_1 = w-1        
            w = w_1 + w_1 + w_1 + 1     
            
            percent_wise_X = X[0] 
            data_till = (int)(m/2)
        
            for i in range( 1, data_till ):
                percent_wise_X = np.append( percent_wise_X, X[i], axis = 0 )       
            for i in range( 5 ): 
               theta = theta_ridge_reg( delta, percent_wise_X, y, data_till, w) 
               J_history[i, 0] = cost_calc(X, y, theta)
               if( i<=3 ):         
                   [percent_wise_X, data_till] = add_data(percent_wise_X, X, data_till, m)           
            if( stt=='lin.txt' or stt=='sph.txt' ):            
                plt.plot(P, J_history, 'xr-', label = 'Polynomial') 
    else:            # Gaussian
            # sph.txt: 0.00001
            w = w_temp               
            X = X_temp
            sigma_inv = 0.00000001
            sigma_inv2 = (X2*X2) * sigma_inv   
            sigma_inv2 = -sigma_inv2
            expp = np.exp( sigma_inv2 )
            #print( expp )
            w_1 = w-1        
            w = w_1 + w_1 + w_1 + w_1
            
            X_1 = expp
            X_2 = expp * X2 * ( ( 2*sigma_inv ) ** 0.5 )     
            X_3 = expp * (X2 * X2) * ( (2) ** 0.5 ) * sigma_inv
            X_4 = expp * (X2 * X2) * X2 * ( (4/3) ** 0.5 ) * ( sigma_inv ** (3/2) )
            
            X = X_1
            X = np.concatenate((X, X_2), axis=1)
            X = np.concatenate((X, X_3), axis=1)
            X = np.concatenate((X, X_4), axis=1)
            
            #print(X)
            percent_wise_X = X[0] 
            data_till = (int)(m/2)
            
            for i in range( 1, data_till ):
                percent_wise_X = np.append( percent_wise_X, X[i], axis = 0 )       
            for i in range( 5 ): 
               theta = theta_ridge_reg( delta, percent_wise_X, y, data_till, w) 
               J_history[i, 0] = cost_calc(X, y, theta)
               if( i<=3 ):         
                   [percent_wise_X, data_till] = add_data(percent_wise_X, X, data_till, m)
            if( stt=='lin.txt' or stt=='sph.txt' ):             
                plt.plot(P, J_history, 'xg-', label = 'Gaussian') 
                plt.legend(loc='upper left')
                plt.show()

                   
    error = np.zeros( shape=(10, 1) )        
    cv = cross_validation.KFold(m, n_folds=10, shuffle=False, random_state=None)        
    pp = 0    
    for traincv, testcv in cv:
        tra = X[ traincv[0], : ]
        test = X[ testcv[0], : ]
        y_tra = y[ traincv[0] ]
        y_test = y[ testcv[0] ]
        
        for j in range( 1, len(traincv) ):
              tra = np.append( tra, X[traincv[j], :] )
              y_tra = np.append( y_tra, y[traincv[j]] )     
        for j in range( 1, len(testcv) ):
              test = np.append( test, X[testcv[j], :] )
              y_test = np.append( y_test, y[testcv[j]] )  
    
        tra = tra.reshape( len( traincv ), w )   
        test = test.reshape( len( testcv ), w ) 
        y_tra = y_tra.reshape( len(traincv), 1 )   
        y_test = y_test.reshape( len(testcv), 1 )    
        theta = theta_ridge_reg( delta, tra, y_tra, len(traincv), w )    
        error[pp] = cost_calc(test, y_test, theta)
        pp += 1
        
    mean_of_test_error = np.mean( error )    
    std_of_test_error = np.std( error )
    
    print( 'Mean of test error: ', mean_of_test_error )
    print( 'Std of test error: ', std_of_test_error )     
    print( 'Cost after training from 50%, 60%, 70%, 80%, 90% of data respectively:' )
    print(J_history)  
    
    
    
    return theta