# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler

def PCA(dataset, col_mean):

    cov_mat = np.dot((dataset - col_mean),(dataset - col_mean).T) / (dataset.shape[1])
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort()
    eigen_pairs = eigen_pairs[::-1]
    
    components = []
    for i in range(4):
        components.append(eigen_pairs[i][1].reshape(4,1))
        
    eigen_vec = np.hstack((components))
    
    return eigen_vec


def MSE(dataset, component_mat):
 
    mse = np.sum(np.square(component_mat-dataset))/dataset.shape[1]
    return mse
    
def reconstruct(u, mean, data, n_components):

    ri = np.dot(u.T, (data - mean)) #ri = UT mi = UT(xi − mean ({x})).
    
    #generate pi
    pi = np.zeros(ri.shape)
    pi[:n_components, :] = ri[:n_components, :]
    #xi = Upi + mean ({x})
    #unrotate and untranslate adding mean: xˆi = Upi + mean ({x})
    xi = np.dot(u, pi)
    #reshape xi to have the shape of the original data

    xi += mean
    return xi
   
N = np.array(pd.read_csv('iris.csv', na_values=" ?")).T
data1 = np.array(pd.read_csv('dataI.csv', na_values=" ?")).T
data2 = np.array(pd.read_csv('dataII.csv', na_values=" ?")).T
data3 = np.array(pd.read_csv('dataIII.csv', na_values=" ?")).T
data4 = np.array(pd.read_csv('dataIV.csv', na_values=" ?")).T
data5 = np.array(pd.read_csv('dataV.csv', na_values=" ?")).T
noisy_data = [data1, data2, data3, data4, data5] 


N0 = np.mean(N, axis=1, keepdims=True)
u_noiseless = PCA(N, N0 )

data1_2pc_representation = []
mse_matrix = []

# For each of the noisy datasets 
# Estimate the MSE between
# A) noiseless data and the PC representation of the noisy dataset
# computed from mean and cov_mat of the noiseless data
# B) noisy data and the PC representation of the same data
# computed from mean and cov_mat of the data itself
for i in range(len(noisy_data)):
    mse = []
    #noiseless
    for j in range(5):
        #do not dot if it is 0 PC
        if j != 0:
            #reconstructed version on noisy dataset i using PC of noiseless data j
            xi = reconstruct(u_noiseless, N0, noisy_data[i], j)
        else:
            xi = N0
            
        #MSE between the noiseless version N and the pca representation
        mse.append(MSE(N, xi))
       
    mean_noisy = np.mean(noisy_data[i], axis=1, keepdims=True)
    u_noisy = PCA(noisy_data[i], mean_noisy)

    for k in range(5):
        
        if k != 0:
            #reconstructed version on noisy dataset i using k PC computed from
            # mean and covmat of the same set
            xi = reconstruct(u_noisy,mean_noisy, noisy_data[i], k)
            if i == 0 and k==2:
                data1_2pc_representation = xi.T
        else:
            xi = mean_noisy
            
        #MSE between the noisy version i and the k-th pca representation of 
        mse.append(MSE(N, xi))
    
    mse_matrix.append(mse)


mse_matrix = np.around(mse_matrix, decimals=2)
data1_2pc_representation = np.around(data1_2pc_representation, decimals=2)
                                     
print(mse_matrix)

np.savetxt("shterev2-numbers.csv", mse_matrix, delimiter=",", fmt='%5.4f')
np.savetxt("shterev2-recon.csv", data1_2pc_representation, delimiter=",", fmt='%5.4f')


 





