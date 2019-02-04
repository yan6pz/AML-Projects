# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:32:49 2019

@author: Yanis
"""
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler

def PCA(dataset, n_components):

    col_mean = np.mean(dataset, axis=0)
    #print(col_mean)
    # center columns by subtracting column means
    
    C = dataset 
    
    cov_mat = (C - col_mean).T.dot((C - col_mean)) / (C.shape[0]-1)
    
    #print('Covariance matrix \n%s' %cov_mat)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    
    #print('Eigenvectors \n%s' %eigenvectors)
    #print('\nEigenvalues \n%s' %eigenvalues)
    
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort()
    eigen_pairs = eigen_pairs[::-1]
    
    components = []
    for i in range(n_components):
        components.append(eigen_pairs[i][1].reshape(4,1))
        
    component_mat = np.hstack((components))
    
    #print('Matrix W:\n', component_mat)

    return component_mat, col_mean


def MSE(dataset, component_mat):
 
    #print(pd.DataFrame(dataset).tail())
    #print(pd.DataFrame(component_mat).tail())
    
    #if dataset.shape[1] > component_mat.shape[1]:
   #     zeros = np.zeros((dataset.shape[1] - component_mat.shape[1] , component_mat.shape[0]))
    #    component_mat = np.concatenate((component_mat, zeros.T), axis=1)
    
    mse = (np.square(dataset - component_mat)).mean(axis=0)
    return mse
    
def reconstruct(u, mean, data):
    
    mean = np.resize(mean,(noisy_data[i].shape[0], noisy_data[i].shape[1]))

    ri = np.dot((data - mean), u) #ri = UT mi = UT(xi − mean ({x})).
    pi = ri
    if 4 > ri.shape[1]:
        zeros = np.zeros((4 - ri.shape[1] , ri.shape[0]))
        pi = np.concatenate((ri, zeros.T), axis=1)
    #xˆi = Upi + mean ({x})

    #unrotate and untranslate adding mean: xˆi = Upi + mean ({x})
    xi = np.dot(pi, u)
    #reshape xi to have the shape of the original data
    if 4 > xi.shape[1]:
        zeros = np.zeros((4 - xi.shape[1] , xi.shape[0]))
        xi = np.concatenate((xi, zeros.T), axis=1)

    xi += mean
    return xi
   
N = np.array(pd.read_csv('iris.csv', na_values=" ?"))
data1 = np.array(pd.read_csv('dataI.csv', na_values=" ?"))
data2 = np.array(pd.read_csv('dataII.csv', na_values=" ?"))
data3 = np.array(pd.read_csv('dataIII.csv', na_values=" ?"))
data4 = np.array(pd.read_csv('dataIV.csv', na_values=" ?"))
data5 = np.array(pd.read_csv('dataV.csv', na_values=" ?"))
noisy_data = [data1, data2, data3, data4, data5] 

#N = StandardScaler().fit_transform(N)

N0 = np.mean(N, axis=0)
N0 = np.resize(N0,(N.shape[0], N.shape[1]))
N1 = PCA(N, 1)
N2 = PCA(N, 2)
N3 = PCA(N, 3)
N4 = PCA(N, 4)
noiseless_data_PCs = [N0, N1, N2, N3, N4]

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
    for j in range(len(noiseless_data_PCs)):
        #do not dot if it is 0 PC
        if j != 0:
            #reconstructed version on noisy dataset i using PC of noiseless data j
            u, mean = noiseless_data_PCs[j]
            xi = reconstruct(u, mean, noisy_data[i])
        else:
            xi = N0
            
        #MSE between the noiseless version N and the pca representation
        mse.append(sum(MSE(N, xi)))
       
    
    #noisy  
    #noisy_data[i] = StandardScaler().fit_transform(noisy_data[i])
    for k in range(5):
        
        if k != 0:
            u, mean = PCA(noisy_data[i], k)
            #reconstructed version on noisy dataset i using k PC computed from
            # mean and covmat of the same set
            xi = reconstruct(u, mean, noisy_data[i])
            if i == 1 and k==2:
                data1_2pc_representation = xi
        else:
            xi = np.mean(noisy_data[i], axis=0)
            xi = np.resize(xi,(noisy_data[i].shape[0], noisy_data[i].shape[1]))
            
        #MSE between the noisy version i and the k-th pca representation of 
        mse.append(sum(MSE(noisy_data[i], xi)))
    
    mse_matrix.append(mse)


#print(mse_matrix)
zeros = np.zeros((2 , 150))
data1_2pc_representation = np.concatenate((data1_2pc_representation, zeros.T), axis=1)

mse_matrix = np.around(mse_matrix, decimals=2)
data1_2pc_representation = np.around(data1_2pc_representation, decimals=2)
                                     
print(mse_matrix)

np.savetxt("shterev2-numbers_unrotated_untranslated1.csv", mse_matrix, delimiter=",", fmt='%5.2f')
np.savetxt("shterev2-recon_unrotated_untranslated1.csv", data1_2pc_representation, delimiter=",", fmt='%5.2f')


 





