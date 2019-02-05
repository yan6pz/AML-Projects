import numpy as np
import csv

def smooth_noise_using_pca(noise_data, model_data, mean, numOfComponent=2):
    #noiseless_covmat = np.dot(noiseless - mean_noiseless, (noiseless - mean_noiseless).T)/noiseless.shape[1]
    covmat = np.dot(model_data - mean, (model_data - mean).T) / (model_data.shape[1])
    #np.cov(noiseless-mean_noiseless)
    eig_value, eig_vector = np.linalg.eig(covmat)
    #re-sort the eigenvalues & eigenvectors in descending order.
    reorder = np.argsort(eig_value)[::-1]
    eig_value = eig_value[reorder]
    eig_vector = eig_vector[:, reorder]
    # rotated using eigenvectors
    r = np.dot(eig_vector.T, (noise_data - mean))
    #r_covmat = np.cov(r)
    # filter out unselected components
    p = np.zeros(r.shape)
    p[:numOfComponent, :] = r[:numOfComponent, :]
    # undo rotation and translation to restore the data
    restored_x = np.dot(eig_vector, p) + mean
    return restored_x


# import the data from the csv.
data = {}
files_names = ["iris.csv", "dataI.csv", "dataII.csv", "dataIII.csv", "dataIV.csv", "dataV.csv"]
for name in files_names:
    with open('./homework3/hw3-data/'+name, newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        data[name] = []
        for row in reader:
            data[name].append(row)
        data[name].pop(0)

noiseless = np.array(data["iris.csv"]).T
mean_noiseless = np.mean(noiseless, axis=1, keepdims=True)

#noisy_set = np.array(data["dataI.csv"]).T
#noisy_set = np.array(data["iris.csv"]).T
#smooth_noise_using_pca(noisy_set, mean_noiseless, numOfComponent=2)

# print out the Mean square errors between noisy and noiseless data set into mse_1.csv
fobj = open('./homework3/minyuan3-numbers.csv', 'a+')
fobj.write('0N,1N,2N,3N,4N,0c,1c,2c,3c,4c\n')

for name in files_names[1:]:
    data_set = np.array(data[name]).T
    mean_noisy = np.mean(data_set, axis=1, keepdims=True)
    n = []
    c = []
    for numberOfComp in range(5):
        # calculate MSE using the mean and covariance matrix of the noiseless set
        restored_set = smooth_noise_using_pca(data_set, noiseless, mean_noiseless, numberOfComp)
        mean_square_error = np.sum((restored_set - noiseless)**2)/noiseless.shape[1]
        n.append(mean_square_error)
        # calculate MSE using the mean and covariance matrix of the noisy set
        restored_set = smooth_noise_using_pca(data_set, data_set, mean_noisy, numberOfComp)
        mean_square_error = np.sum((restored_set - noiseless)**2)/noiseless.shape[1]
        c.append(mean_square_error)
    fobj.write(','.join(np.array(n).astype(str)) + "," + ','.join(np.array(c).astype(str))+'\n')
    #fobj.write((','.join(['%.3f']*len(n))) % tuple(n) + "," + (','.join(['%.3f']*len(n))) % tuple(c) + '\n')

fobj.close()
# print out the reconstructed dataI.csv using 2 principal components.
fobj = open('./homework3/minyuan3-recon.csv', 'a+')
fobj.write('Sepal.Length, Sepal.Width, Petal.Length, Petal.Width\n')
dataI_set = np.array(data['dataI.csv']).T
mean_dataI = np.mean(dataI_set, axis=1, keepdims=True)
reconstructed_dataI = smooth_noise_using_pca(dataI_set, dataI_set, mean_dataI, 2).T
for row in reconstructed_dataI:
    fobj.write(', '.join(row.astype(str))+'\n')
fobj.close()

mean_square_error = np.sum((reconstructed_dataI.T - dataI_set)**2)/reconstructed_dataI.shape[0]
#mean_square_error_against_noiseless = np.sum((reconstructed_dataI.T - noiseless)**2)/noiseless.shape[1]