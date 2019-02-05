import numpy as np
import csv
from sklearn.decomposition import PCA

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

noiseless = np.array(data["iris.csv"])

# save the Mean square errors between noisy and noiseless data set into a csv
fobj = open('./homework3/minyuan3-numbers.csv', 'a+')
fobj.write('0N,1N,2N,3N,4N,0c,1c,2c,3c,4c\n')

for name in files_names[1:]:
    data_set = np.array(data[name])
    n = []
    c = []
    for numberOfComp in range(5):
        # calculate MSE based on the noiseless set
        pca = PCA(n_components=numberOfComp)
        pca.fit(noiseless)
        restored_set = pca.inverse_transform(pca.transform(data_set))
        mean_square_error = np.sum((restored_set - noiseless)**2)/noiseless.shape[0]
        n.append(mean_square_error)
        # calculate MSE based on the noisy set
        pca = PCA(n_components=numberOfComp)
        pca.fit(data_set)
        restored_set = pca.inverse_transform(pca.transform(data_set))
        mean_square_error = np.sum((restored_set - noiseless)**2)/noiseless.shape[0]
        c.append(mean_square_error)
    fobj.write(','.join(np.array(n).astype(str))+ "," + ','.join(np.array(c).astype(str))+'\n')
    #fobj.write((','.join(['%.3f']*len(n))) % tuple(n) + "," + (','.join(['%.3f']*len(n))) % tuple(c) + '\n')

fobj.close()
# print out the reconstructed dataI.csv using 2 principal components.
fobj = open('./homework3/minyuan3-recon.csv', 'a+')
fobj.write('Sepal.Length, Sepal.Width, Petal.Length, Petal.Width\n')
dataI_set = np.array(data['dataI.csv'])
pca = PCA(n_components=2)
pca.fit(dataI_set)
reconstructed_dataI = pca.inverse_transform(pca.transform(dataI_set))
for row in reconstructed_dataI:
    fobj.write(', '.join(row.astype(str))+'\n')
fobj.close()

