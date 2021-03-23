import sys
import numpy as np

sys.path.append('../../2_Iris/scripts/')

import iris_load_visualize as ilv

attributes, labels = ilv.loadDataset(ilv.dsPath)
print(attributes.shape)
mu = attributes.mean(1)
mu = mu.reshape(mu.size, 1)
# Centering dataset
DC = attributes - mu

# Covariance matrix
C = (np.dot(DC, DC.T))/DC.shape[1]

print(mu)
print(C)