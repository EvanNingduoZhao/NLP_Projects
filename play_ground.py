import numpy as np

data = np.zeros((10,4))

data[1,:]=[1,1,1,1]
np.random.shuffle(data)
print(data)