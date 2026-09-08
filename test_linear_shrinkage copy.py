from models.base import *
from models.data import *
from models.linear_shrinkage import *
from tools.tools import *

from sklearn.covariance import LedoitWolf

data = np.random.normal(size=(20, 50))

data_normalisee = (data - data.mean(axis=0)) / data.std(axis=0)

data_object = DataClass()

print("################################################")
print("Prepared data")
print(data.shape)
data_object.fit(data)

my_ls_object = LinearShrinkageEstimator()

my_ls_object.fit(data_object)

LD_ls_object = LedoitWolf(assume_centered=True)

LD_ls_object.fit(data_normalisee)

print("################################################")
print("ls_Matrix")
print(my_ls_object.correlation_)

print("################################################")
print("LD_ls_Matrix")
print(LD_ls_object.covariance_)

print("################################################")
print("Difference")
print(LD_ls_object.covariance_ - my_ls_object.correlation_)