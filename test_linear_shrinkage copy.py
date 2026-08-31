from models.base import *
from models.data import *
from models.linear_shrinkage import *

data = np.random.normal(size=(100, 5))
print(data)

print(data.shape)

Data_object = DataClass(assume_centered=False)
Data_object.fit(data) 

linshrink = LinearShrinkageEstimator()

linshrink.fit(Data_object)
print("#########################")
print(Data_object.eigvals_)
print("#########################")
print(linshrink.eigvals_)