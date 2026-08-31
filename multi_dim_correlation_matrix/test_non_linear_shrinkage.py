from models.base import *
from models.non_linear_shrinkage import *

data = np.random.normal(size=(1000, 5))
print(data)

print(data.shape)

Data_object = DataClass(assume_centered = False)
Data_object.fit(data) 

non_linshrink = NonLinearShrinkageEstimator()

non_linshrink.fit(Data_object)
print("#########################")
# print(Data_object.correlation_)
print("#########################")
print(non_linshrink.correlation_)