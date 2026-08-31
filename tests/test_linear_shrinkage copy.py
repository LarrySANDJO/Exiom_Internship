from models.base import *
from models.linear_shrinkage import *

data = np.random.normal(size=(100, 50))

print(data)

print(data.shape)

rot_int_object = LinearShrinkageEstimator()

rot_int_object.fit(data)

print(rot_int_object.eigvals_)