from models.base import *
from models.sample_correlation import *

data = np.random.normal(size=(100, 50))

print(data)

print(data.shape)

rot_int_object = SampleCorrelationEstimator()

rot_int_object.fit(data)

print(rot_int_object.eigvals_)