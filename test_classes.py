from models.base import *
from models.sample_correlation import *
from models.simple_clipping import *
from models.data import *

data = np.random.normal(size=(20, 50))

# print(data)

# print(data.shape)

print("Raw data")
print(data)

data_object = DataClass()

data  = data_object._prepare(data)

print("Prepared data")
print(data.shape)
data_object.fit(data)

samp_object = SampleCorrelationEstimator()

samp_object.fit(data_object)

print("Sample Matrix")
print(samp_object.correlation_)

clip_object = NaiveClippingEstimator()
clip_object.fit(data_object)

# print(clip_object.eigvals_)

print(clip_object.correlation_)