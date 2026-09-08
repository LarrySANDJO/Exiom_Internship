from models.base import *
from models.data import *
from models.sample_correlation import *
from models.simple_clipping import *
from tools.tools import *

data = np.random.normal(size=(20, 50))

data_object = DataClass()

print("################################################")
print("Prepared data")
print(data.shape)
data_object.fit(data)

samp_object = SampleCorrelationEstimator()

samp_object.fit(data_object)

print("################################################")
print("Sample Matrix")
print(samp_object.correlation_)

clip_object = NaiveClippingEstimator()
clip_object.fit(data_object)

print("################################################")
print("Clipping results")
print(clip_object.correlation_)