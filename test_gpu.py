import mxnet as mx
from mxnet.runtime import feature_list

print(mx.context.num_gpus())
print(feature_list())