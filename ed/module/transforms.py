import numpy as np
    
def transform_expand_dim(axis):
    def fn(arr):
        arr = np.expand_dims(arr, axis=axis)
        return arr
    return fn

def transform_multiply(mul):
    def fn(arr):
        arr = arr * mul
        return arr
    return fn

def transform_divide(div):
    def fn(arr):
        arr = arr / div
        return arr
    return fn

def model_parameters_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
