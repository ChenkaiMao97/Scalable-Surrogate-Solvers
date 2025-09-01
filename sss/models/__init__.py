import gin

@gin.configurable
def model_factory(model_fn, **kwargs):
    return model_fn(**kwargs)