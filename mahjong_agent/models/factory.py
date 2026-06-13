"""Version-aware model creation."""


def create_model(feature_version=2, **kwargs):
    # kwargs 原样传给对应模型构造器，通常来自 checkpoint 的 model_config。
    if int(feature_version) == 1:
        from .hybrid_transformer import HybridTransformer
        return HybridTransformer(**kwargs)
    if int(feature_version) == 2:
        from .token_transformer import TokenTransformer
        return TokenTransformer(**kwargs)
    raise ValueError("unsupported feature_version=%r" % feature_version)
