# V1 扁平特征编码与 JSON 序列化工具。
from .encoder import (ACTION_SIZE, FEATURE_SIZE, compact_observation,
                      deserialize_action, encode_action, encode_observation,
                      expand_observation, serialize_action)
# V2 token 特征编码与固定 shape 常量。
from .token_encoder import (FEATURE_VERSION, MAX_ACTION_TOKENS, MAX_STATE_TOKENS,
                            TOKEN_SIZE, encode_action_v2, encode_observation_v2)

# features 子包的稳定公开 API。
__all__ = [
    "ACTION_SIZE", "FEATURE_SIZE", "compact_observation", "deserialize_action",
    "encode_action", "encode_observation", "expand_observation", "serialize_action",
    "FEATURE_VERSION", "MAX_ACTION_TOKENS", "MAX_STATE_TOKENS", "TOKEN_SIZE",
    "encode_action_v2", "encode_observation_v2",
]
