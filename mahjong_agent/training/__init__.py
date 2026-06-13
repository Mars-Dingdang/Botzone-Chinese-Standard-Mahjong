# 训练子包默认只公开最基础的单局 rollout 接口。
from .rollout import play_episode

# training 子包的稳定公开 API。
__all__ = ["play_episode"]
