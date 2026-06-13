# 规则计算后端及默认单例。
from .backend import RulesBackend, default_backend
# 模拟器和 Botzone 共享的声明阶段约束。
from .legality import action_allowed_in_claim, can_claim_discard, can_kong

# rules 子包的稳定公开 API。
__all__ = [
    "RulesBackend", "default_backend", "action_allowed_in_claim",
    "can_claim_discard", "can_kong",
]
