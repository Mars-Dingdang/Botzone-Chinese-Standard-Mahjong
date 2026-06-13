# PyTorch 是可选依赖；缺失时包仍可用于规则、环境和非模型策略。
try:
    from .hybrid_transformer import HybridTransformer
    from .token_transformer import TokenTransformer
    from .factory import create_model
except ImportError:
    # 以 None 明确表示模型功能不可用，避免导入整个 mahjong_agent 失败。
    HybridTransformer = None
    TokenTransformer = None
    create_model = None

# models 子包的稳定公开 API。
__all__ = ["HybridTransformer", "TokenTransformer", "create_model"]
