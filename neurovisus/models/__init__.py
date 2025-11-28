# neurovisus/models/__init__.py
from .painformer import PainFormer
from .legacy import IPFusionNetResNet  # 假设你放这里了

def get_model(arch_name, **kwargs):
    arch_map = {
        'painformer': PainFormer,
        'resnet': IPFusionNetResNet,
    }
    if arch_name not in arch_map:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return arch_map[arch_name](**kwargs)