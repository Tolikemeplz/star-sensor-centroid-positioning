import torch
import torch.nn as nn
import torch.nn.functional as F
from model.convnext import ConvNeXt
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

in_22k=True
#model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)

# 访问嵌套的模型状态字典
model_state_dict = checkpoint['model']

# 列出想要删除的层的名称
layers_to_remove = [
    "norm.weight",
    "norm.bias",
    "head.weight",
    "head.bias"
]

# 删除指定的层
for layer_name in layers_to_remove:
    if layer_name in model_state_dict:
        del model_state_dict[layer_name]



# 打印状态字典的键，即所有层的名称
for key in model_state_dict.keys():
    print(key)

# 如果您要使用这个修改后的状态字典来初始化模型
model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],phi=2)
model.load_state_dict(model_state_dict, strict=False)

