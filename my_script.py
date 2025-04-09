import torch
import numpy as np
import torch.nn.functional as F

# 随机模拟特征
torch.manual_seed(0)
batch_size = 4
embedding_dim = 3

# 模拟归一化后的图像和文本特征
image_features = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
text_features = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
print(f"image_features_size: {image_features.size()}")
print(f"text_features_size: {text_features.size()}")

# 温度缩放
logit_scale = torch.ones([]) * np.log(1 / 0.07)
print(f"logit_scale: {logit_scale}")
print(f"logit_scale.size(): {logit_scale.size()}")

# 计算相似度（logits）：图像对文本的相似度矩阵
logits = logit_scale * image_features @ text_features.T  # shape: (4, 4)

# 显式 InfoNCE Loss 实现
# softmax + log + 负号 + 求平均
log_probs = F.log_softmax(logits, dim=1)  # dim=1 表示每行是一个图像的预测分布
print(f"log_probs.shape: {log_probs.shape}")

labels = torch.arange(batch_size)# 正确匹配的文本索引
print(f"labels: {labels}")

loss_manual = -log_probs[torch.arange(batch_size), labels].mean()
print(f"loss_manual: {loss_manual}")

# PyTorch 的 cross_entropy 实现
loss_builtin = F.cross_entropy(logits, labels)

print(f"显式 InfoNCE Loss:   {loss_manual.item():.6f}")
print(f"CrossEntropy Loss:   {loss_builtin.item():.6f}")