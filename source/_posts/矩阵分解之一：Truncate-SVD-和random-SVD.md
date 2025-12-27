---
title: "矩阵分解：从 SVD 到现代 AI 应用"
date: 2019-10-04T07:32:18+08:00
tags:
  - machine learning
  - linear algebra
  - LLM
---

矩阵分解是机器学习的基石技术，从传统的推荐系统到现代大语言模型的参数高效微调（LoRA），都离不开矩阵分解的思想。

## 奇异值分解 (SVD)

### 基本形式

任意矩阵 $A \in \mathbb{R}^{m \times n}$ 可以分解为：

$$
A = U \Sigma V^T
$$

其中：
- $U \in \mathbb{R}^{m \times m}$：左奇异向量（正交矩阵）
- $\Sigma \in \mathbb{R}^{m \times n}$：奇异值对角矩阵
- $V \in \mathbb{R}^{n \times n}$：右奇异向量（正交矩阵）

### Truncated SVD

保留前 $r$ 个最大奇异值：

$$
A \approx A_r = U_r \Sigma_r V_r^T
$$

这是**最优**的秩 $r$ 近似（Eckart-Young 定理）：

$$
A_r = \arg\min_{\text{rank}(B) = r} \|A - B\|_F
$$

## Randomized SVD

当矩阵规模巨大时，精确 SVD 计算代价过高。Randomized SVD 提供了高效的近似方法。

### 算法实现

```python
import numpy as np
from scipy import linalg

def randomized_svd(A, n_components, n_oversamples=10, n_iter=4):
    """
    Randomized SVD for large matrices.
    
    Args:
        A: Input matrix (m x n)
        n_components: Number of singular values to compute
        n_oversamples: Additional random vectors for accuracy
        n_iter: Number of power iterations
    
    Returns:
        U, s, Vt: Truncated SVD components
    """
    m, n = A.shape
    n_random = n_components + n_oversamples
    
    # Step 1: Random projection
    Q = np.random.randn(n, n_random)
    
    # Step 2: Power iteration for accuracy
    for _ in range(n_iter):
        Q, _ = linalg.lu(A @ Q, permute_l=True)
        Q, _ = linalg.lu(A.T @ Q, permute_l=True)
    
    Q, _ = linalg.qr(A @ Q, mode='economic')
    
    # Step 3: Project and compute SVD
    B = Q.T @ A
    Uhat, s, Vt = linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    
    return U[:, :n_components], s[:n_components], Vt[:n_components, :]
```

### 复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 精确 SVD | $O(mn \cdot \min(m,n))$ | $O(mn)$ |
| Randomized SVD | $O(mn \cdot r)$ | $O((m+n) \cdot r)$ |
| Truncated SVD (Lanczos) | $O(mn \cdot r)$ | $O((m+n) \cdot r)$ |

## 现代应用：LoRA

LoRA (Low-Rank Adaptation) 是大语言模型参数高效微调的核心技术，直接利用了低秩分解的思想。

### LoRA 原理

预训练权重 $W_0$ 固定，只训练低秩增量：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

### 实现示例

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 原始权重（冻结）
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False
        
        # 低秩分解
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # W(x) + scaling * B(A(x))
        return self.W(x) + self.scaling * self.B(self.A(x))
```

### 参数效率

对于 LLaMA-7B：

| 方法 | 可训练参数 | 显存占用 |
|------|-----------|----------|
| 全量微调 | 7B (100%) | ~140GB |
| LoRA (r=8) | 4.7M (0.07%) | ~14GB |
| LoRA (r=16) | 9.4M (0.13%) | ~16GB |

## 其他应用

### 1. 推荐系统

矩阵分解用于协同过滤：

$$
R \approx U V^T
$$

```python
# 使用 surprise 库
from surprise import SVD, Dataset, Reader

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
model = SVD(n_factors=100)
model.fit(trainset)
```

### 2. 文本表示 (LSA)

潜在语义分析：

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)
```

### 3. 图像压缩

```python
from PIL import Image
import numpy as np

def compress_image(image_path, n_components=50):
    img = np.array(Image.open(image_path).convert('L'))
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    
    # 保留前 n_components 个奇异值
    compressed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    
    return compressed.astype(np.uint8)
```

## 数值稳定性

### 条件数

$$
\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

条件数过大会导致数值不稳定。

### 正则化 SVD

```python
def regularized_svd(A, lambda_reg=0.01):
    """Add regularization for numerical stability."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_reg = s / (s**2 + lambda_reg)
    return U, s_reg, Vt
```

## 延伸阅读

- Halko et al., *Finding Structure with Randomness* (2011)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- [NumPy SVD Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)

---

> 转载请注明出处
