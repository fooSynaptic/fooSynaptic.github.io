---
title: "矩阵分解之一：Truncate SVD 和 Randomized SVD"
date: 2019-10-04T07:32:18+08:00
tags:
  - machine learning
  - linear algebra
---

最近看了一些矩阵分解的论文，发现了一些有趣的内容，准备总结几篇矩阵分解的文章。

## 奇异值分解基础

每个学过线性代数的人都不会对奇异值分解感到陌生，SVD 广泛应用于统计学、信号处理以及机器学习中。

形式上，一个维度为 m × n 的实数矩阵的奇异值分解可以表示为：

**A = U Σ Vᵀ**

其中：
- **U**：m × m 的正交左奇异向量矩阵
- **Σ**：奇异值对角矩阵
- **V**：n × n 的正交右奇异向量矩阵

## Truncate SVD

在矩阵分解问题上，SVD 提供的策略是计算一个相对于 A 更低秩的近似矩阵 Aᵣ（r < m, n），使得 ‖Aᵣ - A‖ 最小。

**Truncate SVD 策略：**

1. 对奇异值进行降序排序
2. 在对角矩阵 Σ 上取前 r 个奇异值
3. 在左右两边的奇异向量矩阵上取相应的 r 列
4. 得到 Aᵣ = Uᵣ Σᵣ Vᵣᵀ

### 与 PCA 的关系

矩阵的奇异值分解能够直接得到矩阵通过 PCA 的投影空间。

对于协变量矩阵 X（m × p，m 为观测数），计算一个 p × l 的矩阵 W 和 l × l 的对角矩阵 Λ，使得：

**Xᵀ X ≈ W Λ Wᵀ**

通过低秩近似 X ≈ Uᵣ Σᵣ Vᵣᵀ：

**Xᵀ X ≈ Vᵣ Σᵣ² Vᵣᵀ**

### Truncate SVD 的缺点

1. **计算效率**：实际工业环境中，矩阵维度巨大，数据往往缺失、不准确
2. **不支持并行计算**：对于工业级数据来说是致命缺陷

## Randomized SVD

Randomized SVD 相对于 Truncate SVD 的优势：

- 非常稳定
- 性能不依赖于局部特征
- 大量矩阵乘法可利用 GPU 并行计算，比 Truncate SVD 更快

### 算法实现

**第一步：找到正交矩阵 Q**

```python
def randomized_range_finder(A, size, n_iter=5):
    Q = np.random.normal(size=(A.shape[1], size))
    
    for i in range(n_iter):
        Q, _ = linalg.lu(A @ Q, permute_l=True)
        Q, _ = linalg.lu(A.T @ Q, permute_l=True)
        
    Q, _ = linalg.qr(A @ Q, mode='economic')
    return Q
```

> LU 和 QR 分解起规范子的作用，QR 相对 LU 更慢但更准确，所以 QR 规范放在最后一层。

**第二步：利用 Q 得到最终的近似结果**

```python
def randomized_svd(M, n_components, n_oversamples=10, n_iter=4):
    # n_random 就是 truncate SVD 中的 r
    n_random = n_components + n_oversamples
    
    Q = randomized_range_finder(M, n_random, n_iter)
    
    # 把原始观测投影到 (k + p) 维度空间
    B = Q.T @ M
    
    # 对 B 进行奇异值分解
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = Q @ Uhat
    
    return U[:, :n_components], s[:n_components], V[:n_components, :]
```

### 算法原理

Randomized SVD 分成两步：

**第一步：求原始矩阵范围的近似 Q**

通过对随机初始化的小维度矩阵 Q 不断与原矩阵相乘然后分解，得到稳定的向量矩阵。目标是得到 A ≈ Q Qᵀ A。

**第二步：构造并分解小矩阵 B**

构造矩阵 B = Qᵀ A，因为 Q 是低秩的，所以矩阵 B 很小。用传统 SVD 对 B 进行分解：B = S Σ Vᵀ。

最终：A ≈ Q Qᵀ A = Q (S Σ Vᵀ) = U Σ Vᵀ

## 直觉理解

为了估计原始矩阵的范围，我们用一些随机向量，通过原始矩阵 A 和这些随机向量的相乘所产生的变动来近似 A 的波动范围。

假设使用高斯向量矩阵 M 与原矩阵相乘，计算 Y = A M，然后对 Y 进行 QR 分解得到 Q R = Y。矩阵 Q 的每一列就是 Y 范围的正交基，因此可以作为 A 的范围近似。

---

> 转载请注明出处
