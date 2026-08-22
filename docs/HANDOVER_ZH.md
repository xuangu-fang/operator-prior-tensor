# 交接技术文稿：Geometry-aware Tensor Factorization through Operator-defined Functional Subspaces

面向接手本项目的同学。本文档假设你没有读过之前的任何记录，目标是让你在读完之后能够
**独立复现全部结果、理解每一个设计决定的来历、并且知道哪些坑已经踩过**。

- 研究日志（按时间顺序的全部迭代，含失败）：[`ITERATIONS.md`](ITERATIONS.md)
- 数据集来源与从零重建方法：[`DATASETS.md`](DATASETS.md)
- 待办与明确不做的事：[`TODO.md`](TODO.md)

---

## 第一部分：这个方法要解决什么

### 1.1 问题

给定一个三阶张量 $\mathcal Y \in \mathbb R^{N_1 \times N_2 \times N_3}$，语义是
`(场景, 时间, 空间节点)`：同一条偏微分方程从 $N_1$ 个不同初始条件出发，在 $N_2$ 个
时刻，采样于一张有 $N_3$ 个节点的网格。我们只观测到其中一个稀疏子集 $\Omega$
（本项目取 2%–20%），要重构全部条目。

普通 Tucker 补全写成

$$
\mathcal Y_{i_1 i_2 i_3} \approx \sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}
\mathcal G_{r_1 r_2 r_3}\, U_1(i_1, r_1)\, U_2(i_2, r_2)\, U_3(i_3, r_3),
$$

其中因子矩阵 $U_m \in \mathbb R^{N_m \times R_m}$ 自由学习。

**这里有一个被浪费掉的信息**：第三维的 index 不是无意义的类别编号，它是网格上一个**有
坐标、有邻接、有边界**的点。普通 Tucker 把墙两侧相距 5 厘米的两个节点，和同一个房间里
相距 5 米的两个节点，同等对待。

### 1.2 更尖锐的形式：传感器采样下问题是未定义的

本项目最重要的一个观察来自采样方式。两种协议：

| 协议 | 含义 | 对经典分解的后果 |
|---|---|---|
| `random` | 均匀随机抽取张量条目 | 每个节点出现在很多观测条目里，$U_3$ 的每一行都可估 |
| `spatial_sensors` | 选少数节点，观测它们的**完整轨迹** | 未观测节点出现在**零个**观测条目里 |

在传感器协议下，一个未被选中的节点 $i_3$，它的因子行 $U_3(i_3, \cdot)$ 出现在
**零个**方程中。它不是难估，是**完全无约束**。我们实测 CP-ALS 与 Tucker-HOOI
（TensorLy 实现，SVD 初始化，秩用 held-out 开天眼选，跑满迭代）在所有布局上
**精确返回 NRMSE = 1.000**，即在未观测处输出零。

这不是把基线做残废——同样两个例程在 `random` 协议下是 0.27–0.66 的正常对手。
两种采样问的是两个不同的问题。

> 这一点写进了单元测试而不是散文：
> `tests/test_benchmark.py::test_classical_completion_is_undefined_under_sensor_sampling`

### 1.3 我们的做法

把空间因子限制在一个**由几何决定的函数空间**里：

$$
U_3 = \Phi\, W, \qquad \Phi \in \mathbb R^{N_3 \times K},\ W \in \mathbb R^{K \times R_3},
$$

其中 $\Phi$ 的列是**该几何上的算子特征函数**的前 $K$ 个。于是要学的参数从
$N_3 R_3$（例如 $5520 \times 16 = 88\,320$）降到 $K R_3$（例如 $16 \times 16 = 256$），
而且——这是关键——**每个节点的因子行都通过 $\Phi$ 与其余所有节点绑定**，未观测节点
不再无约束。

---

## 第二部分：公式与推导

### 2.1 算子与它的谱基

设域为 $\Omega \subset \mathbb R^d$（可以是平面区域、体积、或闭曲面）。取扩散型算子

$$
\mathcal A u = -\nabla\cdot\big(a(x)\,\nabla u\big) + \kappa u ,
$$

$a(x)$ 是扩散系数，$\kappa$ 是反应率。在 $\Omega$ 上用 P1 有限元离散，得到刚度矩阵
$K$ 与质量矩阵 $M$（均为 $N_3 \times N_3$ 稀疏对称阵）：

$$
K_{ij} = \int_\Omega a(x)\, \nabla\varphi_i \cdot \nabla\varphi_j \, dx,
\qquad
M_{ij} = \int_\Omega \varphi_i \varphi_j \, dx .
$$

谱基 $\Phi$ 取广义特征问题的**最低 $K$ 个**解：

$$
K \phi_k = \lambda_k\, M \phi_k, \qquad
\lambda_1 \le \lambda_2 \le \cdots \le \lambda_K,
\qquad
\Phi = [\phi_1, \dots, \phi_K].
$$

特征向量取**质量正交归一**，即 $\Phi^\top M \Phi = I$。这一点不是形式上的讲究：
在非均匀网格上，普通欧氏正交会让**节点密度**而不是**函数频率**决定什么算"低频"。

实现：`src/geoaware/operator_diagnostics.py::sparse_eigenpairs`，用
`scipy.sparse.linalg.eigsh` 的移位反演（shift-invert）Lanczos，取 $\sigma = -10^{-6}$
（纯 Neumann 算子在 0 处奇异，常数场能量为零，所以不能正好在 0 处分解）。

**为什么这能扩展。** 我们只需要最低的十几到几十个特征对，而移位反演 Lanczos 的代价
与请求的模态数成正比、而不是与 $N_3^3$ 成正比。稠密路径在 642 个节点上要 32 秒，
稀疏路径 0.11 秒；40 962 个节点的球面稠密不可行，稀疏 35.6 秒。两条路径在都能跑的
网格上给出**逐位相同**的张量、特征值差 $5.7\times10^{-13}$
（`tests/test_manifold_barrier.py::test_sparse_and_dense_paths_agree_exactly`）。

### 2.2 关键的信息层级：几何已知，材料未知

这是审稿人最容易攻击的地方，必须讲清楚。

- **真值**用的算子是 $\mathcal A_{\text{truth}}$，其中 $a(x)$ 含有一个**平滑的背景变化**
  （对数扩散率对比度 0.3）**以及**障碍物；
- **学习器**拿到的算子是 $\mathcal A_{\text{learner}}$，其中 $a(x) = 1$ 在背景处，
  **只有障碍物的位置是已知的**。

也就是说：**几何（墙、孔洞、边界）是已知元数据；材料是未知的。** 楼层平面图任何建筑
都有；空气的有效扩散率没人测。所以这不是人为构造的信息壁垒。

代码里这一条有单元测试守着：改变真值材料，学习器的所有基**逐位不变**
（`test_learner_bases_never_read_the_truth_material`）。

### 2.3 模型：group-wise Tucker

坐标被划分成若干**组** $g$，每组配一个因子。三阶张量的默认划分是
$\{\{1\},\{2\},\{3\}\}$，但公式对任意划分成立（例如把 $(x,y)$ 合成一个空间组）。

每组的因子有三种：

$$
F_g =
\begin{cases}
\Phi_g W_g & \text{operator：该组有可信算子} \\[2pt]
T_g & \text{table：自由表（如"场景"这种枚举，没有算子可言）} \\[2pt]
\mathrm{MLP}_\theta(x_g) & \text{neural：有坐标但无可信算子}
\end{cases}
$$

**mode-wise 的旧实现是每组恰好含一个轴的特例。** 这一点很重要：它意味着方法不要求
"张量每一维都有一条独立 PDE"，这是一个常见的误解。

预测为

$$
\hat y(i_1,i_2,i_3) = \sum_{r_1,r_2,r_3} \mathcal G_{r_1r_2r_3}
\prod_{m=1}^{3} \tilde F_m(i_m, r_m),
$$

其中 $\tilde F$ 是**逐列单位 RMS 归一化**后的因子。归一化让 Sobolev 罚项与尺度无关，
也让不同组之间可比；代价是因子的尺度不可辨识，但它被 core 吸收，而 core 正是唯一
有后验的对象。

**CP 是同一个模型的对角核特例。** 令 $R_1 = R_2 = R_3 = R$ 且
$\mathcal G_{r_1r_2r_3} = w_r \delta_{r_1 r_2 r_3}$，则

$$
\hat y(i_1,i_2,i_3) = \sum_{r=1}^{R} w_r \prod_m \tilde F_m(i_m, r).
$$

代码里是 `GroupedOperatorTucker(..., core="diagonal")`。这样做的好处是：**CP 基线
与我们共享优化器、归一化、先验和闭式后验**，两者的差异只能来自模型本身，不可能
来自"训练方式不同"。

### 2.4 先验

对 operator 组用 **Sobolev 谱罚**：

$$
\mathcal R_g(W_g) \;=\; \frac{1}{KR}\sum_{k=1}^{K}\sum_{r=1}^{R}
(1+\lambda_k)^{p}\, \widetilde W_{kr}^{2},
\qquad p = 1.5 ,
$$

其中 $\widetilde W$ 是归一化后的谱系数。含义是：**高频模态要付更高的代价**，
$(1+\lambda_k)^p$ 正是 Sobolev 范数在特征基下的权重。

对 table 组（如果给了罚算子 $P$）用二次型 $\operatorname{tr}(F^\top P F)$，其中 $P$
满足 $\phi_k^\top P \phi_k = (1+\lambda_k)^p$。这构造出的对照很关键：**自由表付
与谱因子完全相同的频率代价，唯一的差别是它不被限制在前 $K$ 个模态里**——这正是
"是几何起作用还是截断起作用"这个消融所需要的。

对 neural 组用权重的 $\ell_2$。

总目标：

$$
\mathcal L \;=\; \frac{1}{|\Omega|}\sum_{(i_1,i_2,i_3)\in\Omega}
\big(\hat y - y\big)^2 \;+\; \beta_{\mathrm{reg}}
\Big[\overline{\mathcal G^2} + \sum_g \mathcal R_g\Big],
\qquad \beta_{\mathrm{reg}} = 2\times10^{-3}.
$$

### 2.5 推断：两段式

**第一段（因子）**：AdamW，$\mathrm{lr}=3\times10^{-3}$，`weight_decay` $10^{-6}$，
梯度裁剪 5.0，1500 步，冷随机初始化，**无早停**，但保留目标值最优的 checkpoint
而非最后一步的迭代。

**第二段（core）**：固定因子，对 core 做**闭式高斯后验**。这一段没有梯度。

推导如下。给定因子后，模型对 core 是**线性**的。记

$$
z(i)^\top = \Big(\tilde F_1(i_1,r_1)\,\tilde F_2(i_2,r_2)\,\tilde F_3(i_3,r_3)\Big)_{r_1r_2r_3}
\in \mathbb R^{p},\qquad p = R_1R_2R_3,
$$

即三个因子行的**逐行 Kronecker 积**（CP 情形下是 Khatri–Rao 积，$p=R$）。于是
$\hat y(i) = z(i)^\top \mathrm{vec}(\mathcal G)$。取

$$
p(\mathcal G) = \mathcal N(0, \alpha^{-1} I), \qquad
p(y \mid \mathcal G) = \mathcal N(Z\,\mathrm{vec}(\mathcal G),\ \beta^{-1} I),
$$

标准结果给出高斯后验

$$
\Sigma = \big(\beta Z^\top Z + \alpha I\big)^{-1}, \qquad
\mu = \beta\, \Sigma\, Z^\top y .
$$

$\alpha, \beta$ 用**证据不动点**（type-II 最大似然 / MacKay）估计。定义**有效参数个数**

$$
\gamma = p - \alpha\,\operatorname{tr}(\Sigma),
$$

迭代

$$
\alpha \leftarrow \frac{\gamma}{\|\mu\|^2}, \qquad
\beta \leftarrow \frac{|\Omega| - \gamma}{\|y - Z\mu\|^2}.
$$

**实现上的一个要点**：$Z^\top Z$ 在整个不动点循环中**不变**，所以只做一次对角化
$Z^\top Z = V\Lambda V^\top$，此后每次迭代都在特征基下进行——协方差是对角的、迹是
求和、均值是逐元素缩放——把每次迭代从 $O(p^3)$ 降到 $O(p)$。core 为 1920 时这是
"整个拟合被这个循环主导"与"不被主导"的差别。

预测的方差与**留一校准**：

$$
\operatorname{Var}[\hat y(i)] = z(i)^\top \Sigma z(i) + \beta^{-1},
$$

再乘一个校准系数 $c$：用杠杆值 $h_i = \beta\, z_i^\top \Sigma z_i$ 修正的留一残差
$(y_i - \hat y_i)/(1-h_i)$，取 $|\text{LOO 残差}| / \text{LOO 标准差}$ 的 95 分位除以
1.96，clamp 在 $[0.5, 4]$。所以误差棒是**经过校准的**，不是裸后验方差。

**明确没有做的**：因子上没有后验。这是设计取舍，代价见 §4.3 的 ARD 负结果。

### 2.6 前向计算：不要物化 design 矩阵

朴素实现把 $Z$ 显式建出来再做矩阵乘。当 core 有 1920 个元素、观测有 65k 行时，
$Z$ 是 $65000 \times 1920 \approx 1.25\times10^8$ 个数（500 MB），而且每个梯度步都要重建。

正确做法是**按模态依次收缩**：

$$
t^{(1)}_{n,r_2,r_3} = \sum_{r_1} \tilde F_1(i_1^{(n)}, r_1)\, \mathcal G_{r_1r_2r_3},
\qquad
t^{(2)}_{n,r_3} = \sum_{r_2} \tilde F_2(i_2^{(n)}, r_2)\, t^{(1)}_{n,r_2,r_3},
$$
$$
\hat y_n = \sum_{r_3} \tilde F_3(i_3^{(n)}, r_3)\, t^{(2)}_{n,r_3}.
$$

结果**完全相同**（float32 下差 $2.4\times10^{-6}$，对角核精确为 0，
`test_the_contraction_matches_the_design_matrix_it_replaces`），但算术量约 2.4 倍少、
峰值内存降一个量级。1920 元素的 core 因此与 96 元素的一样快。

---

## 第三部分：诊断量——本项目最该被继承的部分

这些量**在拟合之前**就能算，只用真值张量、候选基和观测掩码。它们是本方法区别于
"试试看"的地方，也是我们几次自我纠错的工具。

### 3.1 投影残差（基能否张成这个场？）

$$
\varepsilon(\Phi) \;=\;
\frac{\big\|\,\mathcal Y - \mathcal Y \times_3 (QQ^\top)\,\big\|_F}{\|\mathcal Y\|_F},
\qquad Q = \operatorname{qr}(\Phi).
$$

$\times_3$ 是模态 3 的乘积。含义是"用这个基去表示这个场，最好也只能错这么多"，
即**偏差下限**。

实现细节：必须写成 $Q(Q^\top \mathcal Y)$ 而不是 $(QQ^\top)\mathcal Y$，否则要物化
$N_3 \times N_3$ 的投影算子，在上万节点时装不下。

`operator_diagnostics.py::product_projection_residual`

### 3.2 可观测性（这个模态在传感器上被激发吗？）

$$
v_k \;=\; \frac{\sum_{i \in \mathcal S} \phi_k(i)^2}{\sum_{i} \phi_k(i)^2}
\;\Big/\;
\frac{|\mathcal S|}{N_3},
$$

$\mathcal S$ 是被观测的节点集合。$v_k = 1$ 表示"这个模态在传感器上的能量占比与平均
节点相当"；$v_k \approx 0$ 表示**传感器看不见它**。

**为什么需要它**：局域在薄障碍内部的模态，特征值是全谱**最低**的（障碍是近零传导率），
但传感器几乎不可能落在薄墙里。这样的模态系数**完全无约束**，模型会用它外推出荒谬的值。
我们实测到 NRMSE 1.246、预测范围 $[-74.8, 77.5]$ 而真值在 $[-5.8, 10.4]$，可见度
$v_5..v_8 = 0.004, 0.001, 0.001, 0.002$。

筛选规则：保留 $v_k \ge 0.1$ 的列（第一列常数模态始终保留）。**只用基和掩码，不碰
数据**，且对所有谱模型一视同仁——几何盲基通常一个模态都不丢，这本身就是结果。

`operator_diagnostics.py::observable_modes`

### 3.3 Headroom（秩能否兑现这个基？）

$$
\text{headroom} \;=\; \frac{\text{实际达到的 NRMSE}}{\text{该模型自身的投影残差}} .
$$

**这个检查我们缺了很久，代价很大。** 见 §4.1。

远大于 1 意味着拟合是**秩受限**的，此时任何关于函数空间的比较**不成立**——所有方法
被同一个与几何无关的天花板压着。健康值在 1.0–1.5。runner 现在每行都输出它。

### 3.4 两条 cutoff 规则，必须同时满足

选 $K$（基的列数）时：

1. **近似能力对齐**：$K$ 要让几何感知的偏差下限跨设定可比（本项目取 $\approx 0.02$）。
   16 列在平面上给出 0.022，在球面 0.101，在三维 0.138——**固定 $K$ 而放任近似质量
   浮动七倍，比较的是截断而不是几何**。
2. **可辨识性**：$K \times R_3$ 不得超过传感器数 $|\mathcal S|$。

**两条冲突时，方法就在适用区之外。** 我们在波动方程上违反了第二条（$64\times16=1024$
对 552 个传感器），结果是我们的基更好却估不出来，输给估得准的差基。

---

## 第四部分：踩过的坑（请务必读完）

按代价从大到小。

### 4.1 秩从别的实验照搬（最大的一个）

`ranks = (4,4,6)` 来自冻结的一维实验，那里张量是 $18\times24\times24$。我把它原样搬到
5000–11000 节点的网格上，从未重新检查。后果是**整张主表被低估**：

| 布局 | ranks (4,4,6) | ranks (12,10,16) |
|---|---:|---:|
| `plane_barrier/sealed_4` 消融 | 3.12× | **9.73×** |
| 同上，对神经方法 | 1.26× | **2.68×** |
| `sphere/open_ocean` 消融 | 1.87× | **13.66×** |

而且基于错误数字写下的结论（"三维有天花板、坐标网络更好"）也部分是错的。

**教训**：任何从别处继承的超参数，在新的数据规模上都必须重新检查，检查工具就是 §3.3。

### 4.2 分辨率没按障碍尺度校验（犯了两次）

障碍比一个网格单元还薄时，结果不可信，**而且偏差的方向不可预测**：

- 第 11 轮：隔板 0.04 宽 vs 单元 0.056 → 粗网格把它表示得**更厚更连续**，**夸大**优势
  （6.28× 实际收敛值是 3.2–4.0×）；
- 第 14 轮：墙 0.12 m 厚 vs 单元 0.135 m → 墙**漏水**，**低估**优势
  （分辨率 90 给 3.4/4.0/10.1，分辨率 130 给 5.2/9.9/13.9，此后收敛）。

**教训**：分辨率必须由**最小几何特征的尺度**决定，不是由节点数决定。定下来之后要做
收敛检查。

### 4.3 想学 cutoff：谱 ARD 失败（结构性，不是调参）

尝试给每列一个精度，用稀疏贝叶斯不动点 $\alpha_k = R / \sum_r W_{kr}^2$ 自动剪枝。
**不工作**：$W$ 是点估计，更新式里缺了后验协方差项，于是罚项对每一列恒等于
$\beta_{\mathrm{reg}} R$，什么都剪不掉。cutoff 32 时有效维度仍是 32，NRMSE 只从
0.655 动到 0.643（cutoff 10 是 0.152）。

要真正剪枝需要**因子上的后验**，而那正是本方法明确不做的部分。代码已撤回，结论保留。

### 4.4 基线初始化：差点报出一个坏掉的基线

Tucker-HOOI 在随机初始化 + EM 填补下**发散**，held-out NRMSE 3–20。看上去像"我们大幅
领先"，实际上是基线坏了。换成 SVD 初始化后它是 0.27–0.65 的正常对手。

**教训**：基线必须以它的最好状态出场。我们现在还额外给它**开天眼选秩**（用 held-out
挑最优秩），我们自己固定秩——这样剩下的差距不可能被归因为调参。

### 4.5 一个被数据推翻的假设

两个真实数据集（圆柱绕流、方腔流）都没有优势。我最初归因为"对流主导，Laplacian 不是
那个动力学的算子"。**Péclet 扫描证明这是错的**：全密封布局在对流强过扩散一百倍时
仍有 6.5× 优势，因为再快的输运也过不去一堵封死的墙；有开口的布局则降到 0.95×。

正确的条件是：**几何必须仍然约束场能去哪里。动力学能绕开的障碍，不再是几何。**
圆柱是开放通道里的孤立小障碍，方腔是开放盒子——都在条件之外。

### 4.6 两个被证伪的早期设计假设

- $\varepsilon_{\text{sub}}$（低频子空间残差）**不预测**表示优势：子空间相距 0.374，
  场残差完全相同。
- joint operator 占优的区域**恰好是退化区**：场塌缩到算子最慢的几个特征向量上，
  此时任何基都行。

两条都记在 `ITERATIONS.md` 的 R5a。

---

## 第五部分：实验（哪些是对的，设定是什么）

**只有下面这些是当前有效的结果。** 更早的数字（`results/` 里 `_r5` 到 `_r12` 的目录）
是在被修正的配置下跑的，仅作为审计痕迹保留，**不要引用**。

### 5.1 冻结配置

```
张量        (12 场景, 12 时间, N 节点)
观测        10%，两种协议：spatial_sensors 与 random
噪声        观测标准差的 10%
ranks       (12, 10, 16)
cutoff      逐家族，见下表
Sobolev p   1.5
reg         2e-3
优化        AdamW 3e-3，1500 步，冷启动，无早停
种子        201–205（五个，全新）
```

**没有选择/确认两段**，因为没有任何配置是在这些数字上选的：隔板对比度、cutoff、
模态筛选、步数、分辨率，全部由拟合前诊断或收敛测试定下。

### 5.2 四个几何家族 + 一个真实几何家族

| 家族 | 几何以什么方式进入 | 节点 | cutoff | geometry-blind 对照 |
|---|---|---:|---:|---|
| `plane_barrier` | 方形内的不透隔板 | 5 520 | 16 | 同一算子、去掉隔板 |
| `plane_domain` | 孔洞与凹角 | 3 941–5 520 | 16 | 无视孔洞的重新三角剖分 |
| `volume_barrier` | 立方体内的隔墙（四面体） | 8 000 | 48 | 同一算子、去掉隔墙 |
| `sphere` | 闭曲面曲率（线性化浅水） | 10 242 | 32 | lat-lon 可分图表基 |
| `floorplan` | 房间与门洞（米制真实尺寸） | 11 310 | 16 | 同一算子、去掉墙 |

共 19 个布局。**每个家族都含一个"无几何"的阴性对照**，那里两个算子逐位相同，优势
必须精确为 1.00。

### 5.3 七个模型，每个有明确角色

| 模型 | 是什么 | 角色 |
|---|---|---|
| `geometry_operator` | 提出的方法 | — |
| `blind_operator` | **同一模型**，算子不知道几何 | 核心消融，只差这一件事 |
| `flat_chart` | 包围盒可分余弦基 | 按轴分解的张量方法 |
| `neural_tucker` | 因子 = 坐标 MLP，稠密核 | functional Tucker |
| `neural_cp` | 因子 = 坐标 MLP，对角核 | 神经 CP |
| `cp_als` / `tucker_als` | TensorLy，SVD 初始化，开天眼选秩 | 标准张量补全 |
| `permuted` | 打乱 index–算子对齐 | 破坏性对照，必须失败 |

### 5.4 主表（传感器协议 10%，五种子）

| 布局 | ours | 去掉几何 | 倍数 | 配对胜 | vs 神经 Tucker |
|---|---:|---:|---:|---|---:|
| `plane_barrier/open` **对照** | 0.021 | 0.021 | **1.00** | 2/5 | 1.50 |
| `plane_domain/square` **对照** | 0.021 | 0.021 | **1.00** | 2/5 | 1.51 |
| `volume_barrier/open` **对照** | 0.039 | 0.039 | **1.00** | 4/5 | 1.73 |
| `floorplan/open_plan` **对照** | 0.018 | 0.018 | **1.00** | — | 2.83 |
| `plane_domain/center_hole` | 0.020 | 0.035 | 1.73 | 5/5 | 1.56 |
| `plane_domain/U_shape` | 0.019 | 0.045 | 2.35 | 5/5 | 1.83 |
| `plane_barrier/arc` | 0.055 | 0.167 | 3.07 | 5/5 | 2.72 |
| `plane_barrier/chamber` | 0.048 | 0.178 | 3.70 | 5/5 | 1.55 |
| `plane_barrier/labyrinth` | 0.051 | 0.227 | 4.47 | 5/5 | 1.95 |
| `floorplan/corridor` | — | — | 4.51 | 5/5 | — |
| `floorplan/apartment` | — | — | **8.23** | 5/5 | — |
| `plane_barrier/sealed_4` | 0.029 | 0.267 | **9.19** | 5/5 | 2.54 |
| `floorplan/lab_suite` | — | — | **11.44** | 5/5 | — |
| `sphere/open_ocean` | 0.041 | 0.563 | **13.91** | 5/5 | 1.28 |

**二维十个布局全部战胜坐标网络（1.50–2.72×）。** 三维隔板布局是唯一的例外（0.51–0.85×），
机制见 §3.4 第二条。

复现命令：

```bash
PYTHONPATH=src python experiments/run_geometry_main.py \
  --families plane_barrier --masks spatial_sensors,random --ratios .10 \
  --seeds 201,202,203,204,205 --steps 1500 --n-scenarios 12 --n-time 12 \
  --ranks 12,10,16 --output results/rk_plane_barrier

PYTHONPATH=src python experiments/run_als_baselines.py \
  --families plane_barrier,plane_domain,volume_barrier,sphere \
  --seeds 201,202,203,204,205 --output results/rk_als_rest

PYTHONPATH=src python experiments/analyze_geometry_main.py \
  --inputs results/rk_plane_barrier results/rk_plane_domain \
           results/rk_volume_barrier results/rk_sphere results/rk_floorplan \
  --als-inputs results/rk_als_rest results/rk_als_floorplan \
  --output results/main_summary_r14 --ratio .10
```

### 5.5 算子不是只对扩散有效（波动方程）

同一套十个二维几何，抛物型换成阻尼双曲型，**随机缺失** 10%，三种子：

| 布局 | ours | 去掉几何 | 倍数 | vs 神经 Tucker |
|---|---:|---:|---:|---:|
| `plane_barrier/open` **对照** | 0.096 | 0.096 | **1.00** | 1.31 |
| `plane_barrier/labyrinth` | 0.122 | 0.220 | 1.80 | 1.94 |
| `plane_barrier/arc` | 0.118 | 0.237 | 2.01 | 2.50 |
| `plane_barrier/sealed_4` | 0.085 | 0.237 | **2.79** | 2.26 |

```bash
PYTHONPATH=src python experiments/run_geometry_main.py \
  --families plane_barrier --dynamics wave --basis-cutoff 64 \
  --masks random --ratios .10 --seeds 201,202,203 --steps 1500 \
  --n-scenarios 12 --n-time 12 --ranks 12,10,16 \
  --output results/wave_plane_barrier
```

⚠️ **传感器协议下同一批实验失败**，原因是违反 §3.4 第二条规则，不是物理问题。

### 5.6 适用条件（Péclet 扫描）

给真值加入无散度胞流（速度在隔板内部为零，所以几何一如既往真实），学习器不动：

| 布局 | Pe | ours | 去掉几何 | 倍数 |
|---|---:|---:|---:|---:|
| `chamber`（有开口） | 0 | 0.012 | 0.206 | **17.7** |
| `chamber` | 100 | 0.013 | 0.012 | **0.95** |
| `sealed_4`（全密封） | 0 | 0.026 | 0.279 | **10.7** |
| `sealed_4` | 100 | 0.033 | 0.215 | **6.5** |

```bash
PYTHONPATH=src python experiments/run_advection_limit.py --output results/advection_limit_r13
```

### 5.7 应用：几何值多少个传感器

`apartment`（11 310 节点，1%–20%，三种子）：

| 传感器 | ours | 去掉几何 | 神经 Tucker |
|---:|---:|---:|---:|
| 113（1%） | **0.040** | 0.194 | 0.228 |
| 2 262（20%） | 0.020 | 0.177 | 0.160 |

**基线在 20 倍预算下仍达不到我们 1% 预算的精度。** 它们不是数据不够，是函数空间放不下
墙上的断层。

```bash
PYTHONPATH=src python experiments/run_sensor_budget.py --output results/sensor_budget_r14
PYTHONPATH=src python experiments/plot_sensor_budget.py \
  --input results/sensor_budget_r14 --output results/figures_r14
```

### 5.8 图

| 文件 | 内容 |
|---|---|
| `results/figures_r14/basis_apartment.png` | **拟合前的字典**：三种基的特征函数叠在平面图上 |
| `results/figures_r14/reconstruction_apartment.png` | 同样 10% 传感器下的重构对比 |
| `results/figures_r14/sensor_budget.png` | 精度 vs 传感器数，一条持续下降两条躺平 |
| `results/figures_r14/geometry_ladder_*.png` | 19 个布局：拟合前偏差下限 vs 实际优势 |

---

## 第六部分：代码地图

### 6.1 当前主线（要读的）

| 文件 | 职责 |
|---|---|
| `benchmark.py` | **所有主实验的数据入口**。四个几何家族，一条代码路径 |
| `floorplan.py` | 楼层平面家族（米制真实尺寸的房间与门洞） |
| `grouped_operator_tucker.py` | **模型**。Tucker/CP、三种因子、闭式核后验 |
| `simplex_fem.py` | 任意维单纯形 P1 有限元（三角形/四面体/曲面），稠密与稀疏 |
| `operator_diagnostics.py` | **诊断量**。投影残差、稀疏特征对、可观测性筛选 |
| `als_baselines.py` | TensorLy 的 CP-ALS 与 Tucker-HOOI 封装 |
| `irregular_fem.py` | 多边形域的自建网格（无外部网格库） |
| `masks.py` | 观测协议（random / spatial_sensors / fibers） |

### 6.2 早期轮次的支撑（不要删，是审计痕迹）

`irregular_green_data.py`、`manifold_barrier_data.py`、`joint_diffusion_2d.py`、
`tensor_bayes.py`、`tensor_data.py`、`bases.py`、`operator_tucker_baselines.py`。

其中 `tensor_bayes.py` / `tensor_data.py` 是**冻结的一维锚点**，从头到尾没有改动过。

### 6.3 更早期、当前未被引用的模块

`neural_geometry.py`、`geometry_no_tensor.py`、`variational_domain_gp.py`、
`domain_kernels.py`、`the_well_pilot.py`、`well_baselines.py`、`neural_tensor.py`、
`irregular_domain_solver.py`、`independent_wave_solver.py`、
`traveling_harmonic_generator.py`、`phase_wave_protocol.py`、`bayes_data.py`、
`cli.py`、`statistics.py`、`plotting.py`。

这些来自项目更早的探索方向，**当前主线不依赖它们**。删除前请先确认没有 `results/`
里的产物需要它们复现。

### 6.4 测试

```bash
PYTHONPATH=src python -m pytest -q        # 56 项，约 75 秒
```

值得先读的几个，它们把关键性质写成了断言而不是散文：

- `test_benchmark.py::test_a_barrier_free_domain_leaves_nothing_to_know` — 阴性对照
- `test_benchmark.py::test_classical_completion_is_undefined_under_sensor_sampling` — §1.2
- `test_benchmark.py::test_the_mode_screen_reads_the_mask_and_never_the_data` — §3.2 无泄漏
- `test_manifold_barrier.py::test_sparse_and_dense_paths_agree_exactly` — §2.1
- `test_manifold_barrier.py::test_surface_elements_reproduce_the_sphere_spectrum` — FEM 对
  解析谱 $l(l+1)$ 的验证

---

## 第七部分：环境

```
Python      /home/ubuntu/project/yanjiu/.venv/bin/python
            （torch 2.8.0+cu128, numpy 2.5, scipy 1.18, tensorly 0.9.0）
必须         PYTHONPATH=src
硬件         单张 A100-80GB；模型很小（峰值约 1.5 GB），并行 6 个进程约 2.2 倍吞吐
无网格库     没有 gmsh / scikit-fem / meshio，网格是自建的（scipy.spatial.Delaunay）
```

换环境时需要的依赖只有 `torch`、`numpy`、`scipy`、`tensorly`、`matplotlib`。
`requirements-cu128.txt` 是本机的完整冻结列表。

---

## 第八部分：Related work（写论文时的定位）

> ⚠️ 下面按**方法族**组织，每族给出我确信存在的锚点工作。**具体的年份、作者拼写和
> 卷期请自行核对后再写进论文**——我没有联网，不应该被当作书目来源。

### 8.1 张量分解与补全

经典分解有两支：Tucker（Tucker, 1966）与 CP / CANDECOMP–PARAFAC（Carroll & Chang;
Harshman, 1970）。Kolda & Bader 的综述（*SIAM Review*, 2009）是标准入口。补全侧的
代表是带缺失的加权最小二乘，如 CP-WOPT（Acar et al.）以及各类核范数/低秩正则方法。
工程实现方面 TensorLy（Kossaifi et al., *JMLR*）是我们直接使用的基线来源。

**我们与它们的关系**：不是提出新的分解结构，而是**约束因子所在的函数空间**。在
`random` 采样下它们是正常对手；在 `spatial_sensors` 采样下它们的问题是未定义的（§1.2）。
这是我们最该强调的对比，因为它是**结构性**的，不是精度差距。

### 8.2 贝叶斯张量分解

用 ARD / 稀疏贝叶斯自动定秩是这一支的主线（Tipping 的 sparse Bayesian learning，
2001；MacKay 的 evidence framework，1992）；张量上的代表工作有各类 Bayesian
CP/Tucker with automatic rank determination。

**我们与它们的关系**：我们只在 **core** 上做闭式后验 + 证据不动点（§2.5），因子是
点估计。这个取舍有明确代价：§4.3 记录了谱 ARD 因缺少因子后验协方差而无法剪枝的
负结果。如果后续要做因子后验，这是一个自然的扩展方向。

### 8.3 Functional / 连续张量分解

把离散因子换成连续函数是我们最直接的邻居：functional tensor decomposition、
Gaussian-process 因子、以及用坐标网络参数化因子的做法。本项目的 `neural_tucker` /
`neural_cp` 就是这一族的实现（MLP 读 $(x,y)$ 或 $(x,y,z)$ 坐标）。

**我们与它们的关系**：同样是"因子是空间的函数"，但**函数空间的来源不同**——它们让网络
从数据里学出空间结构，我们直接从已知几何解出来。参数量差一个量级（288 vs 2982），
而且在传感器稀疏时我们更好（§5.4）。

### 8.4 隐式神经表示 / 神经场

SIREN（Sitzmann et al., 2020）、NeRF（Mildenhall et al., 2020）等把场表示为坐标到值的
网络。`neural_tucker` 可以看作它在张量分解框架内的对应物。

**关键区别**：坐标网络对"哪些点在物理上相邻"没有任何先验——墙两侧 5 厘米的两点在它
看来就是相邻的。这正是重构图（`reconstruction_apartment.png`）里它把浓度抹过墙的原因。

### 8.5 图 Laplacian 正则与流形学习

用图 Laplacian 做正则、或用其特征向量作基，是几何处理与流形学习的标准工具：Laplacian
eigenmaps（Belkin & Niyogi, 2003）、manifold harmonics（Vallet & Lévy, 2008）、
以及各类图正则矩阵/张量补全。

**我们与它们的关系**：这是**最近的邻居**，必须在论文里正面处理。区别有二：

1. 我们的算子来自**已知的物理几何**（FEM 装配、真实的边界条件与材料对比度），
   不是从数据里估计的相似度图；
2. 我们做了 **2×2 消融**把"几何"与"表示形式"分开：几何感知谱基 / 几何感知带罚自由表 /
   几何盲谱基 / 几何盲带罚自由表。结论是**几何是活性成分**，而**截断是让几何可用的
   那一步**（空间是稀缺坐标时，自由表因参数过多而崩溃）。

论文里应当把 `laplacian_geo` / `laplacian_blind` 这一对结果放出来，它直接回应
"这不就是图正则吗"。

### 8.6 物理信息机器学习与算子学习

PINN 一支把 PDE 残差放进损失；算子学习一支（DeepONet, Lu et al. 2021；FNO,
Li et al. 2021）学习解算子。

**我们与它们的关系**：信息层级不同，这一点要写清楚（§2.2）。我们**不假设已知方程**，
只假设**已知几何**——墙在哪里、边界是什么形状。材料（扩散率）是未知的。这比 PINN 弱得多，
也因此更容易在真实场景成立（楼层平面人人都有，扩散率没人测）。

### 8.7 传感器布放与压缩感知重构

从少数传感器重构场，是压缩感知与最优实验设计的经典问题。我们的 §5.7 与这一支直接相关，
而 §3.2 的可观测性判据本质上是一个（尚未展开的）传感器布放准则。

**这是一个自然的后续方向**：目前传感器是随机放的；用 $v_k$ 去**选**放哪里，应该能进一步
拉开差距。

---

## 第九部分：最重要的三个主实验（如果只能复现三个）

### 实验一：核心消融（论文的主表）

**问题**：几何信息本身值多少？

**设计的关键**：`geometry_operator` 与 `blind_operator` 是**同一个模型**——同一节点集、
同一 decoder、同一优化器、同一先验、同一闭式核后验——唯一区别是定义基的算子知不知道几何。
所以任何差异只能来自几何。

**必须同时报告阴性对照**：`open` / `square` / `open_plan` 上两个算子逐位相同，优势必须
精确为 1.00，胜率必须是随机的（1/5 或 2/5）。**这是整个论证的支点**：它排除了容量、
条件数、调参等一切替代解释。

设定见 §5.1，结果见 §5.4，命令见 §5.4 末尾。

### 实验二：传感器采样下经典方法未定义

**问题**：为什么需要一个把节点绑在一起的先验？

**设计的关键**：两种采样协议问两个不同的问题。`random` 下每个节点被观测多次，经典分解
良定；`spatial_sensors` 下未观测节点出现在零个方程里。

**报告方式**：不要说"我们赢 N 倍"，要说"**它们的因子行被零个方程约束**"，并给出
CP-ALS / Tucker-HOOI 在**所有布局、所有秩、所有迭代预算**下精确返回 1.000 的事实；
同时给出它们在 `random` 下是 0.27–0.66 的正常对手，证明不是把基线做残废。

基线必须以最好状态出场：**SVD 初始化**（随机初始化会发散到 3–20，§4.4），**开天眼选秩**。

### 实验三：适用条件（Péclet 扫描）

**问题**：这个方法什么时候不该用？

**设计的关键**：只改变生成真值的物理（加入对流），学习器的算子、基、网格、隔板全部不动；
速度在隔板内部为零，所以几何一如既往真实。

**结论的形状**：两条曲线终点不同才是结论——密封布局在 Pe=100 仍有 6.5×，有开口的布局
降到 0.95×。所以条件不是"算子类别必须正确"，而是**几何必须仍然约束场能去哪里**。

这个实验把两个真实数据集的失败从"没做出来"变成了"落在事前可查的条件之外"。

设定与结果见 §5.6。
