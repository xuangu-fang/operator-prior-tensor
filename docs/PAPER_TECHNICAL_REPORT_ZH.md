# 方向 1 论文级技术报告：Group-wise Operator-Prior Tucker

更新时间：2026-08-22

> **阅读顺序**：本文是按时间累积的，§1–§12 记录的是项目早期（一维 Green tensor）的
> 状态，**当前结论在 [§14](#14-2026-08-21-最终主线可事前判断的几何先验) 与
> [§15](#15-2026-08-22-主表定稿修正秩之后)**。想直接上手请读
> [`HANDOVER_ZH.md`](HANDOVER_ZH.md)。
>
> §1–§12 里的具体数字**不要引用**：它们来自 `ranks=(4,5,5)` 的一维实验，以及后来被
> 证明秩受限的 `(4,4,6)` 配置（见 `HANDOVER_ZH.md` §4.1）。这些章节仍然有效的部分是
> **formulation 与 inference 的符号定义**，以及 §9 关于"算子从哪来"的信息层级讨论。

当前冻结版本：`rk_*`（种子 201–205，ranks (12,10,16)）
当前判断：**十五个带几何的布局在两个采样协议下全部 5/5；四个无几何的对照精确并列。
二维十个布局全部战胜坐标网络。三维强隔板布局是已知例外，机制已定位。**

## 摘要

本文研究一个很具体的问题：当物理张量的一组坐标上存在可信的联合微分算子时，能否让该 operator 直接约束其真实定义的 coordinate group，从而减少连续 Tucker 分解在稀疏观测下需要学习的自由度？我们不要求每个 tensor axis 都有一条独立 PDE。

我们提出 Group-wise Operator-Prior Tucker。对有可信 operator 的 coordinate group $g$，把普通因子替换成
$$F_g=\Phi_g W_g,$$
其中 $\Phi_g$ 是定义在联合坐标 $x_g$ 上的物理算子有限谱基，$W_g$ 是待学习的小矩阵；没有可靠 operator 的 groups 保留 neural functional factors。现有 mode-wise 实现是每个 group 恰好含一个 tensor axis 的特例。模型仍保留非对角 Tucker core。训练阶段对 $W_g$、neural factors 和 core 做正则化点估计；因子固定后，对小 core 做解析高斯后验推断。

在变系数扩散 Green-response tensor 上，我们先用旧 seeds 选择 cutoff 8 和 Tucker rank $(4,5,5)$，随后完全冻结模型、400 次更新、噪声和所有超参数。五个全新 seeds 的结果表明：10% random mask 下，Operator Tucker 以 `0.164±0.011` 的 NRMSE 优于宽 Neural Functional Tucker 的 `0.207±0.054`，paired wins 为 4/5；10% receiver-fiber 下仍为 4/5 wins。置乱算子基的 negative control 接近 NRMSE 1，说明收益来自正确的 index–operator 对齐。不过 source-fiber 只有 3/5 wins，2%--5% 也不稳定。

因此本文最合适的主张不是“低秩张量用于 PDE”，而是：

> 一个近似正确的 operator-defined factor space 能显著减少 Tucker 因子的估计方差；有限谱截断同时引入可测的 projection bias，两者共同形成随观测率、截断阶数和算子失配变化的 bias–variance phase diagram。

## 1. 为什么要做这件事

### 1.1 普通 Tucker 忽略了 mode index 的物理含义

给定三阶张量 $Y\in\mathbb R^{N_1\times N_2\times N_3}$ 和稀疏观测集合 $\Omega$，普通 Tucker completion 写成
$$Y_{i_1i_2i_3}\approx \sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3} G_{r_1r_2r_3} U_1(i_1,r_1)U_2(i_2,r_2)U_3(i_3,r_3).$$
如果直接学习 $U_m\in\mathbb R^{N_m\times R_m}$，模型把每个 index 当作无关类别。它不知道：

- 相邻网格点应由扩散算子连接；
- Dirichlet、Neumann 或周期边界对应不同的函数空间；
- 网格改变、不规则边界或孔洞会改变 Laplacian 的特征函数；
- 时间 mode 的合理衰减率可由演化算子的谱给出。

神经 functional Tucker 用一维 MLP 替代 factor table，能让因子连续，但它仍需从稀疏样本中学习“什么函数是合理的”。当算子已知时，这部分自由度没有必要从头估计。

### 1.2 核心假设和适用边界

我们的核心假设只有一个：真实 tensor 在**有可靠 operator 的 coordinate groups** 上，大部分能量位于该联合算子的有限低频子空间附近。没有可靠 operator 的 groups 不受这个假设约束，而由 neural factors 表示。这个假设可被直接测量，而不是用“physics-informed”一词替代验证。

令 $\mathcal P_{\mathrm{op}}$ 是 operator-equipped groups，$P_g$ 是其 learner basis 的正交投影。对 grouped tensor 定义 product-space projection residual
$$\epsilon_{\mathrm{proj}} =\frac{\left\lVert Y-Y\times_{g\in\mathcal P_{\mathrm{op}}}P_g\right\rVert_F}{\lVert Y\rVert_F}.$$
$\epsilon_{\mathrm{proj}}$ 是任何被限制在这些 basis 中的方法都无法消除的近似偏差下界。较大的 basis 可减小它，却会增加待估计系数和 core interaction 的方差。因此本文要研究的是偏差和方差的平衡，而不是宣称 operator basis 永远正确。

## 2. 方法

### 2.1 Coordinate group 与算子谱

先将原始 tensor axes 划分成 coordinate groups $\mathcal P=\{g_1,\ldots,g_J\}$。operator 的作用域由真实 PDE 决定：它可以约束单个 mode，也可以约束合并后的 $(x,y)$、$(x,y,t)$ 或全部 mesh-node spatial group。

对有可信物理的 group $g$ 给定自伴正半定算子 $\mathcal A_g$：
$$\mathcal A_g\phi_{gk}=\lambda_{gk}\phi_{gk},\qquad 0\leq\lambda_{g1}\leq\lambda_{g2}\leq\cdots.$$
在联合离散坐标 $x_{g,i_g}$ 上评价前 $K_g$ 个特征函数，得到
$$\Phi_g(i_g,k)=\phi_{gk}(x_{g,i_g}),\qquad \Phi_g\in\mathbb R^{N_g\times K_g}, \quad N_g=\prod_{m\in g}N_m.$$
空间 group 可以使用完整二维有限差分、有限元或 graph Laplacian 的 eigenvectors；演化 group 可以使用由算子 eigenvalues 诱导的半群函数。方法不要求规则矩形，只要求能离线得到 joint discrete operator 的前若干特征对。只有 operator 可分或允许明确近似时，才将 $\Phi_g$ 替换成 per-axis product basis。

### 2.2 Group-wise Operator-prior Tucker

对 operator-equipped group，不直接学习大因子表，而是令
$$\widetilde F_g=\Phi_gW_g, \qquad W_g\in\mathbb R^{K_g\times R_g}.$$
对未知 group 使用 $F_g(i_g,:)=\operatorname{MLP}_g(x_g)$ 或 factor table。grouped prediction 是
$$\widehat Y_{i_1,\ldots,i_M} = \sum_{r_1,\ldots,r_J} G_{r_1,\ldots,r_J} \prod_{j=1}^{J}F_{g_j}(i_{g_j},r_j).$$
下面的 column normalization、spectral penalty 和 core posterior 对每个 operator group 原样成立。现有 Green-tensor 代码采用 singleton partition，因此下文保留 $m$ notation 描述已经冻结的实现；它是 group-wise formulation 的特例，而不是对所有新数据的强制拆轴规则。

为消除 Tucker 的连续缩放歧义，对每个 factor column 做单位 RMS 归一化：
$$s_{mr}=\sqrt{N_m^{-1}\lVert \Phi_mW_m(:,r)\rVert_2^2},\qquad U_m(:,r)=\frac{\Phi_mW_m(:,r)}{s_{mr}}.$$
预测保持标准 Tucker 形式：
$$\widehat Y_{ijk}=\sum_{abc}G_{abc}U_1(i,a)U_2(j,b)U_3(k,c).$$
Tucker core 是必要组件，而不是装饰：Green response 的 source/receiver modes 通常共享谱结构，但其跨 mode interaction 并不一定是 CP 的超对角形式。

### 2.3 算子谱正则化

令归一化因子对应的谱系数为 $\bar W_m(:,r)=W_m(:,r)/s_{mr}$。使用 Sobolev 型能量
$$E_m(\bar W_m)=\frac{1}{K_mR_m} \sum_{k,r}(1+\lambda_{mk})^p\bar W_{mkr}^2.$$
较高算子频率受到更强惩罚。实际优化目标是
$$\mathcal L= \frac1{|\Omega|}\sum_{(i,j,k)\in\Omega} (y_{ijk}-\widehat Y_{ijk})^2 +\rho\left[ \frac{\lVert G\rVert_F^2}{R_1R_2R_3}+\sum_m E_m(\bar W_m) \right].$$
这一步应准确称为 regularized point estimation 或 Gaussian MAP，而不是完整 Bayesian inference。当前实现用 AdamW、固定 400 steps、随机 cold start；确认集不参与 early stopping 或超参数选择。

### 2.4 固定因子后的 core 后验

训练因子后，对一个 entry $q=(i,j,k)$ 定义 Tucker row feature
$$z_q=U_1(i,:)\otimes U_2(j,:)\otimes U_3(k,:).$$
将 core 向量化为 $g=\mathrm{vec}(G)$，建立
$$g\sim\mathcal N(0,\alpha^{-1}I),\qquad y_\Omega\mid g\sim\mathcal N(Z_\Omega g,\beta^{-1}I).$$
于是
$$\Sigma_g=(\beta Z_\Omega^\top Z_\Omega+\alpha I)^{-1},\qquad \mu_g=\beta\Sigma_gZ_\Omega^\top y_\Omega.$$
$\alpha$ 和 $\beta$ 通过 evidence fixed-point updates 得到。预测均值和方差为
$$\mathbb E[y_*]=z_*^\top\mu_g, \qquad \operatorname{Var}(y_*)=z_*^\top\Sigma_gz_*+\beta^{-1}.$$
这是 conditional empirical Bayes：core posterior 在固定因子条件下是解析的，但 factor、rank、cutoff 和 operator 的不确定性没有被积分。因此当前论文以 NRMSE/MAE 为主，不能把不完整的 UQ 当作主要贡献。

### 2.5 参数量

确认配置为 $K=(8,8,8)$、$R=(4,5,5)$，因此 Operator Tucker 的可训练参数数为
$$8\times4+8\times5+8\times5+4\times5\times5=212.$$
我们同时使用两个 Neural Functional Tucker：

- **宽模型**：每个 mode 两层宽度 48 的 MLP，共 8130 参数，作为强容量对照；
- **同参数模型**：隐藏宽度 3，共 210 参数，rank 和 core 与 proposed 完全相同，用来隔离 operator basis 的参数效率。

报告两者很重要：与宽模型比较回答“operator prior 是否仍能战胜强 neural regression”；与同参数模型比较回答“收益是否只是 proposed 参数更少”。

## 3. 物理 POC 数据

### 3.1 变系数扩散 Green response

在一维 Neumann 区间上构造
$$\partial_tu+[L_a+\kappa I]u=0, \qquad L_a=-\partial_x(a(x)\partial_x),$$
其中
$$a(x)=\exp\{c[\cos(2\pi x)+0.35\sin(3\pi x+0.37)]\}.$$
真值使用 $L_a$ 的前 14 个 eigenmodes，张量为
$$Y(t,x_r,x_s)=\sum_{q=1}^{14} e^{-t(\kappa+\mu_q)} \psi_q(x_r)\psi_q(x_s)(1+\mu_q)^{-0.18}.$$
shape 为 `18×24×24`，三个 modes 分别是 time、receiver 和 source。contrast 固定为 $c=1$。learner 不读取真实变系数 eigenvectors，而只使用常系数 reference operator 的前 8 个 modes，因此不是把真值生成 basis 原样交给模型。该配置实测
$$\epsilon_{\mathrm{proj}}=0.0699.$$
观测加入相对于 observed-value 标准差 10% 的 Gaussian noise；标准化统计量仅由 observations 计算。

### 3.2 三种缺失协议

1. `random`：从所有 entries 中均匀抽取 2%/5%/10%。
2. `source_fibers`：随机选择若干固定 $(t,x_r)$ 对，并观测对应的整条 source vector $Y(t,x_r,:)$。
3. `receiver_fibers`：随机选择若干固定 $(t,x_s)$ 对，并观测对应的整条 receiver vector $Y(t,:,x_s)$。

后两种 mask 的缺失高度相关，不能被解释为随机 entry completion。每种协议都使用五个全新 seeds `101–105`；旧 seeds `41–43` 只用于冻结 cutoff 8 和 rank $(4,5,5)$。

## 4. Baselines 和它们各自排除的解释

| 方法 | 与 proposed 的共同部分 | 被移除或替换的部分 | 回答的问题 |
|---|---|---|---|
| Neural Functional Tucker（宽） | 相同 Tucker ranks/core、连续 mode factors | 用宽 MLP 替换 operator basis | 强 neural functional tensor 能否从数据学到更好因素 |
| Neural Functional Tucker（同参数） | 相同 ranks/core，约相同参数数 | 210 参数 MLP factors | operator basis 的参数效率 |
| Wrong-operator Tucker | 相同参数数、优化和谱 eigenvalues | 独立置乱 basis 的 index 对齐 | 正信号是否真的依赖正确物理排列 |
| Operator CP | 相同 operator factors | 对角 core | 非对角 Tucker interaction 是否必要 |
| Discrete Tucker | 相同 decoder | 自由 factor tables | 谱截断相对普通低秩的作用 |
| Flat operator GP | 相同 operator features | 不做 multilinear compression | Tucker bottleneck 是否必要 |

确认 gate 的主比较只用前三项，避免在 fresh seeds 上再次选择模型。其余 baseline 用于机制消融和旧 selection 阶段。

## 5. 冻结确认实验

### 5.1 预注册式冻结

- cutoff：8；truth modes：14；
- Tucker rank：$(4,5,5)$；
- contrast：1；noise：10%；
- optimizer：AdamW；steps：400；random initialization；
- ratios：2%、5%、10%，从不超过 10%；
- confirmation seeds：101、102、103、104、105；
- promotion gate：至少一个不超过 10% 的 ratio 上 Operator Tucker 对宽 Neural Tucker 达到 4/5 paired wins、NRMSE 明显低于 1，并在至少一种 structured-fiber mask 下复现。

### 5.2 结果解释原则

主指标是所有未观测 entries 上的 NRMSE。我们同时报告：

- 五 seed mean/std，而不是只展示最好 seed；
- paired seed wins，避免均值被单个异常 seed 支配；
- wrong-operator control 的绝对 NRMSE；
- 固定 projection residual 和参数量；
- source 和 receiver fibers 分开，不在看到结果后合并成更有利的“structured”均值。

数值表和图由 `experiments/analyze_confirmation_gate.py` 从原始 JSON 直接生成，最终结果见 `results/diffusion_confirmation_summary_r4/summary.json`。

### 5.3 五个 fresh seeds 的完整结果

表中均为 held-out NRMSE 的 mean±sample std；`wins` 是 Operator Tucker 相对宽 Neural Functional Tucker 的 paired seed 胜场。

| Mask | Obs. | Operator Tucker（212） | Neural F-Tucker wide（8130） | Neural F-Tucker matched（210） | Wrong operator（212） | wins / 5 |
|---|---:|---:|---:|---:|---:|---:|
| random | 2% | 0.302±0.091 | **0.263±0.049** | 0.438±0.031 | 1.045±0.073 | 1 |
| random | 5% | **0.230±0.028** | 0.239±0.035 | 0.426±0.035 | 0.987±0.013 | 2 |
| random | 10% | **0.165±0.010** | 0.207±0.054 | 0.428±0.035 | 0.942±0.014 | **4** |
| source fibers | 2% | 0.713±0.202 | **0.443±0.043** | 0.518±0.088 | 1.207±0.178 | 0 |
| source fibers | 5% | 0.445±0.083 | **0.358±0.120** | 0.435±0.032 | 1.162±0.141 | 2 |
| source fibers | 10% | 0.294±0.189 | **0.256±0.118** | 0.429±0.027 | 1.061±0.077 | 3 |
| receiver fibers | 2% | 0.773±0.151 | **0.440±0.069** | 0.466±0.046 | 1.040±0.051 | 0 |
| receiver fibers | 5% | 0.394±0.098 | **0.360±0.126** | 0.428±0.027 | 1.025±0.020 | 2 |
| receiver fibers | 10% | **0.217±0.052** | 0.269±0.112 | 0.422±0.024 | 0.960±0.029 | **4** |

这张表给出三个清楚结论。

第一，冻结 gate 按预先规则通过：random 10% 和 receiver-fiber 10% 都达到 4/5 wins，且绝对 NRMSE 分别为 0.165 和 0.217。第二，收益不是“参数更多”：proposed 只有宽 neural baseline 的约 2.6% 参数，并显著优于 210 参数的同参数 neural baseline。第三，收益也不是 Tucker decoder 本身造成的：置乱 operator 的同架构模型基本不能完成有效重建。

但 2% 的两个 structured masks 均明显失败；source-fiber 即使在 10% 也有 seed 104 的 0.631 异常失败，导致均值优势和 4/5 gate 均不成立。最准确的状态应是 **10% random/receiver-fiber conditional GO，extreme sparsity 与 source-fiber NO-GO**。

![冻结确认实验](../results/diffusion_confirmation_summary_r4/confirmation_nrmse.png)

![paired advantage](../results/diffusion_confirmation_summary_r4/paired_advantage.png)

## 6. 投稿主张、不能主张的内容与定位

### 6.1 可以主张

1. **可审计的 operator prior。** 物理信息不只是加入 loss，而是明确决定可学习 factor space；置乱 basis 会破坏性能。
2. **可测的 bias–variance mechanism。** projection residual 测量有限谱近似偏差；cutoff sweep 已表明 residual 更低不保证稀疏恢复更好。
3. **参数效率。** 212 参数 operator model 与 210 参数 neural Tucker 的直接对照隔离了表示先验。
4. **有限但真实的确认信号。** random 和 receiver-fiber 的 10% 结果通过冻结的 4/5 gate。

### 6.2 不能主张

- 不能声称 2% 下普遍优于 neural functional tensor；
- 不能声称所有 structured masks 都更强，source-fiber 只有 3/5；
- 不能声称这是 full Bayesian Tucker；
- 不能声称只凭当前一维 diffusion POC 已证明对任意 PDE、任意不规则域有效；
- 不能把较小 projection residual 等同于较小 reconstruction error。

### 6.3 与近邻工作的清楚区别

- [INN-TD（ICML 2025）](https://proceedings.mlr.press/v267/guo25p.html)把有限元局部插值函数嵌入 neural tensor architecture，重点是大规模 parametric PDE 的精度、稀疏计算和可解释性。我们的重点不是新的有限元网络，而是**已知 mode operator 的截断谱先验、稀疏 tensor completion 和可测 projection-bias/estimation-variance 相图**。
- [Sample Efficient Learning of Factored Embeddings of Tensor Fields（AISTATS 2024）](https://proceedings.mlr.press/v238/heo24a.html)通过 progressive sketching 和 Thompson sampling 选择 tensor slices，重点是数据采集策略和 compact sketches。我们的 observation mask 是外生固定的，贡献在 reconstruction prior，而不是 active sampling。
- [Functional Tensor Decompositions for PINNs（ICPR 2024）](https://arxiv.org/abs/2408.13101)用 neural CP/TT/Tucker 分离变量以缓解高维 PDE 求解。它证明 functional tensor architecture 本身并非新点；因此我们把 Neural Functional Tucker 作为主 baseline，创新聚焦于 operator-defined factor space 及其 bias–variance 可审计性。

## 7. 最简论文结构

1. **Introduction：** 稀疏 tensor completion 中 factor table 浪费已知物理结构；neural factors 连续但仍需从稀疏数据学习函数空间；提出 operator-defined finite factor space，并研究其 bias–variance 边界。
2. **Method：** mode operator spectra、Operator Tucker、normalized spectral penalty、conditional core posterior。
3. **Controlled phase diagram：** 连续改变 operator mismatch、cutoff、rank 和 observation ratio；横轴必须包含实测 projection residual。
4. **Physical confirmation：** diffusion Green tensor，fresh random/fiber masks，matched/wide neural baseline 和 wrong operator。
5. **Limitations：** structured source fibers、高 operator mismatch、局部高频和 posterior factor uncertainty。

## 8. 下一步只做与论文主张直接相关的工作

1. 保留当前 1D diffusion Green tensor 作为数学最干净的主实验；不在 confirmation seeds 上继续调参。
2. 将方法轻量扩展为 **group-wise Operator-Prior Tucker**：operator 约束其真实定义的联合坐标组，不能被默认拆成每轴一条虚构 PDE。
3. 在规则二维 diffusion 上比较 joint spatial operator 与 per-axis approximation，并把 operator separability residual 作为新的可审计横轴。
4. 只有规则二维 group-wise 机制跑通后，才进入不规则 mesh/孔洞；用 FEM stiffness/mass generalized eigenvectors检查 geometry change。
5. 增加 graph/Laplacian-regularized Tucker、宽 Neural Functional Tucker 与参数匹配对照，排除“任意平滑或额外容量都能得到相同结果”。
6. source-fiber 失败继续作为 identifiability limitation；只有预先定义的新协议和新 seeds 才能验证解释。

当前不建议增加更复杂的 attention、operator encoder 或 full factor posterior。论文价值来自一个简单可解释的限制、一个可测的失配量和一组严格冻结的对照。

## 9. 一个容易误解但必须讲清的问题：每一维的算子从哪里来？

### 9.1 不需要假设“张量每一维都有一条独立 PDE”

公式中写了 $\mathcal A_1,\mathcal A_2,\mathcal A_3$，这只是说每个 tensor mode 都需要一个可审计的函数空间先验，并不意味着研究者必须事先知道三条互不相关的 PDE。当前 Green-response POC 恰好只从**同一条扩散 PDE** 出发：
$$\partial_tu+(L_a+\kappa I)u=0, \qquad L_a=-\partial_x(a(x)\partial_x).$$
若 $L_a\psi_q=\mu_q\psi_q$，其 Green kernel 具有
$$G(t,x_r,x_s)=\sum_q e^{-t(\kappa+\mu_q)}\psi_q(x_r)\psi_q(x_s)c_q$$
这样的分离结构。因此三个 mode basis 的来源是：

| Tensor mode | 坐标 | basis 来源 | 当前 POC 的实现 |
|---|---|---|---|
| time | $t$ | 空间算子 eigenvalue 诱导的半群衰减 $e^{-t(\kappa+\mu_q)}$ | 对前 8 个参考衰减函数做 QR 正交化 |
| receiver | $x_r$ | Green operator 输出侧的 eigenfunctions | 常系数参考扩散算子的前 8 个 eigenvectors |
| source | $x_s$ | Green operator 输入侧/伴随侧的 eigenfunctions | 自伴情形与 receiver 使用同一组 eigenvectors |

所以真正的建模问题不是“如何猜出每一维的 PDE”，而是：**这个 mode 的 index 是否带有已知的物理关系；若有，哪个离散算子最诚实地表达该关系？**

对非自伴算子，source 和 receiver 不应再强行共用同一组基；自然选择是输出侧使用 right modes，输入侧使用 adjoint/left modes。对参数、材料编号等没有可信算子的 mode，可以保留 neural factor 或普通 factor table，形成 operator/neural hybrid Tucker。不能为了套公式而人为发明一个 PDE。

### 9.2 算子信息按可信度分四档

后续实验必须明确自己属于哪一档，不得混写：

1. **精确算子：** simulator 的离散 stiffness/mass matrix 与边界条件可得。此时可直接求低频谱，主要研究有限截断和稀疏估计。
2. **名义算子：** PDE family 和边界条件已知，但材料系数未知或有偏。当前 POC 属于这一档：真值是变系数扩散，learner 只见常系数 reference operator。
3. **几何算子：** 只知道网格、边界和孔洞，不知道精确 PDE 系数。可用 Laplace--Beltrami、FEM Laplacian 或 graph Laplacian；论文应称 geometry prior，而不是 exact physics prior。
4. **无可信算子：** 既无拓扑/几何，也无 PDE family。方向一不适用，应该退回 neural functional tensor；这也是方法的适用边界，而不是需要用复杂 encoder 掩盖的问题。

### 9.3 不规则边界和孔洞应怎样进入模型

在二维不规则域 $\Omega_g$ 上，不应分别为 $x$ 和 $y$ 人造两条一维 PDE。将所有有效 mesh nodes 视为一个空间 mode，有限元离散后求
$$K_g\phi_k=\lambda_k M_g\phi_k,$$
其中 $K_g$ 是包含边界条件和材料系数的 stiffness matrix，$M_g$ 是 mass matrix。孔洞通过 mesh connectivity 和 hole boundary condition 直接改变 $K_g$，从而改变 eigenfunctions。若张量是 $Y(t,x_r,x_s)$，则形状仍为 `time × receiver-node × source-node`，只是两个空间 axes 的 index 都来自同一个不规则 mesh，而不是规则网格坐标。

这给出一种明确的几何泛化含义：对新几何 $g'$，重新由其 $K_{g'},M_{g'}$ 计算低频 basis，再复用共享的小维映射/core 或进行少量观测下的适配。当前代码尚未验证这一点；现有 1D POC 只证明了“名义算子失配下仍可能降低方差”。

### 9.4 正式扩展：Group-wise Operator-Prior Tucker

旧写法容易被理解为：一个 $M$ 阶 tensor 的每个 axis 都必须配一条算子 $\mathcal A_m$。这不是我们希望主张的物理假设。正式原则改为：

> **The physical operator is attached to the coordinate group on which it is defined, rather than artificially projected onto every tensor axis.**
>
> 物理算子应该作用在它真实定义的联合坐标域上，而不是被人为投影到每一个张量坐标轴。

设原始坐标轴为 $\{1,\ldots,M\}$，将它们划分成互不重叠的 coordinate groups

$$\mathcal P=\{g_1,\ldots,g_J\}, \qquad \bigcup_{j=1}^J g_j=\{1,\ldots,M\}.$$

一个 group $g$ 可以只含一个 axis，例如 time；也可以合并多个耦合坐标，例如 $g=\{x,y\}$。将组内 index 合成 $i_g=(i_m:m\in g)$ 后，模型写为

$$\widehat Y_{i_1,\ldots,i_M} = \sum_{r_1,\ldots,r_J} G_{r_1,\ldots,r_J} \prod_{j=1}^{J}F_{g_j}(i_{g_j},r_j).$$

对有可靠 operator $\mathcal A_g$ 的 coordinate group，

$$F_g=\Phi_g W_g, \qquad \mathcal A_g\phi_{gk}=\lambda_{gk}\phi_{gk}.$$

对没有可靠 operator 的 group，不人为构造一条 PDE，而使用

$$F_g(i_g,r)=\operatorname{MLP}_g(x_g)_r$$

或普通 factor table。因此 group-wise 方法仍是显式低秩 Tucker；变化只是 factor 的作用域与真实 operator domain 对齐，而不是增加一个庞大的 neural operator encoder。

### 9.5 当前方法与 group-wise 方法的关系

| 场景 | 合理 factor/operator 处理 | 论文中的角色 |
|---|---|---|
| 当前 Green tensor | 一条 PDE 的半群和 Green kernel 分别诱导 time、receiver、source factors | 已完成的数学锚点；无需推翻 |
| 规则且精确可分 PDE | joint operator 是 Kronecker sum，可等价或近似拆成 per-axis operators | per-axis 版本是高效特例 |
| 规则但不可分的时空/空间 PDE | 把耦合的 $(x,y)$ 或 $(x,y,t)$ 合为 grouped mode，使用完整 joint operator spectrum | 新方法的关键主流 setting |
| 近似可分 operator | 比较 joint basis 与 per-axis projected/estimated operators | 可审计计算近似与 ablation |
| 不规则边界或孔洞 | 全部 mesh nodes 构成一个 spatial group，使用 FEM/graph operator | group-wise 的自然几何扩展 |
| 某些坐标无可信 operator | 对已知 groups 使用 operator factors，对未知 groups 使用 neural factors | hybrid setting，不强行物理化 |

这使两条故事互补而不冲突：

1. **Green tensor：** 一条 PDE 如何为多个 tensor modes 提供可推导的精确结构；
2. **Spatiotemporal field：** 一个联合 PDE operator 如何约束一组耦合的空间/时空 coordinates。

per-axis operator 不再被默认当作物理真值。只有在 operator 精确可分时它才是等价特例；在近似可分时，它是一个更便宜、误差可测的 approximation。

### 9.6 Operator separability residual

对规则二维离散 operator，令联合 stiffness/mass matrices 为 $(K_{xy},M_{xy})$。由 per-axis operators 构造 Kronecker-sum approximation

$$K_{\mathrm{sep}} = K_x\otimes M_y +M_x\otimes K_y +\kappa M_x\otimes M_y.$$

先对 operator 做 mass whitening：

$$\bar K=M_{xy}^{-1/2}K_{xy}M_{xy}^{-1/2}, \qquad \bar K_{\mathrm{sep}} =M_{xy}^{-1/2}K_{\mathrm{sep}}M_{xy}^{-1/2}.$$

定义主要诊断量

$$\epsilon_{\mathrm{sep}} = \frac{\lVert \bar K-\bar K_{\mathrm{sep}}\rVert_F} {\lVert \bar K\rVert_F}.$$

它测量“把 joint operator 拆成 per-axis operators”在算子层面的失真。实现还应报告低频子空间残差

$$\epsilon_{\mathrm{sub}} = \frac{\lVert P_{\mathrm{joint}}-P_{\mathrm{prod}}\rVert_F} {\lVert P_{\mathrm{joint}}\rVert_F},$$

其中 $P_{\mathrm{joint}}$ 是 joint low-frequency eigenspace 的 projector，$P_{\mathrm{prod}}$ 是 per-axis product basis 的 projector。$\epsilon_{\mathrm{sep}}$ 回答 PDE operator 有多不可分；$\epsilon_{\mathrm{sub}}$ 回答 completion 实际使用的有限低频 factor space 有多不一致。两者都必须由 learner 可见的 nominal operator 计算，不能读取 held-out field 后反推。

## 10. 当前 POC 的可执行规格：另一个 session 应先原样复现

### 10.1 数据生成与 learner 可见信息

- 网格：24 个空间点，Neumann zero-flux boundary；18 个时间点均匀覆盖 `0.025–0.55`。
- 真值 diffusivity：$a(x)=\exp\{\cos(2\pi x)+0.35\sin(3\pi x+0.37)\}$，其数值范围约为 `0.265–3.349`。
- 真值：变系数离散算子的前 14 个 modes，reaction $\kappa=0.15$，谱幅度乘 $(1+\mu_q)^{-0.18}$。
- learner basis：常系数算子，即把 contrast 设为 0，只保留前 8 个 modes；learner 不读取真实 eigenvectors。
- tensor shape：`18 × 24 × 24`，对应 time、receiver、source；product-space projection residual 为 `0.0698647`。
- 噪声：只在 observed entries 上加入标准差 `0.1 × std(Y_Ω)` 的独立高斯噪声；均值和尺度也只能由 observed entries 估计。

实现入口是 `src/geoaware/tensor_data.py::make_diffusion_green_tensor`。复现时首先检查 metadata 中的 truth/reference spectra、diffusivity range、projection residual 和 shape，任一不一致都不得把结果与 R4 合并。

### 10.2 训练与评估冻结项

- rank `(4,5,5)`；operator cutoff `(8,8,8)`；Operator Tucker 共 212 个参数。
- AdamW，learning rate `3e-3`，400 updates，random cold start；不按 validation early-stop。
- observation ratios 为 2%、5%、10%；mask 为 random、source-fibers、receiver-fibers。
- selection seeds `41–43` 只能解释早期 cutoff/rank 选择；confirmation seeds `101–105` 不能再用于选择。
- 主指标在**所有未观测 entries** 上计算 NRMSE；同时报告 MAE、逐 seed paired wins 和 projection residual。
- 主对照：宽 Neural Functional Tucker、210 参数匹配 Neural Functional Tucker、wrong-operator Tucker。wrong operator 必须只置乱 basis 的 index 对齐，同时保持 eigenvalues、参数量、优化器和 mask 不变。

可执行入口为 `experiments/run_tensor_bayes.py`，冻结汇总由 `experiments/analyze_confirmation_gate.py` 生成。新的 session 不应先重调当前 POC；应先复现表 5.3，再另开 experiment id。

主确认实验的等价命令为：

```bash
PYTHONPATH=src python experiments/run_tensor_bayes.py \
  --output results/diffusion_confirmation_r4_reproduction \
  --task diffusion_green --mismatch 1 --basis-cutoff 8 --truth-modes 14 \
  --models geo_btucker,neural_functional_tucker,neural_functional_tucker_matched,wrong_btucker \
  --ratios .02,.05,.10 --masks random,source_fibers,receiver_fibers \
  --seeds 101,102,103,104,105 --tucker-ranks 4,5,5 \
  --steps 400 --power 1.5 --reg .002 --noise .1 --init random --device cuda
```

若当前 CLI 或 CUDA 环境使该命令不能运行，应修复兼容性并记录 commit，不得静默更换 seeds、预算或数据参数。

### 10.3 已完成四轮迭代说明了什么

| 轮次 | 唯一改变 | 结论 |
|---|---|---|
| R1 operator mismatch | contrast 从 0 增至 2 | projection residual 从约 0.046 增至 0.097；10% 较常见正信号，2%/5% 不稳定 |
| R2 cutoff | cutoff 5/8/12 | residual `0.165/0.070/0.025`，但 2% NRMSE `0.293/0.273/0.331`；更低投影偏差不等于更低恢复误差 |
| R3 Tucker rank | `(3,4,4)/(4,5,5)/(6,7,7)` | 10% 信号跨多个 rank 存在，2% 仍受估计方差/优化限制 |
| R4 frozen confirmation | 新 seeds 101–105 | random 与 receiver-fiber 10% 达 4/5；source-fiber 和极稀疏条件未过 gate |

这四轮共同支持的是 bias--variance phase diagram，而不是“谱基越精确越好”或“物理先验在 2% 一定赢”。

## 11. 最新计划：五层 evidence ladder，而不是继续加组件

### POC-A：保留当前 Green tensor（已完成的数学锚点）

当前 1D Green-response confirmation 不删除、不重写结果，也不因 group-wise 扩展而降级。它回答一个非常干净的问题：**同一条 PDE 的谱怎样同时诱导 time decay、receiver eigenfunctions 和 source/adjoint eigenfunctions。**

新实验必须使用新的 experiment id 和 seeds；不得修改 R4 generator、artifacts 或冻结表格。论文中先用它介绍 operator prior 的机制和 bias--variance phase diagram，再进入更主流的 spatiotemporal grouped setting。

### POC-B：规则二维 joint operator vs per-axis approximation（最高优先级的新实验）

**唯一假设。** 当二维 operator 接近可分时，per-axis approximation 应与 joint grouped operator 接近且计算更便宜；随着不可分耦合和 $\epsilon_{\mathrm{sep}}$ 增大，joint operator factor 应更稳定地优于 per-axis factor。若性能差与 separability residual 无关，group-wise 主张不成立。

**受控 PDE family。** 在规则方形、固定 Neumann 或 periodic boundary 上构造

$$\mathcal L_\eta = -\partial_x\!\left(a_x(x)\partial_x\right) -\partial_y\!\left(a_y(y)\partial_y\right) +\eta\,\mathcal C_{xy} +\kappa I,$$

其中 $\mathcal C_{xy}$ 是固定的对称正半定 nonseparable diffusion coupling。$\eta=0$ 时离散算子是精确 Kronecker sum；增加 $\eta$ 产生连续可控的 joint coupling，同时保持同一 PDE family、边界、网格和噪声。

**任务。** 由不同初值或 forcing scenarios 生成

$$Y(t,x,y,s),$$

并使用 coordinate partition $\{\{t\},\{x,y\},\{s\}\}$。space group 使用 joint eigenbasis；scenario 没有 PDE 时使用 neural factor 或 table，不给它虚构算子。

**必须比较的模型。**

1. **Joint Group-wise Operator Tucker：** $F_{xy}=\Phi_{xy}W_{xy}$；
2. **Per-axis Operator Tucker：** 分别使用 $\Phi_x,\Phi_y$，作为计算效率 approximation；
3. **Wrong-joint operator：** 使用相同参数量但错误 $\eta$/错误 coupling 的 joint basis；
4. **Grouped Neural Functional Tucker：** 用二维 coordinate MLP 学 $F_{xy}(x,y)$；
5. **Discrete/Laplacian-regularized Tucker：** 使用相同 graph smoothness，但不做 spectral truncation。

joint 与 per-axis 的 decoder 维数不同，必须同时报告两种公平口径：matched trainable parameters，以及 matched effective spatial latent dimension。不能只展示对 proposed 更有利的一种。

**实验轴。**

- $\eta$ 至少包含精确可分、弱耦合、中耦合、强耦合四档；
- observations 为 2%/5%/10%，早筛重点 5% 和 10%；
- random entries 与 fixed spatial sensors（观测少量 $(x,y)$ sites 的完整时间轨迹）分开报告；
- 每个 cell 保存 $\epsilon_{\mathrm{sep}}$、$\epsilon_{\mathrm{sub}}$、product-space projection residual、held-out NRMSE、参数量、basis 构造时间、峰值内存和训练时间。

**预注册预测和 gate。**

1. $\eta=0$ 时 joint 与 per-axis 应接近；若 joint 大幅领先，先检查参数/截断是否不公平；
2. 随 $\epsilon_{\mathrm{sep}}$ 增大，per-axis minus joint NRMSE 应总体上升；
3. joint 必须在至少一个 5% 或 10% structured-sensor cell 达到 3/3 paired wins，同时 wrong-joint 明显退化；
4. 若 joint 只改善 projection residual、不改善 reconstruction，则回到 cutoff/rank bias--variance，而不增加 attention；
5. per-axis 若在低 residual 区域近似无损，则作为论文中的高效版本，而不是被描述成错误 baseline。

早筛使用 3 seeds、400 updates；只有上述趋势成立后才冻结 fresh 5-seed confirmation。

### POC-C：不规则域 + 孔洞的受控 grouped spatial operator（第三优先级）

**目的。** 检验最初设想的几何含义是否成立：边界和孔洞改变算子谱时，geometry-correct basis 是否比规则网格/错误几何 basis 更适合稀疏 functional Tucker。

**最小生成器。** 在单位方形减去 0/1/2 个圆孔的域上，用固定 mesh policy 解 screened diffusion/heat equation。每个 geometry 输出离散 $K_g,M_g$、node coordinates、boundary tags、solution/Green responses。首轮只做自伴椭圆/扩散，不加入 advection、非线性或移动边界。

**张量定义。** 优先使用 `time × receiver-node × source-node` Green tensor；若完整 source-node 扫描成本太高，可固定 16–32 个可复现 source locations，但必须保留 source coordinates 和 source-to-mesh map。

**四个必要模型。**

1. geometry-correct FEM spectral Tucker：由当前 geometry 的 $(K_g,M_g)$ 求基；
2. wrong-geometry Tucker：使用另一个孔洞布局的基，经明确 node correspondence/interpolation 后输入；
3. geometry-blind Neural Functional Tucker：输入坐标与时间，容量宽于 proposed；
4. Laplacian-regularized discrete Tucker：使用同一 mesh graph 的平滑正则，但不做谱截断，排除“任意平滑都有效”。

**实验轴。** observations 2%/5%/10%，random 与 source/receiver fibers；孔边界附近单独报告 boundary-band NRMSE。geometry mismatch 用孔位置偏移或孔半径偏移连续控制，并同时画 `projection residual → held-out NRMSE`。

**首轮 gate。** 3 seeds、400 updates。geometry-correct 在至少一个 5% 或 10% structured mask 上达到 3/3 paired wins，且相对 Neural Tucker 平均 NRMSE 降低至少 10%；wrong-geometry 明显退化；若只在训练几何内有效而 unseen geometry 失败，论文主张退回“geometry-specific completion”。

这里不再把 $x/y$ 当成两个独立 axes。全部有效 mesh nodes 构成一个 grouped spatial mode；geometry-correct basis 来自 $(K_g,M_g)$。规则二维 POC-B 先证明“为什么需要 joint group”，POC-C 再证明同一机制怎样自然延伸到孔洞和不规则边界。

### POC-D：只知道部分物理的 hybrid mode factors（第四优先级）

**目的。** 回答现实中某些 axes 没有已知 PDE 的问题，并防止方法被误解为要求每维精确算子。

构造 `time × receiver-node × material-parameter` tensor：time 与 receiver 使用参考扩散谱，material parameter 使用小 MLP factor。与 all-neural Tucker、错误地给 parameter mode 使用 RBF/Laplacian basis、以及 oracle operator Tucker 比较。只改变第三个 mode 的先验，保持 core/rank/预算一致。

成功标准不是必须超过 oracle，而是 hybrid 显著优于“强行给所有 mode 加错误算子”，并接近 all-neural 的灵活性，同时在 5%/10% 下优于 all-neural 的方差。

### POC-E：外部数据压力测试（第五优先级）

公开数据不应直接替代 controlled POC，因为很多数据没有离散算子或 Green source/receiver 语义。优先顺序是：

1. PDEBench 中扩散/反应扩散或 Darcy 数据：可复用其生成代码，补存离散 operator metadata；
2. AirfRANS 或 RealPDEBench cylinder：只用于检验 geometry-only Laplacian 是否仍有用，不能声称 exact operator prior；
3. OpenFWI/The Well acoustic：source–receiver–time 语义很合适，但波动算子、吸收边界和可能的非自伴/高频行为会引入另一篇论文级复杂度，暂不作为第一外部 gate。

外部数据若所有方法 NRMSE 接近 1，应判为任务未跑通，不讨论小幅相对优势。

## 12. 新 session 的工程交接清单

### 12.1 建议新增而不是改坏冻结实现

```text
src/geoaware/
├── grouped_operator_tucker.py # coordinate partition 与 joint/neural group factors
├── joint_diffusion_2d.py      # 可控 nonseparable 2D operator 与时空数据
├── operator_diagnostics.py    # separability/subspace/projection residuals
├── irregular_fem.py          # mesh、K/M、边界标签和低频广义特征对
├── irregular_green_data.py   # irregular-domain Green tensor 与 metadata
├── basis_transfer.py         # 新旧 geometry 间可审计的 node mapping
└── hybrid_operator_tucker.py # operator/neural 混合 mode factor
experiments/
├── run_grouped_joint_vs_axis.py
├── analyze_grouped_phase_diagram.py
├── run_irregular_green_poc.py
├── run_irregular_mismatch_sweep.py
└── run_hybrid_mode_poc.py
tests/
├── test_grouped_operator_tucker.py
├── test_operator_diagnostics.py
├── test_irregular_fem.py
├── test_irregular_green_data.py
└── test_hybrid_operator_tucker.py
```

冻结的 `make_diffusion_green_tensor` 与 R4 artifacts 不应原地修改。新数据版本必须保存：mesh hash、geometry parameters、boundary types、$K/M$ checksum、eigensolver tolerance、source locations、split、mask indices、noise seed 和 projection residual。

除此之外，规则二维 grouped POC 必须保存 coordinate partition、joint/per-axis operator checksum、coupling $\eta$、$\epsilon_{\mathrm{sep}}$、$\epsilon_{\mathrm{sub}}$、两类公平预算口径和 basis 构造成本。per-axis operators 必须注明是解析可分特例、由 joint operator 投影得到，还是从数据估计；三者不能混写。

### 12.2 最低测试要求

1. $\eta=0$ 时 joint matrix 与 Kronecker-sum approximation 在容差内一致，$\epsilon_{\mathrm{sep}}\approx0$；
2. 增大 nonseparable coupling 时 $\epsilon_{\mathrm{sep}}$ 按预定趋势变化；
3. group/un-group reshape 前后 tensor entries 与 masks 完全一致；
4. generalized eigenvectors 满足 $\Phi^TM\Phi\approx I$ 和 $K\Phi\approx M\Phi\Lambda$；
5. 孔内不存在有效 observation node，孔边界标签可重建；
6. 同 seed 的 operator、mesh、source 和 mask bitwise reproducible；
7. wrong-operator/wrong-geometry controls 不得意外读取 truth basis；
8. normalization 只使用 training observations，held-out metric 不包含 observed entries；
9. 3-seed screening 结束后先形成机器可读 summary，再决定是否消费 5-seed confirmation。

### 12.3 明确停止条件

- 若 correct 与 wrong geometry 的 projection residual 和恢复误差都不可区分，先检查 tensor 是否真的包含边界敏感结构，不增加网络复杂度。
- 若 $\epsilon_{\mathrm{sep}}$ 增大但 joint 与 per-axis 的差距没有趋势，停止把 group-wise 作为贡献，per-axis 保留为工程近似。
- 若 $\eta=0$ 时 joint 明显胜过 per-axis，先审计 spatial latent dimension、basis cutoff 和参数预算，不能当作正信号。
- 若 correct residual 明显更低但恢复不更好，优先调 cutoff/rank/regularization 的 bias--variance 轴，而不是加入 attention。
- 若所有方法 NRMSE 接近 1，说明 task、mask 或优化尚未跑通；不能把 proposed 的微小领先当作正信号。
- 若只有 exact truth operator 有效而 nominal/geometry operator 完全失败，方向一只能定位为 simulator-metadata method，不能讲广泛几何感知。

共享数据的位置、官方资源、准入条件和各数据适合哪一类实验，见 [`DATASETS_AND_RESOURCES.md`](DATASETS_AND_RESOURCES.md)。

---

## 13. 2026-08-19 定稿方向：Geometry-aware group-wise tensor factorization

本节记录当前正在推进的主线，取代第 11 节中"POC-B 规则二维 joint vs per-axis"的
优先级安排。第 1–10 节的冻结实验与结论**不变**，继续作为方法的一维锚点。

### 13.1 一句话故事

> 对于不同的几何边界，普通张量分解方法不好用；把边界信息提取成算子谱子空间来约束
> 因子，就能在稀疏观测下明显更好。
>
> **Geometry-aware group-wise tensor factorization through operator-defined
> functional subspaces.**

主表必须简单到一眼能懂：一个二维域，若干障碍物/隔板，一个时空张量，2%–10% 观测。
不追求通用性，只证明机制在明确界定的场景下 work。

### 13.2 为什么放弃"joint vs per-axis 可分性"作为主轴

R5a 的诊断（见 `ITERATIONS.md`）证伪了两条假设：`ε_sub` 不预测表示优势；joint 占优的
区域恰好是张量塌缩为 rank-5 的退化区，而那正是报告第 12.3 节列为 NO-GO 的
"正信号只在 eigenbasis-generated truth 上存在"。因此 `ε_sep`/`ε_sub` 降级为诊断量，
不再作为论文主轴。

### 13.3 主数据：一个网格，多种障碍布局

单位方形上固定一套 P1 三角网格。障碍是网格内部**近零传导率的薄带**（直隔板或弧形
隔板），而不是从网格里挖掉的洞。这个选择很关键：

- 所有布局共享**同一节点集**，因此任何对照与 proposed 只差"算子是否知道障碍"一件事，
  不需要任何节点对应或插值；
- learner 被告知障碍位置（这就是要利用的几何信息），但**不知道**背景材料的平滑变化，
  所以这是 geometry prior 而非 exact-physics prior（对应第 9.2 节第 3 档）。

张量为 `Y(scenario, time, node)`：同一 PDE 从若干光滑初始条件出发的扩散场，空间模态是
全部网格节点。第二个 setting 保留 `Y(t, receiver-node, source-node)` 的 Green 响应，
与冻结的一维实验语义完全一致。

布局按"几何有多重要"排序，用几何盲基的投影残差度量。下表是**冻结配置**
（`basis_cutoff = 10`，分辨率 18，`truth_modes = 60`）下的实测值：

| 布局 | 几何盲偏差下限 | 几何感知偏差下限 | 比值 |
|---|---:|---:|---:|
| `open`（无障碍，阴性对照） | 0.081 | 0.081 | 1.0× |
| `labyrinth` | 0.104 | 0.077 | 1.3× |
| `arc`（弧形，坐标网络难以表示） | 0.116 | 0.090 | 1.3× |
| `chamber` | 0.303 | 0.067 | 4.5× |
| `sealed_4`（四个密封象限） | 0.391 | 0.062 | 6.3× |

（此前版本的表来自 `basis_cutoff = 32`，比值高达 101×。截断降到 10 是第 13.5 节的
可辨识性规则所要求的，绝对偏差因此整体抬高；**必须引用上表**，它对应实际报告结果的
那个配置。）

### 13.3b 主表：冻结配置 + 全新种子（101–105）的确认结果

以下为论文主表的当前数据，判据在见到这些数字之前已写死（见 `ITERATIONS.md` 第 8 节
的预注册协议）。传感器协议、10% 观测、held-out NRMSE，均值 ± 五个种子的样本标准差：

| 模型 | open | labyrinth | arc | chamber | sealed_4 |
|---|---:|---:|---:|---:|---:|
| **几何算子（ours）** | 0.260±0.019 | 0.237±0.026 | 0.238±0.009 | **0.246±0.096** | **0.167±0.013** |
| 拓扑抹除 | 0.260±0.019 | 0.271±0.025 | 0.267±0.019 | 0.477±0.111 | 0.629±0.223 |
| 包围盒乘积基 | 0.247±0.035 | 0.258±0.041 | 0.269±0.050 | 0.475±0.114 | 0.627±0.221 |
| 坐标网络（宽） | 0.269±0.020 | 0.279±0.017 | 0.305±0.026 | 0.342±0.048 | 0.467±0.078 |
| 坐标网络（参数匹配） | 0.274±0.025 | 0.290±0.039 | 0.319±0.062 | 0.372±0.026 | 0.540±0.063 |
| 离散 Tucker | 1.578±0.194 | 1.575±0.193 | 1.584±0.190 | 1.596±0.151 | 1.578±0.152 |
| 置换对照 | 1.232±0.054 | 1.222±0.067 | 1.244±0.092 | 1.254±0.073 | 1.199±0.045 |

配对胜负：`chamber` 5/5、5/5、4/5；`sealed_4` 5/5、5/5、5/5（依次对拓扑抹除、包围盒、
宽坐标网络）。`open` 上前两行**逐位相同**，阴性对照成立。

随机缺失协议更强，且能下探到 2%：`sealed_4` 上 ours 为 `0.150 / 0.160 / 0.180`
（10%/5%/2%），对应几何盲谱基 `0.418 / 0.426 / 0.452`、宽坐标网络
`0.236 / 0.238 / 0.263`，全部 5/5 配对胜。

### 13.3c 相变位置：优势何时开启

把上表按"几何盲基的偏差下限"排序，得到一个定量边界而非定性论断（10% 传感器）：

| 布局 | 盲基偏差下限 | 盲基 / ours | 坐标网络 / ours |
|---|---:|---:|---:|
| `open` | 0.082 | 1.00 | 1.24 |
| `labyrinth` | 0.104 | 1.01 | 1.14 |
| `arc` | 0.116 | 1.03 | 1.29 |
| `chamber` | 0.303 | 2.03 | 1.65 |
| `sealed_4` | 0.391 | 3.18 | 2.58 |

ours 在整个家族上稳定在 `0.158–0.223`。**只要盲基偏差下限低于这个可达误差，优势严格
为 1.00；一旦超过就陡升。** 论文应陈述的因此不是"几何有用"，而是：几何先验在忽略几何
所引入的近似误差追平估计器可达误差时开始生效。这与第 1–10 节冻结的一维 bias–variance
边界是同一条规律，只是换成沿几何轴度量。

### 13.4 2×2 消融：把"几何"与"表示形式"分开

两个轴交叉：算子**是否知道障碍**，以及因子是**截断谱基**还是**带同一光滑罚的自由表**。

| | 谱截断 | 自由表 + 光滑罚 |
|---|---|---|
| **知道几何** | `fem_operator`（proposed） | `laplacian_geo` |
| **不知道几何** | `topology_erased` / `bounding_box` | `laplacian_blind` |

外加 `neural_coords`（宽坐标 MLP）、`neural_matched`（参数匹配）、`discrete_table`
（无先验）、`permuted`（破坏 index–算子对齐的破坏性对照）。

这个设计的关键性质：在 `open` 布局上，两个谱变体与两个罚变体**分别数值相同**，因为
没有障碍时它们本就是同一个算子。优势必须在这里精确归零，否则说明存在混淆。

### 13.5 可辨识性规则

传感器协议（只观测少数节点的完整轨迹）下发现：basis cutoff 必须与被观测节点数相称。
324 个节点只观测 16 个时，cutoff 32、rank 8 意味着 256 个节点系数只有 16 个位置提供
约束，所有谱模型 NRMSE 接近 1。cutoff 降到 10 后恢复正常，且在 cutoff 6/10/16 上稳定。

这不是调参技巧而是可陈述的设计规则，应写进论文的实践指导。我们尝试用 per-column
spectral ARD 自动确定有效维度，**失败**：点估计的 ARD 不动点缺少后验协方差项，惩罚对
每列的贡献恒为 `reg·R`，不产生剪枝。该负结果已记录，代码已撤回。

### 13.6 可迁移性

因为所有布局共享节点集，几何迁移的对比异常干净：把在几何 A 上学到的谱系数与 core
搬到几何 B，只把 `Φ_A` 换成 `Φ_B`。源域 10% 传感器，目标域 2% 传感器，五对布局 ×
三个种子的平均：

| 模型 | 源域 | zero-shot | few-shot（只重拟合 core） | few-shot（全参数微调） | 目标域从零训练 |
|---|---:|---:|---:|---:|---:|
| **几何算子（ours）** | 0.191 | 1.512 | **0.535** | 0.976 | 1.013 |
| 拓扑抹除 | 0.370 | 0.649 | 7.703 | 4.128 | 3.443 |
| 坐标网络 | 0.340 | 0.651 | 3.764 | 0.742 | 1.039 |
| 离散 Tucker | 1.543 | 1.502 | 20.086 | 1.425 | 1.482 |

最干净的陈述是内部对比：换基后只重拟合 core，在 **15/15** 个 pair×seed 格子上优于在
目标域 2% 上从零训练（0.535 vs 1.013），且全参数微调反而更差（0.976）——价值恰在于
不重估因子。报告 `few_shot_full` 与 `scratch` 是为了公平：只重拟合 core 是算子模型的
自然动作、却是坐标网络的糟糕动作，只引用它会构造性地偏袒 proposed。

**方向性**：迁移到结构更强的目标 3/3 全胜；`sealed_4 → open` 0/3，这是自洽性检查而非
缺陷——在无障碍目标上拓扑抹除算子本就是正确算子。规则：目标几何的结构不少于源几何时
迁移有效。

**局限**：zero-shot 我们较差（1.512，差于盲基线的 0.65），因为换基改变了列归一化，
迁移过来的系数在绝对尺度上未标定。只有函数空间迁移了，函数空间中的坐标没有。

### 13.7 当前不做的事

不做球面/浅水、不接 PDEBench、不追求跨 PDE 家族的通用性。这些属于机制确立之后的
扩展。当前唯一目标是把"几何信息 → 算子子空间 → 稀疏张量补全"这条链在受控场景下
做到无可争议。

---

## 14. 2026-08-21 最终主线：可事前判断的几何先验

本节取代第 13 节的实验安排。第 13 节的方法论述仍然有效，但主张的**组织方式**变了，
实验也全部在统一的 `src/geoaware/benchmark.py` 上重跑。

### 14.1 主张的正确形式

第 13 节写的是"用几何信息重构更好"。跑完四类几何、两个真实数据集和一个受控对流扫描
之后，这句话不够准确，也不是最强的那句。数据支持的是：

> **几何先验值不值得用，可以在拟合任何模型之前算出来。**
> 值得时把几何写进函数空间显著更好；不值得时优势精确为零。而"值不值得"有一个可计算、
> 且在所有已测场景（含两个真实数据集）上都成立的判据。

这个组织方式之所以更强，是因为我们的胜负分布本身不均匀：二维稳赢、三维输、真实数据打平。
若标题主张是"更好"，三维与真实数据是两处硬伤；若标题主张是"**可事前判断的适用条件**"，
它们是落在判据正确一侧的**样本外证据**。

### 14.2 Benchmark：四类几何，一条代码路径

| 家族 | 几何以什么方式进入 | 节点数 | cutoff | geometry-blind 对照 |
|---|---|---:|---:|---|
| `plane_barrier` | 方形内的不透隔板 | 5 520 | 16 | 同一算子、去掉隔板 |
| `plane_domain` | 孔洞与凹角 | 3 941–5 520 | 16 | 无视孔洞的重新三角剖分 |
| `volume_barrier` | 立方体内的隔墙 | 8 000 | 48 | 同一算子、去掉隔墙 |
| `sphere` | 闭曲面曲率（线性化浅水） | 10 242 | 32 | lat-lon 可分图表基 |

cutoff **逐家族**而非全局，按"使几何感知近似下限可比（≈0.02）"在拟合前定下。16 列在
平面留下 0.022 的下限、球面 0.101、三维 0.138——固定截断而放任近似质量浮动七倍，
比较的就是截断而不是几何。任何一次比较中两个基**永远拿到同样多的列**。

### 14.3 主表：消融（五个全新种子 101–105，10% 传感器观测）

| 布局 | ours | 同一模型去掉几何 | 倍数 | 配对胜 |
|---|---:|---:|---:|---|
| `plane_barrier/open`（阴性对照） | 0.117 | 0.117 | **1.00** | 1/5 |
| `plane_domain/square`（阴性对照） | 0.117 | 0.117 | **1.00** | 1/5 |
| `volume_barrier/open`（阴性对照） | 0.273 | 0.273 | **1.00** | 0/5 |
| `plane_barrier/labyrinth` | 0.111 | 0.245 | 2.20 | 5/5 |
| `plane_barrier/arc` | 0.153 | 0.219 | 1.43 | 5/5 |
| `plane_barrier/chamber` | 0.109 | 0.202 | 1.84 | 5/5 |
| `plane_barrier/sealed_4` | 0.094 | 0.279 | **2.98** | 5/5 |
| `plane_domain/center_hole` | 0.080 | 0.085 | 1.07 | 5/5 |
| `plane_domain/two_holes` | 0.098 | 0.106 | 1.08 | 5/5 |
| `plane_domain/L_shape` | 0.088 | 0.097 | 1.11 | 5/5 |
| `plane_domain/U_shape` | 0.065 | 0.087 | 1.33 | 5/5 |
| `volume_barrier/sealed_8` | 0.299 | 0.381 | 1.27 | 5/5 |
| `sphere/open_ocean` | 0.315 | 0.591 | **1.87** | 5/5 |

十二个带几何的布局在两个协议下**全部 5/5**；三个阴性对照并列到小数点后三位、胜率随机。
两侧共享节点集、decoder、优化器、先验与闭式核后验，只差一件事。

没有选择/确认两段，因为**没有任何配置是在这些数字上选的**：隔板对比度、cutoff、
模态筛选、步数、球面不加陆地，全部由拟合前诊断或收敛测试定下。

### 14.4 基线

**二维上全胜**：对 neural functional Tucker 1.05–1.32×（4/5 或 5/5），对 neural
functional CP 1.51–1.88×。倍数不大，但**参数量差一个量级**：`sealed_4` 上 288 个参数
对 2 982 个，且误差更低（0.094 vs 0.107）。在 5 520 个节点上，坐标网络要用 MLP 学出
整个空间结构，我们只在算子已给好的特征函数上学系数。

**经典 ALS 按协议分裂，而这个分裂本身是结论**（SVD 初始化 + held-out 开天眼选秩，
即报的是经典方法的性能上界）：

- 随机缺失：CP-ALS / Tucker-HOOI 是 0.27–0.66 的真对手；
- 传感器采样：所有布局、所有秩、所有迭代预算**精确返回 1.000**。未观测节点的因子行
  出现在零个观测条目中——问题在那个模型类里不是难，是**未定义**。

### 14.5 适用条件：几何必须仍然约束场

`build_advected_barrier` 给真值加入无散度胞流（Péclet 数标定），学习器的算子、基、网格、
隔板全部不动；速度在隔板内部为零，所以几何一如既往地真实。

| 布局 | Pe | ours | 去掉几何 | 盲/ours |
|---|---:|---:|---:|---:|
| `chamber`（有开口） | 0 | 0.012 | 0.206 | **17.7** |
| `chamber` | 10 | 0.030 | 0.109 | 3.6 |
| `chamber` | 100 | 0.013 | 0.012 | **0.95** |
| `sealed_4`（全密封） | 0 | 0.026 | 0.279 | **10.7** |
| `sealed_4` | 100 | 0.033 | 0.215 | **6.5** |

两条曲线终点不同，这是结论所在。**对流本身并不破坏算子先验**：全密封布局在对流强过
扩散一百倍时仍有 6.5×，因为再快的输运也过不去一堵封死的墙。有开口的布局则被对流把场
直接搬过开口，Pe=100 时优势精确归零。

于是条件不是"算子类别必须正确"，而是：

> **几何必须仍然约束场能去哪里。动力学能绕开的障碍，不再是几何。**

### 14.6 两个真实数据集：都落在条件之外

| 数据 | 来源 | 结果 |
|---|---|---|
| 圆柱绕流 | RealPDEBench，PIV 实测，Re 1 875–11 625 | 1.00–1.08×（Neumann/Dirichlet/涡量均试） |
| 方腔流 24 个长宽比 | CFDBench，OpenFOAM | 0.74–1.07×（长宽比 1.0 精确并列） |

两者的共同点正是 14.5 的条件：**圆柱是开放通道里的孤立小障碍，方腔是开放盒子**，
流体想去哪去哪，没有约束可供算子编码。不是难，是在条件之外，而条件事前可查。

### 14.7 已知失效边界

1. **三维传感器采样下有天花板**。十二组配置全扫，坐标网络全部更好（我们最好 0.207 vs
   0.177）。机制清楚：三维达到同等近似质量的模态数按 `k^d` 增长，而 cutoff 被可辨识性
   顶死（8 000 节点、800 个传感器、cutoff 48 × rank 16 = 768 个系数）。消融在最好配置上
   **也最强**（1.50×），说明卡住的是截断而非几何。
2. **`plane_domain` 的优势偏小**（1.07–1.33×），与其几何盲偏差下限较小一致。
3. **真实数据尚无正面结果**。按 14.5 的条件，下一步应找**几何真正约束场**的场景
   （封闭/半封闭域），而不是开放流场里的孤立障碍。


---

## 15. 2026-08-22 主表定稿：修正秩之后

第 14 节的结论方向正确，但**数字全部被低估**，原因是一个贯穿全部实验的配置错误。本节
给出修正后的主表，并记录这个错误，因为它比任何单个数字都值得写下来。

### 15.1 错在哪：秩是从一维实验照搬的

`ranks = (4, 4, 6)` 来自冻结的一维实验，那里张量是 `18 x 24 x 24`。我把它原样搬到了
5 000–11 000 个节点的网格上，从未重新检查。后果是**每一个模型都停在远高于自身近似
下限的地方**：`plane_barrier/sealed_4` 上 ours 达到 0.089，而它的基本可以到 0.022。

在这种状态下，任何关于**函数空间**的比较其实是在比较**秩天花板**——所有方法被同一个
与几何无关的限制压着。楼层平面家族让它暴露：偏差下限相差 3.37 倍，实际拟合只差 1.03 倍。

### 15.2 缺的那个检查

我每一轮都做拟合前诊断，但它们全部在问同一个问题：

| 检查 | 问的是 | 做了吗 |
|---|---|---|
| 投影残差 | 基能否张成这个场？ | ✅ 每轮都做 |
| **达到误差 / 自身下限** | **秩能否兑现这个基？** | ❌ 一次都没做 |

第二个比值远大于 1 时，函数空间的比较**不成立**。它只需要一次除法，现在是 runner 的
标准输出项，每一行结果旁边都有。

### 15.3 修正后的主表（种子 201–205，10% 传感器，ranks (12,10,16)）

| 布局 | ours | 去掉几何 | 倍数 | 旧值 | vs 神经 Tucker |
|---|---:|---:|---:|---:|---:|
| `plane_barrier/open`（对照） | 0.021 | 0.021 | **1.00** | 1.00 | 1.50 |
| `plane_domain/square`（对照） | 0.021 | 0.021 | **1.00** | 1.00 | 1.51 |
| `volume_barrier/open`（对照） | 0.039 | 0.039 | **1.00** | 1.00 | 1.73 |
| `floorplan/open_plan`（对照） | 0.018 | 0.018 | **1.00** | — | 2.83 |
| `plane_domain/center_hole` | 0.020 | 0.035 | 1.73 | 1.07 | 1.56 |
| `plane_domain/U_shape` | 0.019 | 0.045 | 2.35 | 1.33 | 1.83 |
| `plane_barrier/arc` | 0.055 | 0.167 | 3.07 | 1.43 | 2.72 |
| `plane_barrier/chamber` | 0.048 | 0.178 | 3.70 | 1.84 | 1.55 |
| `plane_barrier/labyrinth` | 0.051 | 0.227 | 4.47 | 2.20 | 1.95 |
| `floorplan/corridor` | — | — | 4.51 | — | — |
| **`floorplan/apartment`** | — | — | **8.23** | — | — |
| **`plane_barrier/sealed_4`** | 0.029 | 0.267 | **9.19** | 2.98 | 2.54 |
| **`floorplan/lab_suite`** | — | — | **11.44** | — | — |
| **`sphere/open_ocean`** | 0.041 | 0.563 | **13.91** | 1.87 | 1.28 |

**二维十个布局全部战胜坐标网络（1.50–2.72×）**，此前只有 1.05–1.32×。四个阴性对照
仍然精确并列。

真实几何（楼层平面）在同等偏差下限上优势**高于**合成隔板：`apartment` 在 0.175 上
8.23×，而合成 `chamber` 在 0.173 上只有 3.70×。

### 15.4 一个训练前就能算的标量，跨 19 个布局

把 5 个家族的 19 个布局放在同一张图上（`results/figures_r14/geometry_ladder_*.png`），
横轴是几何盲基的投影残差（无需训练），纵轴是实际得到的优势倍数。四个阴性对照聚在
比值 1.00，其余单调上升。三维的三个点系统性偏低，原因见 15.6。

### 15.5 算子不是只对扩散有效

同一套十个二维几何，把抛物型换成双曲型（阻尼波动），随机缺失 10%：

| 布局 | ours | 去掉几何 | 倍数 | vs 神经 Tucker |
|---|---:|---:|---:|---:|
| `plane_barrier/open`（对照） | 0.096 | 0.096 | **1.00** | 1.31 |
| `plane_barrier/labyrinth` | 0.122 | 0.220 | 1.80 | 1.94 |
| `plane_barrier/arc` | 0.118 | 0.237 | 2.01 | 2.50 |
| `plane_barrier/sealed_4` | 0.085 | 0.237 | **2.79** | 2.26 |

### 15.6 两条 cutoff 规则必须同时满足

传感器协议下同一批波动实验**失败**（0.53–0.98×）。原因不是物理，是我只用了两条规则
中的一条：

1. **近似能力对齐**：cutoff 要让几何感知的偏差下限跨家族可比 → 波动场需要 cutoff 64；
2. **可辨识性**：`cutoff x rank` 不能超过传感器数 → 64 × 16 = 1024 > 552。

两条冲突时方法就在适用区之外。headroom 把它写在脸上：ours 达到 0.255 而下限 0.015
（**17×**，严重欠定），blind 达到 0.249 而下限 0.228（1.09×，正常）——**我们的基更好
但估不出来，它的基更差但估得准**。

球面同为波动方程却成功（13.91×），正因为 10 242 个节点给出 1 024 个传感器，而
cutoff 32 × rank 16 = 512 < 1024，两条规则都满足。三维隔板布局偏低是同一机制。

### 15.7 应用：几何值多少个传感器

`apartment`（11 310 节点，1%–20% 传感器，三个种子）：

| 传感器 | ours | 去掉几何 | 神经 Tucker |
|---:|---:|---:|---:|
| 113（1%） | **0.0400** | 0.194 | 0.228 |
| 2 262（20%） | 0.0202 | 0.177 | 0.160 |

**基线在 20 倍预算下仍达不到我们 1% 预算的精度**，而且从 1% 到 20% 几乎不动
（0.194 → 0.177）。它们不是数据不够，是函数空间放不下墙上的断层，加传感器补不上。
合成的 `sealed_4` 上基线最终能追上，代价是 **8 倍**传感器。
