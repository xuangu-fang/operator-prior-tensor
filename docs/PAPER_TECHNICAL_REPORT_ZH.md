# 方向 1 论文级技术报告：用物理算子约束连续 Tucker 因子

更新时间：2026-08-19

实验冻结版本：`diffusion_confirmation_r4`
当前判断：**随机缺失与 receiver-fiber 的 10% 确认 gate 通过；source-fiber 只有 3/5，因而是有边界的 GO，不是“极稀疏条件下普遍优于神经张量”。**

## 摘要

本文研究一个很具体的问题：当物理张量的每个 mode 对应时间、接收位置或激励位置，并且这些坐标上已有可信的微分算子时，能否直接利用该算子的低频特征函数，减少连续 Tucker 分解在稀疏观测下需要学习的自由度？

我们把普通 Tucker 因子表替换成

\[
U_m=\Phi_m W_m,
\]

其中 $\Phi_m$ 是第 $m$ 个物理算子的有限谱基，$W_m$ 是待学习的小矩阵。模型仍保留一个非对角 Tucker core，从而表达时间、接收端和源端之间的耦合。训练阶段对 $W_m$ 和 core 做带算子谱能量的正则化点估计；因子固定后，对小 core 做解析高斯后验推断。

在变系数扩散 Green-response tensor 上，我们先用旧 seeds 选择 cutoff 8 和 Tucker rank $(4,5,5)$，随后完全冻结模型、400 次更新、噪声和所有超参数。五个全新 seeds 的结果表明：10% random mask 下，Operator Tucker 以 `0.164±0.011` 的 NRMSE 优于宽 Neural Functional Tucker 的 `0.207±0.054`，paired wins 为 4/5；10% receiver-fiber 下仍为 4/5 wins。置乱算子基的 negative control 接近 NRMSE 1，说明收益来自正确的 index–operator 对齐。不过 source-fiber 只有 3/5 wins，2%--5% 也不稳定。

因此本文最合适的主张不是“低秩张量用于 PDE”，而是：

> 一个近似正确的 operator-defined factor space 能显著减少 Tucker 因子的估计方差；有限谱截断同时引入可测的 projection bias，两者共同形成随观测率、截断阶数和算子失配变化的 bias–variance phase diagram。

## 1. 为什么要做这件事

### 1.1 普通 Tucker 忽略了 mode index 的物理含义

给定三阶张量 $Y\in\mathbb R^{N_1\times N_2\times N_3}$ 和稀疏观测集合 $\Omega$，普通 Tucker completion 写成

\[
Y_{i_1i_2i_3}\approx
\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}
G_{r_1r_2r_3}
U_1(i_1,r_1)U_2(i_2,r_2)U_3(i_3,r_3).
\]

如果直接学习 $U_m\in\mathbb R^{N_m\times R_m}$，模型把每个 index 当作无关类别。它不知道：

- 相邻网格点应由扩散算子连接；
- Dirichlet、Neumann 或周期边界对应不同的函数空间；
- 网格改变、不规则边界或孔洞会改变 Laplacian 的特征函数；
- 时间 mode 的合理衰减率可由演化算子的谱给出。

神经 functional Tucker 用一维 MLP 替代 factor table，能让因子连续，但它仍需从稀疏样本中学习“什么函数是合理的”。当算子已知时，这部分自由度没有必要从头估计。

### 1.2 核心假设和适用边界

我们的核心假设只有一个：真实 tensor 在每个 mode 上的大部分能量位于已知算子的有限低频子空间附近。这个假设可被直接测量，而不是用“physics-informed”一词替代验证。

令 $P_m$ 是 learner basis 的正交投影，定义 product-space projection residual

\[
\epsilon_{\mathrm{proj}}
=\frac{\|Y-Y\times_1P_1\times_2P_2\times_3P_3\|_F}{\|Y\|_F}.
\]

$\epsilon_{\mathrm{proj}}$ 是任何被限制在这些 basis 中的方法都无法消除的近似偏差下界。较大的 basis 可减小它，却会增加待估计系数和 core interaction 的方差。因此本文要研究的是偏差和方差的平衡，而不是宣称 operator basis 永远正确。

## 2. 方法

### 2.1 每个 mode 的算子谱

对 mode $m$ 给定自伴正半定算子 $\mathcal A_m$：

\[
\mathcal A_m\phi_{mk}=\lambda_{mk}\phi_{mk},\qquad
0\leq\lambda_{m1}\leq\lambda_{m2}\leq\cdots.
\]

在离散坐标 $x_{mi}$ 上评价前 $K_m$ 个特征函数，得到

\[
\Phi_m(i,k)=\phi_{mk}(x_{mi}),\qquad
\Phi_m\in\mathbb R^{N_m\times K_m}.
\]

空间 mode 可以使用有限差分、有限元或 graph Laplacian 的 eigenvectors；演化 mode 可以使用由算子 eigenvalues 诱导的指数衰减函数。方法本身不要求规则矩形，只要求能离线得到离散算子的前若干特征对。

### 2.2 Operator-prior Tucker

我们不直接学习大因子表，而是令

\[
\widetilde U_m=\Phi_mW_m,
\qquad W_m\in\mathbb R^{K_m\times R_m}.
\]

为消除 Tucker 的连续缩放歧义，对每个 factor column 做单位 RMS 归一化：

\[
s_{mr}=\sqrt{N_m^{-1}\|\Phi_mW_m(:,r)\|_2^2},\qquad
U_m(:,r)=\frac{\Phi_mW_m(:,r)}{s_{mr}}.
\]

预测保持标准 Tucker 形式：

\[
\widehat Y_{ijk}=\sum_{abc}G_{abc}U_1(i,a)U_2(j,b)U_3(k,c).
\]

Tucker core 是必要组件，而不是装饰：Green response 的 source/receiver modes 通常共享谱结构，但其跨 mode interaction 并不一定是 CP 的超对角形式。

### 2.3 算子谱正则化

令归一化因子对应的谱系数为 $\bar W_m(:,r)=W_m(:,r)/s_{mr}$。使用 Sobolev 型能量

\[
E_m(\bar W_m)=\frac{1}{K_mR_m}
\sum_{k,r}(1+\lambda_{mk})^p\bar W_{mkr}^2.
\]

较高算子频率受到更强惩罚。实际优化目标是

\[
\mathcal L=
\frac1{|\Omega|}\sum_{(i,j,k)\in\Omega}
(y_{ijk}-\widehat Y_{ijk})^2
+\rho\left[
\frac{\|G\|_F^2}{R_1R_2R_3}+\sum_m E_m(\bar W_m)
\right].
\]

这一步应准确称为 regularized point estimation 或 Gaussian MAP，而不是完整 Bayesian inference。当前实现用 AdamW、固定 400 steps、随机 cold start；确认集不参与 early stopping 或超参数选择。

### 2.4 固定因子后的 core 后验

训练因子后，对一个 entry $q=(i,j,k)$ 定义 Tucker row feature

\[
z_q=U_1(i,:)\otimes U_2(j,:)\otimes U_3(k,:).
\]

将 core 向量化为 $g=\mathrm{vec}(G)$，建立

\[
g\sim\mathcal N(0,\alpha^{-1}I),\qquad
y_\Omega\mid g\sim\mathcal N(Z_\Omega g,\beta^{-1}I).
\]

于是

\[
\Sigma_g=(\beta Z_\Omega^\top Z_\Omega+\alpha I)^{-1},\qquad
\mu_g=\beta\Sigma_gZ_\Omega^\top y_\Omega.
\]

$\alpha$ 和 $\beta$ 通过 evidence fixed-point updates 得到。预测均值和方差为

\[
\mathbb E[y_*]=z_*^\top\mu_g,
\qquad
\operatorname{Var}(y_*)=z_*^\top\Sigma_gz_*+\beta^{-1}.
\]

这是 conditional empirical Bayes：core posterior 在固定因子条件下是解析的，但 factor、rank、cutoff 和 operator 的不确定性没有被积分。因此当前论文以 NRMSE/MAE 为主，不能把不完整的 UQ 当作主要贡献。

### 2.5 参数量

确认配置为 $K=(8,8,8)$、$R=(4,5,5)$，因此 Operator Tucker 的可训练参数数为

\[
8\times4+8\times5+8\times5+4\times5\times5=212.
\]

我们同时使用两个 Neural Functional Tucker：

- **宽模型**：每个 mode 两层宽度 48 的 MLP，共 8130 参数，作为强容量对照；
- **同参数模型**：隐藏宽度 3，共 210 参数，rank 和 core 与 proposed 完全相同，用来隔离 operator basis 的参数效率。

报告两者很重要：与宽模型比较回答“operator prior 是否仍能战胜强 neural regression”；与同参数模型比较回答“收益是否只是 proposed 参数更少”。

## 3. 物理 POC 数据

### 3.1 变系数扩散 Green response

在一维 Neumann 区间上构造

\[
\partial_tu+[L_a+\kappa I]u=0,
\qquad
L_a=-\partial_x(a(x)\partial_x),
\]

其中

\[
a(x)=\exp\{c[\cos(2\pi x)+0.35\sin(3\pi x+0.37)]\}.
\]

真值使用 $L_a$ 的前 14 个 eigenmodes，张量为

\[
Y(t,x_r,x_s)=\sum_{q=1}^{14}
e^{-t(\kappa+\mu_q)}
\psi_q(x_r)\psi_q(x_s)(1+\mu_q)^{-0.18}.
\]

shape 为 `18×24×24`，三个 modes 分别是 time、receiver 和 source。contrast 固定为 $c=1$。learner 不读取真实变系数 eigenvectors，而只使用常系数 reference operator 的前 8 个 modes，因此不是把真值生成 basis 原样交给模型。该配置实测

\[
\epsilon_{\mathrm{proj}}=0.0699.
\]

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

1. 把当前 1D diffusion POC 迁移到一个具有明确 boundary/operator metadata 的公开 solver benchmark；不在确认 seeds 上继续调参。
2. 在不规则 mesh 上使用有限元 stiffness/mass generalized eigenvectors，直接检查孔洞和边界改变时的 projection residual。
3. 增加完整 graph/Laplacian-regularized Tucker 与 side-information Bayesian CP，排除“任何平滑正则都能得到相同结果”。
4. source-fiber 失败应作为机制问题研究：它可能来自 time–receiver pair 覆盖不足导致的 factor identifiability，而不是 core 表达能力。只有预先定义的新协议和新 seeds 才能验证这一解释。

当前不建议增加更复杂的 attention、operator encoder 或 full factor posterior。论文价值来自一个简单可解释的限制、一个可测的失配量和一组严格冻结的对照。

## 9. 一个容易误解但必须讲清的问题：每一维的算子从哪里来？

### 9.1 不需要假设“张量每一维都有一条独立 PDE”

公式中写了 $\mathcal A_1,\mathcal A_2,\mathcal A_3$，这只是说每个 tensor mode 都需要一个可审计的函数空间先验，并不意味着研究者必须事先知道三条互不相关的 PDE。当前 Green-response POC 恰好只从**同一条扩散 PDE** 出发：

\[
\partial_tu+(L_a+\kappa I)u=0,
\qquad L_a=-\partial_x(a(x)\partial_x).
\]

若 $L_a\psi_q=\mu_q\psi_q$，其 Green kernel 具有

\[
G(t,x_r,x_s)=\sum_q e^{-t(\kappa+\mu_q)}\psi_q(x_r)\psi_q(x_s)c_q
\]

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

\[
K_g\phi_k=\lambda_k M_g\phi_k,
\]

其中 $K_g$ 是包含边界条件和材料系数的 stiffness matrix，$M_g$ 是 mass matrix。孔洞通过 mesh connectivity 和 hole boundary condition 直接改变 $K_g$，从而改变 eigenfunctions。若张量是 $Y(t,x_r,x_s)$，则形状仍为 `time × receiver-node × source-node`，只是两个空间 axes 的 index 都来自同一个不规则 mesh，而不是规则网格坐标。

这给出一种明确的几何泛化含义：对新几何 $g'$，重新由其 $K_{g'},M_{g'}$ 计算低频 basis，再复用共享的小维映射/core 或进行少量观测下的适配。当前代码尚未验证这一点；现有 1D POC 只证明了“名义算子失配下仍可能降低方差”。

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

## 11. 最新计划：三个递进 POC，而不是继续加组件

### POC-A：不规则域 + 孔洞的受控 Green tensor（最高优先级）

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

### POC-B：只知道部分物理的 hybrid mode factors（第二优先级）

**目的。** 回答现实中某些 axes 没有已知 PDE 的问题，并防止方法被误解为要求每维精确算子。

构造 `time × receiver-node × material-parameter` tensor：time 与 receiver 使用参考扩散谱，material parameter 使用小 MLP factor。与 all-neural Tucker、错误地给 parameter mode 使用 RBF/Laplacian basis、以及 oracle operator Tucker 比较。只改变第三个 mode 的先验，保持 core/rank/预算一致。

成功标准不是必须超过 oracle，而是 hybrid 显著优于“强行给所有 mode 加错误算子”，并接近 all-neural 的灵活性，同时在 5%/10% 下优于 all-neural 的方差。

### POC-C：外部数据压力测试（第三优先级）

公开数据不应直接替代 controlled POC，因为很多数据没有离散算子或 Green source/receiver 语义。优先顺序是：

1. PDEBench 中扩散/反应扩散或 Darcy 数据：可复用其生成代码，补存离散 operator metadata；
2. AirfRANS 或 RealPDEBench cylinder：只用于检验 geometry-only Laplacian 是否仍有用，不能声称 exact operator prior；
3. OpenFWI/The Well acoustic：source–receiver–time 语义很合适，但波动算子、吸收边界和可能的非自伴/高频行为会引入另一篇论文级复杂度，暂不作为第一外部 gate。

外部数据若所有方法 NRMSE 接近 1，应判为任务未跑通，不讨论小幅相对优势。

## 12. 新 session 的工程交接清单

### 12.1 建议新增而不是改坏冻结实现

```text
src/geoaware/
├── irregular_fem.py          # mesh、K/M、边界标签和低频广义特征对
├── irregular_green_data.py   # irregular-domain Green tensor 与 metadata
├── basis_transfer.py         # 新旧 geometry 间可审计的 node mapping
└── hybrid_operator_tucker.py # operator/neural 混合 mode factor
experiments/
├── run_irregular_green_poc.py
├── run_irregular_mismatch_sweep.py
└── run_hybrid_mode_poc.py
tests/
├── test_irregular_fem.py
├── test_irregular_green_data.py
└── test_hybrid_operator_tucker.py
```

冻结的 `make_diffusion_green_tensor` 与 R4 artifacts 不应原地修改。新数据版本必须保存：mesh hash、geometry parameters、boundary types、$K/M$ checksum、eigensolver tolerance、source locations、split、mask indices、noise seed 和 projection residual。

### 12.2 最低测试要求

1. generalized eigenvectors 满足 $\Phi^TM\Phi\approx I$ 和 $K\Phi\approx M\Phi\Lambda$；
2. 孔内不存在有效 observation node，孔边界标签可重建；
3. 同 seed 的 mesh、source 和 mask bitwise reproducible；
4. wrong-geometry control 不得意外读取 truth basis；
5. normalization 只使用 training observations；
6. held-out metric 不包含 observed entries；
7. 3-seed screening 结束后先形成机器可读 summary，再决定是否消费 5-seed confirmation。

### 12.3 明确停止条件

- 若 correct 与 wrong geometry 的 projection residual 和恢复误差都不可区分，先检查 tensor 是否真的包含边界敏感结构，不增加网络复杂度。
- 若 correct residual 明显更低但恢复不更好，优先调 cutoff/rank/regularization 的 bias--variance 轴，而不是加入 attention。
- 若所有方法 NRMSE 接近 1，说明 task、mask 或优化尚未跑通；不能把 proposed 的微小领先当作正信号。
- 若只有 exact truth operator 有效而 nominal/geometry operator 完全失败，方向一只能定位为 simulator-metadata method，不能讲广泛几何感知。

共享数据的位置、官方资源、准入条件和各数据适合哪一类实验，见 [`DATASETS_AND_RESOURCES.md`](DATASETS_AND_RESOURCES.md)。
