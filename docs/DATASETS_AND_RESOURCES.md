> **已被取代。** 当前的数据来源、生成流程与从零重建方法见
> [`DATASETS.md`](DATASETS.md)。本文件保留为审计痕迹：它记录的是 2026-08-19 时的
> 数据规划，其中"下一数据构造顺序"一节的 POC-B 已在 R5a 被证伪（见
> [`ITERATIONS.md`](ITERATIONS.md)），不要照它执行。

# 方向 1：数据集与实现资源

更新时间：2026-08-19

中心总目录见 [Physics-Informed Tensor Learning Hub](https://github.com/xuangu-fang/Geo-Aware-Tensor/blob/master/SHARED_DATASETS.md)。本文件只回答 Operator-prior Tucker 的特殊问题：**能否为 tensor mode 构造一个不泄漏真值、且物理含义诚实的 operator basis？**

## 1. 当前已用数据

当前变系数扩散 Green tensor 由代码生成，入口为 src/geoaware/tensor_data.py::make_diffusion_green_tensor。axes 是 time × receiver × source，shape 为 18×24×24。truth 使用变系数 Neumann diffusion 的 14 modes；learner 只见常系数 reference operator 的 8 modes。冻结配置 projection residual 为 0.0698647，R4 fresh seeds 为 101–105。

它验证 nominal-operator mismatch，却尚未验证二维不规则边界、孔洞或 unseen geometry。

## 2. 本机可复用资源

| 路径 | 用途 | 限制 |
|---|---|---|
| /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/functional-operator-completion/data | 复用孔洞 geometry schema、SDF、domain split 的设计 | 现有数据不是 Green tensor，不能直接当 Track 1 主实验 |
| /mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a | source–receiver–time 语义的后期波动压力测试 | 需明确吸收边界、velocity operator 与左右谱；不作为首轮 |
| /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/Geo-Aware-Tensor/data | 历史 acoustic/irregular smoke test | 只用 manifest 指定的固定子集，不能根据结果挑样本 |

## 3. 下一数据构造顺序

### 3.1 先做规则二维 group-wise benchmark

在进入 mesh 工程前，先用规则二维 diffusion 回答 joint operator 是否必要。构造

$$
\mathcal L_\eta
=\mathcal L_x\otimes I+I\otimes\mathcal L_y+\eta\mathcal C_{xy}+\kappa I,
$$

其中 $\eta=0$ 为精确可分，$\eta>0$ 连续增加 nonseparable coupling。数据 axes 为 time × x × y × scenario；方法使用 coordinate groups $\{\{t\},\{x,y\},\{s\}\}$。每个版本必须保存 joint/per-axis matrices、operator separability residual、low-frequency subspace residual、projection residual 和 coordinate partition。

这里的 per-axis operator 是高效 approximation/ablation，而不是默认物理真值。必须区分解析可分、joint-to-axis projection 和 data-estimated 三种来源。

### 3.2 再做不规则 FEM Green benchmark

不规则阶段的目标不是立即下载更大数据，而是建立一个能精确审计 geometry/operator 的小 benchmark：

1. 域为单位方形减去 0/1/2 个圆孔；冻结 meshing tolerance 与边界类型；
2. 保存 node coordinates、elements、outer/hole boundary tags、stiffness K_g、mass M_g；
3. 求 K_g phi = lambda M_g phi，保存 eigensolver tolerance 和谱残差；
4. 生成 time × receiver-node × source-node screened-diffusion Green tensor；
5. train geometry 使用 0/1 hole，2/3 holes 作为 sealed topology test；
6. observation 为 2%/5%/10% random 与完整 source/receiver fibers；
7. 对每个 geometry 保存 exact/nominal/wrong-geometry 三套 basis 及 product projection residual。

两个 controlled POC 都先用 3 seeds、400 updates；规则二维 joint-vs-axis 趋势过 gate 后才进入不规则域，之后才冻结新 seeds 做 5-seed confirmation。详细 gate 和工程文件见 PAPER_TECHNICAL_REPORT_ZH.md 第 11–12 节。

## 4. 公开数据优先级

| 资源 | operator basis 可怎样得到 | 定位 |
|---|---|---|
| [PDEBench](https://github.com/pdebench/PDEBench) diffusion/reaction-diffusion/Darcy | 从官方生成代码和已知 BC 重建离散 reference operator；把生成器 commit 与矩阵 checksum 写入 manifest | 第一外部机制 gate |
| [AirfRANS](https://github.com/Extrality/airfrans_lib) | 只能从 mesh connectivity 构造 geometry Laplacian，除非另行可靠重建线性化 RANS operator | geometry-only 外测，不称 exact physics |
| [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench) cylinder | 从 domain/mesh 构造 geometry prior；真实测量侧的精确离散 operator 不可假定已知 | sim-to-real stress |
| [OpenFWI](https://openfwi-lanl.github.io/) | velocity map 可定义 wave operator；source/receiver 是自然双侧 modes | 高风险后期压力测试 |
| [The Well acoustic scattering](https://polymathic-ai.org/the_well/datasets_overview/) | 根据介质和边界 metadata 构造 Helmholtz/wave reference operator | 大规模后期验证，先固定小子集 |

## 5. 数据准入与禁止泄漏

- 用 truth field 反推并调到最优的 basis 是 oracle，只能报告上界，不能作为 proposed。
- geometry-correct basis 可以读取测试 geometry/mesh；能否读取测试 PDE coefficient 必须按任务定义预先声明。
- cutoff、rank、regularization 和 mesh resolution 必须在 selection geometries/seeds 上冻结。
- 新旧 geometry 的 basis transfer 必须保存 node correspondence/interpolation；禁止把 truth eigenvectors按 index 直接对齐。
- normalization 只读 observed training entries；metric 只算 held-out entries。
- 若外部数据没有可信 operator metadata，诚实改称 geometry-Laplacian prior 或放弃 Track 1。

## 6. 新 session 的第一周顺序

1. 原样复现 R4 表格与现有单元测试；
2. 实现 group partition、规则二维 joint operator、per-axis approximation 和 separability residual 测试；
3. 跑 joint/per-axis/wrong-joint/wide Neural Tucker 的 3-seed phase screen；
4. group-wise 趋势成立后，再实现 FEM K/M、谱残差和单孔小数据；
5. 信号存在后加 Laplacian-regularized Tucker 与 hybrid mode；
6. controlled gates 通过后才接 PDEBench；OpenFWI/The Well 不排在第一周。
