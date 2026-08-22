# 数据集：来源、构造、以及从零重建

本文档回答两个问题：

1. 现在的实验用的是什么数据、怎么来的；
2. **如果换一台机器、不用本地已有文件**，怎么从头把它们建起来。

---

## 第一部分：主实验的数据全部由代码生成

**这是好消息**：主表的 19 个布局、五个几何家族，**没有一个依赖外部文件**。装好
`torch / numpy / scipy` 就能在任何机器上完整复现，不需要下载任何东西。

### 1.1 唯一入口

```python
from geoaware.benchmark import FAMILIES, build_family

data = build_family("plane_barrier", "sealed_4",
                    n_scenarios=12, n_time=12)   # cutoff 与分辨率取家族默认值
```

`FAMILIES` 里登记了五个家族，每个家族自带**已冻结**的分辨率与 basis cutoff：

| 家族 | 分辨率 | 节点 | cutoff | 布局 |
|---|---:|---:|---:|---|
| `plane_barrier` | 80 | 5 520 | 16 | `open` `labyrinth` `arc` `chamber` `sealed_4` |
| `plane_domain` | 80 | 3 941–5 520 | 16 | `square` `center_hole` `two_holes` `L_shape` `U_shape` |
| `volume_barrier` | 20 | 8 000 | 48 | `open` `window` `chamber` `sealed_8` |
| `sphere` | 5（细分次数） | 10 242 | 32 | `open_ocean` |
| `floorplan` | 130 | 11 310 | 16 | `open_plan` `corridor` `apartment` `lab_suite` |

**这些数字不要随便改**，每一个都有来历：分辨率由**最小几何特征的尺度**定（见交接文档
§4.2），cutoff 由**近似能力对齐 + 可辨识性**两条规则定（§3.4）。

### 1.2 生成流程（以 `plane_barrier` 为例）

```
1. build_mesh(80, polygon=UNIT_SQUARE, seed=0)
      单位正方形上的抖动网格 + Delaunay 三角剖分（scipy.spatial.Delaunay）
      抖动是刻意的：不抖动的网格是伪装的张量积，会讨好"把域当矩形"的对照
   ↓
2. 真值材料  a_truth(x) = exp(0.3·sin(2π·⟨w,x⟩)/2)，障碍处置为 1e-2
   学习器材料 a_learner(x) = 1，障碍处置为 1e-2      ← 只有障碍位置是共享的
   ↓
3. assemble_sparse → 稀疏 K, M（P1 有限元）
   ↓
4. sparse_eigenpairs(K_truth, M, 60)  → 真值的 60 个模态
   sparse_eigenpairs(K_learner, M, 16) → 学习器的 16 列基（geometry_operator）
   sparse_eigenpairs(K_open, M, 16)    → 去掉障碍的 16 列基（blind_operator）
   ↓
5. 初始条件：12 个高斯鼓包（宽度为域尺度的 14%–26%，随机中心，seed 7717）
      刻意不用算子自己的特征函数——否则投影残差会变成生成器的假象
   ↓
6. 时间演化：Y[s,t,n] = Σ_q a_sq · exp(-(κ+λ_q)t) · φ_q(n)，t ∈ linspace(0.15, 3.0, 12)
   ↓
7. 全局标准化：(Y - mean) / std
```

障碍传导率 **1e-2（100:1 对比度）** 也有来历：1e-3 时隔板自身的弛豫比场还慢，它会变成
一个独立的慢子系统，任何截断基都得把前几个模态花在墙里面（见 `ITERATIONS.md` 第 11 轮）。

### 1.3 五个家族分别用什么几何

| 家族 | 几何机制 | 实现 |
|---|---|---|
| `plane_barrier` | 方形内的薄隔板（直的与弧形） | `Wall` / `ArcWall`，网格内的低传导率带 |
| `plane_domain` | 圆孔与凹角（L 形、U 形） | 从网格里**挖掉**，几何盲对照是无视孔洞的重新三角剖分 |
| `volume_barrier` | 立方体内的隔墙（带窗口） | 四面体网格 + `Partition` |
| `sphere` | 闭曲面曲率 | 测地球面（二十面体细分），Laplace–Beltrami，几何盲对照是 lat-lon 可分基 |
| `floorplan` | 房间与门洞 | `Segment`（带门洞的直墙），米制真实尺寸 |

`floorplan` 的几何来历要说清楚：**是按真实建筑的量级自己写的，不是从某栋具体建筑描下来的**。
12×8 m 楼层、0.9 m 标准门洞、0.12 m 墙厚。这一点写在元数据的 `geometry_provenance`
字段里，不要在论文里含糊过去。

### 1.4 换环境的注意事项

- **随机性**：网格抖动（`mesh_seed=0`）、初始条件（`scenario_seed=7717`）、置换对照
  （`permutation_seed=9173`）全部是固定种子，跨机器可复现。
- **无网格库依赖**：网格是自建的（`irregular_fem.py` / `simplex_fem.py`），只需要
  `scipy.spatial.Delaunay`。**不需要** gmsh / meshio / scikit-fem。
- **验证**：`build_family` 返回的元数据里有 `mesh_hash`，跨机器应当一致。

---

## 第二部分：外部真实数据（两个，都是负面结果）

这两个数据集**不在主表里**，它们的作用是验证适用条件（交接文档 §4.5、§5.6）。
如果你要复现这部分，下面是从零获取的方法。

### 2.1 RealPDEBench — 圆柱绕流（PIV 实测）

| | |
|---|---|
| **来源** | `AI4Science-WestlakeU/RealPDEBench` |
| **下载** | https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench |
| **子集** | `cylinder/real`（粒子图像测速的实测流场） |
| **原始网格** | 64 × 128；本项目用的是 2× 空间降采样 + 4× 时间降采样 → 32 × 64 |
| **本机路径** | `/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/realpde_cylinder_fresh_locked/locked_r64.npz` |
| **规模** | 11 条记录 × 998 帧 × 2 分量 × 32 × 64；Reynolds 数 1 875–11 625 |

从零获取：

```bash
pip install datasets huggingface_hub
python - <<'PY'
from datasets import load_dataset
# 数据集较大（该子集的 arrow 分片总计约 16 GB），建议先只取需要的分片
ds = load_dataset("AI4Science-WestlakeU/RealPDEBench", "cylinder", split="train",
                  streaming=True)
for record in ds.take(1):
    print({k: (getattr(v, "shape", type(v))) for k, v in record.items()})
PY
```

⚠️ **配置名（`"cylinder"`）与字段名需要按数据集卡片核对**，我没有联网确认过当前的接口。
本机的 `.npz` 是别人预处理好的产物，其 manifest 记录了来源分片与选择种子
（`subset_manifest.json`，`selection_seed=20260812`）。

**几何怎么拿到**：数据里没有障碍物掩膜，圆柱位置是**从数据本身按规则定出来的**：

```python
from geoaware.cylinder_flow import locate_cylinder
# 规则：取时均速度落在最低 2% 分位的格点，拟合覆盖它们的圆盘
centre_row, centre_col, radius = locate_cylinder(mean_speed, quantile=.02)
# 本数据上得到：圆心 (16.56, 22.12)，半径 3.96 格
```

规则是**写死并可审计**的，不是目测的。径向剖面本身很干净：距圆心 3 格内的时均速度是
0.088，6 格外是 0.198。

**结论**：几何感知基没有优势（1.00–1.08×）。原因见交接文档 §4.5——圆柱是开放通道里的
孤立小障碍，流体自由绕行，几何不约束场。

入口：`src/geoaware/cylinder_flow.py::cylinder_flow_tensor`

### 2.2 CFDBench — 方腔流（24 种长宽比）

| | |
|---|---|
| **来源** | CFDBench（OpenFOAM 生成的 CFD 基准） |
| **子集** | `cavity/geo`——**专门变化几何**的子集 |
| **本机路径** | `/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/cfdbench/extracted/cavity/geo/` |
| **规模** | 24 个案例，每个 `u.npy` / `v.npy` 形状 (23, 64, 64) |
| **几何** | height × width ∈ {0.01…0.05}²，长宽比 0.2–5.0；`case.json` 里有 |

CFDBench 的三个子集是 `bc`（变边界条件）、`prop`（变物性）、`geo`（**变几何**）。
只有 `geo` 与本项目相关。

从零获取：搜索 "CFDBench" 的官方仓库/发布页，下载 `cavity` 部分并解压。本机的
`raw/cavity.zip` 即原始压缩包。

**结论**：知道真实长宽比**反而更差**（0.74–1.07×）。长宽比 1.0 的四个案例精确并列
1.00（对照正确）。原因同上：方腔是开放盒子，顶盖驱动的剪切层结构与算子的低模态无关。

---

## 第三部分：如果要找**能成功**的真实数据

按交接文档 §4.5 确立的条件筛选，而不是随便试：

> **几何必须仍然约束场能去哪里。**

推荐方向（尚未验证，按符合条件的程度排序）：

1. **封闭/半封闭域内的标量场**——建筑内的温度或污染物、地下含水层、管网。房间/隔层
   之间只通过有限通道相连，这正是 `floorplan` 家族的真实版本。
2. **室内空气质量公开数据集**——传感器协议是它的原生采样方式。
3. **带强反射边界的声学/振动场**——腔体模态由几何决定。

**不推荐**：开放流场里的孤立障碍（圆柱、翼型绕流）、开放腔体驱动流。它们已经被验证
在条件之外，不是数据不好，是问题不在方法的适用区。

---

## 第四部分：`results/` 目录导航

**只有下面这些是当前有效的。** 其余目录是审计痕迹，是在**已被修正的配置**下跑的，
不要引用它们的数字。

| 目录 | 内容 |
|---|---|
| `rk_plane_barrier` `rk_plane_domain` `rk_volume_barrier` `rk_sphere` `rk_floorplan` | **主表**（种子 201–205，ranks (12,10,16)） |
| `rk_als_rest` `rk_als_floorplan` | 经典 ALS 基线（SVD 初始化，开天眼选秩） |
| `main_summary_r14` | 主表汇总 + 图 |
| `wave_plane_barrier` `wave_plane_domain` `wave_summary_r14` | 波动方程变体 |
| `advection_limit_r13` | Péclet 扫描（适用条件） |
| `sensor_budget_r14` | 传感器预算（应用研究） |
| `figures_r14` | 全部论文图 |

被取代的（保留但不要引用）：`irregular_*`、`wall_*`、`geometry_*`、`main_*_r12`、
`diffusion_*`、`track1_*`。它们对应 `ITERATIONS.md` 里第 5–13 轮，其中大部分是在
`ranks=(4,4,6)` 下跑的，那个配置被证明是秩受限的（交接文档 §4.1）。

**冻结不动的**：`diffusion_confirmation_r4` 及相关，是一维锚点，从头到尾没改过。
