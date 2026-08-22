# Backlog

Ordered by what the paper needs next, not by what is interesting.

## Deferred: source localization from sparse sensors

Recovering *where* a release happened, rather than what the field is, is the
obvious downstream task and it is harder than it looks.  Reconstruction is
scored against a field we already have; localization needs a dataset built so
that the source is identifiable in the first place -- releases far enough apart
to be distinguishable at the sensor budget, a prior over source location that is
not the one that generated them, and an error measure in metres rather than
NRMSE, which makes it a different experiment rather than another column.

Worth doing as a follow-up branch.  Not worth doing badly to have a downstream
task in the table.

## Not planned

- **More geometric complexity.**  Nineteen layouts across five families is
  already more than the claim needs.  Mazes, office floors and building cores
  were written and reverted: they make the benchmark harder without making the
  argument stronger.
- **Chasing the three-dimensional sensor ceiling.**  The mechanism is understood
  and recorded; further tuning would be effort spent on a limitation rather than
  on the claim.

## 已验证会改善、但需要更多算力的方向

**直接优化边缘似然（把核积掉），而不是"因子 MAP + 核精确后验"。**

实测（秩 (8,6,10)，10% 传感器，两个种子）：无几何的对照上两者相同；有几何的
`sealed_4` 上边缘似然目标稳定更好约 14–15%（`0.0362 / 0.0312` 对
`0.0425 / 0.0363`），且估出的核先验精度更大，即对核收缩更强。

代价是每步都要形成 `Z^T Z`，即 `O(n P^2)`：核为 96 时只贵 1.1 倍，核为 1920
（当前配置）时贵 **18.5** 倍，单次拟合从 33 秒变成 10 分钟。

这是一个明确的精度／算力权衡，不是"只能这样做"。若有更大算力预算，值得把主表用边缘
似然目标重跑一遍。实现思路见报告 §3 中"为什么不把因子也放进边缘似然"一节。

## 未验证但形式上更优美的变体

不做谱截断，直接以稀疏精度矩阵 `(M + K/pi^2)^p` 作为空间因子的高斯马尔可夫随机场先验。
这样**不需要特征分解，截断阶数 K 这个超参彻底消失**，未观测节点仍通过精度矩阵的邻接
被约束。代价是空间参数量从 `K*R` 回到 `N*R`（当前配置下 256 变成约 18 万）。
