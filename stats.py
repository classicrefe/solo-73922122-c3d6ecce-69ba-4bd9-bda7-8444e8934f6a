"""抽查判定的统计计算。

判定模型:
  假设全批错录数 D 为"刚超过错录率上限的最小整数",
  在累计抽查 n 条、实测错录 x 条的检查点上, 计算超几何分布尾概率
      P(X <= x),  X ~ Hypergeometric(总体 N, 错录数 D, 无放回抽 n)
  若 P <= 5% / 检查点数, 则该检查点判定通过(批次放行)。
"""

from math import exp, lgamma

MAX_CHECKPOINTS = 3
ALPHA = 0.05


def min_exceeding_errors(size: int, limit_pct: int) -> int:
    """刚超过错录率上限的全批错录数(最小整数): floor(N * p / 100) + 1。"""
    if not isinstance(limit_pct, int) or isinstance(limit_pct, bool) or not 0 <= limit_pct <= 99:
        raise ValueError("错录率上限必须是 0~99 的整数")
    if not isinstance(size, int) or size < 1:
        raise ValueError("批量必须至少为 1")
    return size * limit_pct // 100 + 1


def _log_comb(a: int, b: int) -> float:
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def hypergeom_cdf(x: int, population: int, defects: int, draws: int) -> float:
    """P(X <= x): 从 population 件(含 defects 件错录)中无放回抽 draws 件,
    抽到不多于 x 件错录的概率。用对数组合数计算, 避免大数溢出。"""
    if not (0 < draws <= population):
        raise ValueError("抽查数必须在 1..批量 之间")
    if not (0 <= defects <= population):
        raise ValueError("错录数假设必须在 0..批量 之间")
    lo = max(0, draws - (population - defects))
    hi = min(x, draws, defects)
    if hi < lo:
        return 0.0
    log_den = _log_comb(population, draws)
    log_terms = [
        _log_comb(defects, k) + _log_comb(population - defects, draws - k) - log_den
        for k in range(lo, hi + 1)
    ]
    peak = max(log_terms)
    total = sum(exp(t - peak) for t in log_terms) * exp(peak)
    return min(1.0, max(0.0, total))


def pass_threshold(num_checkpoints: int) -> float:
    """通过阈值: 5% 除以检查点数。"""
    if not 1 <= num_checkpoints <= MAX_CHECKPOINTS:
        raise ValueError("检查点数必须为 1~3")
    return ALPHA / num_checkpoints
