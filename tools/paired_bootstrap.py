# Query-level paired bootstrap (10,000 resamples over the queries of both folds) and Wilcoxon signed-rank test
# between two models, from the per-query npz files written by eval_perquery.py (Section 5.1).
# usage: python tools/paired_bootstrap.py <model_a_fold_A.npz> <model_a_fold_B.npz> <model_b_fold_A.npz> <model_b_fold_B.npz>
import sys
import numpy as np
from scipy.stats import wilcoxon

B = 10000
rng = np.random.default_rng(0)


def load(fa, fb):
    a, b = np.load(fa), np.load(fb)
    return np.concatenate([a["ap"], b["ap"]]), np.concatenate([a["cmc"][:, 0], b["cmc"][:, 0]])


ap_a, r1_a = load(sys.argv[1], sys.argv[2])
ap_b, r1_b = load(sys.argv[3], sys.argv[4])
n = len(ap_a)
idx = rng.integers(0, n, size=(B, n))  # paired: the same resampled queries for both models
ci = lambda x: np.percentile(x, [2.5, 97.5])
for name, xa, xb in (("mAP", ap_a, ap_b), ("Rank-1", r1_a, r1_b)):
    ma, mb = xa[idx].mean(1), xb[idx].mean(1)
    d = ma - mb
    print(f"{name}: A {xa.mean():.4f} [{ci(ma)[0]:.4f}, {ci(ma)[1]:.4f}]  B {xb.mean():.4f} [{ci(mb)[0]:.4f}, {ci(mb)[1]:.4f}]  "
          f"A-B {d.mean():+.4f} [{ci(d)[0]:+.4f}, {ci(d)[1]:+.4f}]")
print(f"n = {n} queries, Wilcoxon signed-rank test on per-query AP: p = {wilcoxon(ap_a, ap_b).pvalue:.2e}")
