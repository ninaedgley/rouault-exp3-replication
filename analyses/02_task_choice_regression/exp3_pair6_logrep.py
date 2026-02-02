import numpy as np
import scipy.io as sio
import statsmodels.api as sm

MAT_PATH = "DATA/Exp3.mat"
mat = sio.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
Exp3 = mat["Exp3"]

X = np.array(Exp3.X_ser6_all, dtype=float)

#X_ser6_all columns (from BehaviorGroupExp3.m):
#c1 = trial index within pairing 6 ; c2 = pairing index ; c3 = block/difficulty code ; c4 = accuracy difference ; c5 = reaction time difference ; c6 = confidence difference ; c7 = task choice (1=task1, 2=task2)
assert np.all(X[:,1]==6), "Expected pairing id column to be all 6"
assert set(np.unique(X[:,6])).issubset({1,2}), "Expected choices coded as 1/2"

Xacc = X[:, 3] #accuracy different regressor
Xrt = X[:, 4] #RT difference regressor
Xconf = X[:, 5] #confidence difference regressor
choice = X[:, 6] #1 or 2

Y = (choice - 1).astype(int)

def z(x:np.ndarray) -> np.ndarray:
    """z-score with population std to match MATLAB std(x)."""
    return (x - x.mean()) / x.std(ddof=0)

n = Y.shape[0]

#Full Model : [accDiff, rtDiff, confDiff) + intercept
X_full = np.column_stack([z(Xacc), z(Xrt), z(Xconf)])
X_full = sm.add_constant(X_full)
res_full = sm.GLM(Y, X_full, family=sm.families.Binomial()).fit()

#Reduced Model : [accDiff, rtDiff] + intercept (drop confDiff)
X_red = np.column_stack([z(Xacc),z(Xrt)])
X_red = sm.add_constant(X_red)
res_red = sm.GLM(Y, X_red, family=sm.families.Binomial()).fit()

#BIC : (-2)*LL + k*log(n), k = #parameters (intercept included)
BIC_full = (-2.0) * res_full.llf + res_full.params.size * np.log(n)
BIC_red = (-2.0) * res_red.llf + res_red.params.size * np.log(n)

print("=== FULL MODEL (accDiff + rtDiff + confDiff) ===")
print(res_full.summary())
print(f"LL(full)={res_full.llf:.6f} k={res_full.params.size} BIC(full)={BIC_full:.4f}")

print("=== REDUCED MODEL (accDiff + rtDiff) ===")
print(res_red.summary())
print(f"LL(red)={res_red.llf:.6f} k={res_red.params.size} BIC(red)={BIC_red:.4f}")

print("\n=== COMPARISON ===")
print(f"ΔBIC (red - full) = {BIC_red - BIC_full:.4f}  (positive => full preferred)")

beta0 = res_full.params[0] #intercept check
p_baseline = 1 / (1 + np.exp(-beta0))
print("Baseline probability from intercept:", round(p_baseline, 3))
print("Empirical proportion of choosing task 2:", round(Y.mean(), 3))