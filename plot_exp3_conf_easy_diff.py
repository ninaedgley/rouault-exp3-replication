from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def mat_to_py(obj): 
    """Recursively convert scipy mat_struct / nested arrays into python dicts/arrays."""
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array([mat_to_py(x) for x in obj],dtype=object)
    
    #mat_struct-like: has _fieldnames
    if hasattr(obj, "_fieldnames"):
        return {f: mat_to_py(getattr(obj, f)) for f in obj._fieldnames}
    
    return obj

# --------- paths ----------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "DATA" / "Exp3.mat"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --------- load -----------
mat = loadmat(DATA_PATH.as_posix(), squeeze_me=True, struct_as_record=False)
Exp3 = mat_to_py(mat["Exp3"])

conf = np.array(Exp3["confidence_level"], dtype=float)
pooled = np.array(Exp3.get("confidence_level_pooled"), dtype=float) if "confidence_level_pooled" in Exp3 else None

if pooled is None or pooled.size == 0:
    raise ValueError("confidence_level_pooled not found or empty in Exp3. Tell me its shape and we'll adapt.")

if pooled.ndim == 2 and 2 in pooled.shape:
    if pooled.shape[1] == 2:
        easy = pooled[:, 0]
        diff = pooled[:,1]
    else:
        easy = pooled[0, :]
        diff = pooled[1, :]
    
else:
    raise ValueError(f"Unexpected confidence_level_pooled shape: {pooled.shape}")

means = [np.nanmean(easy), np.nanmean(diff)]
sems = [np.nanstd(easy, ddof=1) / np.sqrt(np.sum(~np.isnan(easy))),
        np.nanstd(diff, ddof=1) / np.sqrt(np.sum(~np.isnan(diff)))]

plt.figure()
plt.bar(["Easy", "Difficult"], means, yerr=sems)
plt.ylabel("Mean confidence")
plt.title("Exp3: Confidence by Difficulty (mean ± SEM )")

out = FIG_DIR / "exp3_confidence_easy_vs_difficult.png"
plt.tight_layout()
plt.savefig(out.as_posix(), dpi=200)
print(f"Saved: {out}")
