"""
Exp 3 -- Subject-level confidence means
Goal: Compare Easy vs Difficult confidence (per subject)
"""

import numpy as np
import scipy.io as sio
import os

MAT_PATH = "DATA/Exp3.mat"
mat = sio.loadmat(MAT_PATH, squeeze_me = True, struct_as_record = False) 
Exp3 = mat["Exp3"]

cl = Exp3.confidence_level
cl_p = Exp3.confidence_level_pooled

print("confidence_level shape:", np.array(cl).shape)
print("confidence_level_pooled shape:", np.array(cl_p).shape)
print("row0 pooled:", np.array(cl_p)[0])

easy = cl_p[:,0]
diff = cl_p[:,1]

print("easy shape:", easy.shape)
print("diff shape:", diff.shape)
print("Easy first 5 values:", easy[:5])
print("Diff first 5 values:", diff[:5])
print("Easy mean:",easy.mean())
print("Diff mean:", diff.mean())

import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR,"exp3_two_panel_confidence_subject_means.png")

delta = easy - diff

fig, axes = plt.subplots(1,2,figsize=(10,4), sharex=True, sharey=True)

bins = 12

#LEFT : Easy
axes[0].hist(easy, bins=bins)
axes[0].set_title("Easy : subject mean confidence")

#RIGHT : Difficult
axes[1].hist(diff, bins=bins)
axes[1].set_title("Difficult : subject mean confidence")

#Same x limit
xmin = min(easy.min(),diff.min())
xmax = max(easy.max(),diff.max())

for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Mean confidence")
axes[0].set_ylabel("Number of subjects")

fig.suptitle(f"Exp3 subject means: Easy vs Difficult | mean(delta)={delta.mean():.3f}")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print("Saved:", OUT_PATH)
