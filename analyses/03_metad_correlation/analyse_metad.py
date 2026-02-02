"""
Meta-d' Analysis: Rouault et al. (2019) Experiment 3
Uses pre-computed hierarchical Bayesian meta-d' estimates from original analysis
"""
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Load data
mat = sio.loadmat("/Users/ninaedgley/Research/RouaultDayanFleming-master/DATA/Exp3.mat", 
                  squeeze_me=True, struct_as_record=False)
Exp3 = mat["Exp3"]

# Extract pre-computed M-ratios (hierarchical Bayesian estimates from paper)
mratios = np.array(Exp3.mratios, dtype=float)
M_ratio_easy = mratios[:, 0]
M_ratio_diff = mratios[:, 1]
M_ratio_avg = (M_ratio_easy + M_ratio_diff) / 2

n_subjects = len(M_ratio_easy)

# Extract task choice data
T2chperser = Exp3.T2chperser
T1chperser = Exp3.T1chperser
delta_task_choice = T2chperser[:, 5] - T1chperser[:, 5]  # Pairing 6: No-feedback conditions

# Descriptive statistics
print("\n" + "="*70)
print("Metacognitive Efficiency (M-ratio)")
print("="*70)
print(f"N = {n_subjects}")
print(f"\nEasy condition:      M = {np.mean(M_ratio_easy):.3f}, SD = {np.std(M_ratio_easy):.3f}")
print(f"Difficult condition: M = {np.mean(M_ratio_diff):.3f}, SD = {np.std(M_ratio_diff):.3f}")
print(f"Average:             M = {np.mean(M_ratio_avg):.3f}, SD = {np.std(M_ratio_avg):.3f}")

# Key analysis: Relationship between metacognition and global self-performance
print("\n" + "="*70)
print("Metacognition and Global Self-Performance Estimates")
print("="*70)

r_pearson, p_pearson = pearsonr(delta_task_choice, M_ratio_avg)
r_spearman, p_spearman = spearmanr(delta_task_choice, M_ratio_avg)

print(f"Pearson correlation:  r = {r_pearson:.3f}, p = {p_pearson:.4f}")
print(f"Spearman correlation: ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

if p_spearman < 0.05:
    sig_level = "**" if p_spearman < 0.01 else "*"
    print(f"Result: Significant {sig_level} (p < {0.01 if p_spearman < 0.01 else 0.05})")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: M-ratio distributions
ax1 = axes[0]
bins = np.linspace(0, 2.5, 20)
ax1.hist(M_ratio_easy, bins=bins, alpha=0.6, label='Easy', color='#009933', edgecolor='black')
ax1.hist(M_ratio_diff, bins=bins, alpha=0.6, label='Difficult', color='#FF9915', edgecolor='black')
ax1.axvline(np.mean(M_ratio_easy), color='#009933', linestyle='--', linewidth=2)
ax1.axvline(np.mean(M_ratio_diff), color='#FF9915', linestyle='--', linewidth=2)
ax1.set_xlabel('M-ratio', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Metacognitive Efficiency Distribution', fontsize=13)
ax1.legend(fontsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Correlation (replicates Figure 5d from paper)
ax2 = axes[1]
ax2.scatter(delta_task_choice, M_ratio_avg, s=80, alpha=0.6, 
           color='#8B008B', edgecolors='black', linewidth=0.5)

# Regression line
z = np.polyfit(delta_task_choice, M_ratio_avg, 1)
p_line = np.poly1d(z)
x_line = np.linspace(delta_task_choice.min(), delta_task_choice.max(), 100)
ax2.plot(x_line, p_line(x_line), 'r--', linewidth=2, alpha=0.8)

ax2.set_xlabel('Task Choice Difference\n(Easy - Difficult)', fontsize=12)
ax2.set_ylabel('M-ratio (Average)', fontsize=12)
ax2.set_title(f'Metacognition Predicts Global SPE\nρ = {r_spearman:.3f}, p = {p_spearman:.4f}', 
             fontsize=13)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/ninaedgley/Research/RouaultDayanFleming-master/metad_analysis.png', 
            dpi=300, bbox_inches='tight')

print(f"\nFigure saved: metad_analysis.png")
print("="*70 + "\n")