import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paper Parameters (From Table III setup) ---
t = 50.0       # Total simulation time
eps = 0.0001   # Target error precision
J = 1.0        # Interaction strength
d = 4.0        # Lattice degree (e.g., 2D grid)
Delta = 10.0   # Low-energy bound

# Scale system size N from 10 to 10,000 spins
N_values = np.logspace(1, 4, 20)

# --- 2. Calculate Asymptotic Complexity (Trotter Number r) ---
# Formula 1: Best-known Worst-Case Complexity (p=1)
# O(t * (t/eps) * d^2 * N * J^2)
r_worst = t * (t / eps) * (d**2) * N_values * (J**2)

# Formula 2: Low-Energy Complexity (p=1)
# O(t * (t/eps) * (d*Delta + d^3*J)^2) + O(t * (t/eps)^(1/3) * d^2 * N^(2/3) * J^(4/3))
term1 = t * (t / eps) * (d * Delta + (d**3) * J)**2
term2 = t * ((t / eps)**(1/3)) * (d**2) * (N_values**(2/3)) * (J**(4/3))
r_low = term1 + term2

# Calculate the computational savings multiplier
savings_ratio = r_worst / r_low

# --- 3. The Dual-Axis Plot ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Left Y-Axis: Trotter Number (Orange)
color_cost = 'tab:orange'
ax1.set_xlabel('System Size (Number of Spins $N$) - Log Scale', fontsize=12, fontweight='bold')
ax1.set_ylabel('Required Trotter Steps ($r$) - Log Scale', color=color_cost, fontsize=12, fontweight='bold')

# Plot Worst-Case (Dashed)
ax1.loglog(N_values, r_worst, 's--', color=color_cost, linewidth=3, markersize=10, label='Worst-Case Complexity $\mathcal{O}(N)$')
# Plot Low-Energy (Solid)
ax1.loglog(N_values, r_low, 'o-', color=color_cost, linewidth=3, markersize=10, label='Low-Energy Complexity $\mathcal{O}(N^{2/3})$')

ax1.tick_params(axis='y', labelcolor=color_cost)

# Right Y-Axis: Savings Ratio (Blue)
ax2 = ax1.twinx()  
color_ratio = 'tab:blue'
ax2.set_ylabel('Computational Savings Ratio (Worst / Low-Energy)', color=color_ratio, fontsize=12, fontweight='bold')

# Plot the Savings Ratio (Dashed Diamonds)
ax2.loglog(N_values, savings_ratio, 'd--', color=color_ratio, linewidth=3, markersize=10, label='Gate Savings Multiplier')

ax2.tick_params(axis='y', labelcolor=color_ratio)

# Combine legends beautifully
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0.02, 0.95), framealpha=0.9)

# Draw the Crossover Point
cross_indices = np.where(savings_ratio > 1.0)[0]
if len(cross_indices) > 0:
    c_idx = cross_indices[0]
    N_cross = N_values[c_idx]
    ax1.axvline(x=N_cross, color='black', linestyle=':', linewidth=2)
    ax1.text(N_cross * 1.1, r_low[c_idx] * 0.2, f'The Crossover Point\n(N ≈ {int(N_cross)})', 
             fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('Table III: Theoretical Asymptotic Crossover ($p=1$)\nProjecting the Algorithm to Massive Quantum Computers', fontsize=14)
plt.grid(True, which="both", linestyle=':', alpha=0.5)
fig.tight_layout()
plt.show()