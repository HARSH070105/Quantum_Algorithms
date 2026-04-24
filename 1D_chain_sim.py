import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

# --- Pauli Matrices ---
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_interaction_term(N, site1, site2, op1, op2):
    term = np.array([1])
    for i in range(N):
        if i == site1:
            term = np.kron(term, op1)
        elif i == site2:
            term = np.kron(term, op2)
        else:
            term = np.kron(term, I)
    return term

# --- System Definition ---
N = 8
n_dim = 2**N
H1 = np.zeros((n_dim, n_dim), dtype=complex)
H2 = np.zeros((n_dim, n_dim), dtype=complex)

print(f"Building Hamiltonian for 1D {N}-spin chain...")
for i in range(N - 1):
    interaction = - (get_interaction_term(N, i, i+1, X, X) + 
                     get_interaction_term(N, i, i+1, Y, Y) + 
                     get_interaction_term(N, i, i+1, Z, Z))
    
    if i % 2 == 0:
        H1 += interaction
    else:
        H2 += interaction

H = H1 + H2

print("Diagonalizing Hamiltonian...")
evals, evecs = la.eigh(H)
idx = np.argsort(evals)
evecs = evecs[:, idx]

# --- Simulation Parameters ---
s_values = np.linspace(0, 0.02, 20) 
n_values = [1, 10, 50, 100, 150, 200]

err_W1 = {n: [] for n in n_values}; err_W1['worst'] = []
err_W2 = {n: [] for n in n_values}; err_W2['worst'] = []
err_W4 = {n: [] for n in n_values}; err_W4['worst'] = []

p1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
p0 = 1.0 - 4.0 * p1

def get_W2(time_step):
    return la.expm(-0.5j * time_step * H1) @ la.expm(-1j * time_step * H2) @ la.expm(-0.5j * time_step * H1)

print("Simulating Time Evolutions...")
for s in s_values:
    U = la.expm(-1j * s * H)
    W1 = la.expm(-1j * s * H1) @ la.expm(-1j * s * H2)
    W2 = get_W2(s)
    W4 = get_W2(p1 * s) @ get_W2(p1 * s) @ get_W2(p0 * s) @ get_W2(p1 * s) @ get_W2(p1 * s)
    
    diff_W1, diff_W2, diff_W4 = U - W1, U - W2, U - W4
    
    # Global worst-case error
    err_W1['worst'].append(la.norm(diff_W1, ord=2))
    err_W2['worst'].append(la.norm(diff_W2, ord=2))
    err_W4['worst'].append(la.norm(diff_W4, ord=2))
    
    # Low-energy subspace errors
    for n in n_values:
        V_n = evecs[:, :n]
        err_W1[n].append(la.norm(diff_W1 @ V_n, ord=2))
        err_W2[n].append(la.norm(diff_W2 @ V_n, ord=2))
        err_W4[n].append(la.norm(diff_W4 @ V_n, ord=2))

# --- Plotting ---
print("Generating and saving plots...")
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink']

for ax, err_dict, title in zip(axs, [err_W1, err_W2, err_W4], ['1st Order ($W_1$)', '2nd Order ($W_2$)', '4th Order ($W_4$)']):
    for idx, n in enumerate(n_values):
        ax.plot(s_values, err_dict[n], label=f'$n = {n}$', color=colors[idx])
    ax.plot(s_values, err_dict['worst'], 'k--', label='Worst Case', linewidth=2.5)
    ax.set_title(f'{title} Error - 1D Chain')
    ax.set_ylabel('Spectral Norm Error')
    ax.grid(True, linestyle='--', alpha=0.6)
    if ax == axs[0]: ax.legend(loc='upper left')

axs[2].set_xlabel('Time Step ($s$)')
plt.tight_layout()
plt.savefig('images/1d_chain_errors.png', dpi=300, bbox_inches='tight')
print("Saved to images/1d_chain_errors.png")