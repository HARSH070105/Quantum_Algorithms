import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

Nx, Ny = 2, 6       # 2x6 Spin Ladder
N = Nx * Ny         # 12 Spins
dim = 2**N          # 4096 x 4096 Matrix

print(f"Initializing a {dim}x{dim} system for the 2x6 Ladder...")

I = sp.eye(2, format='csr', dtype=complex)
X = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

def get_interaction_sparse(site1, site2, op):
    ops = [I] * N
    ops[site1] = op
    ops[site2] = op
    term = ops[0]
    for i in range(1, N):
        term = sp.kron(term, ops[i], format='csr')
    return term

edges = []
# Horizontal edges (intra-leg bonds)
for x in range(Nx):
    for y in range(Ny - 1):
        edges.append((x * Ny + y, x * Ny + y + 1))
# Vertical edges (rungs)
for y in range(Ny):
    edges.append((0 * Ny + y, 1 * Ny + y))

H1_sparse = sp.csr_matrix((dim, dim), dtype=complex)
H2_sparse = sp.csr_matrix((dim, dim), dtype=complex)

for i, (u, v) in enumerate(edges):
    # Ferromagnetic Heisenberg Model: H = - (XX + YY + ZZ)
    term = -(get_interaction_sparse(u, v, X) + 
             get_interaction_sparse(u, v, Y) + 
             get_interaction_sparse(u, v, Z))
    
    if i % 2 == 0:
        H1_sparse += term
    else:
        H2_sparse += term

H_sparse = H1_sparse + H2_sparse

print("\nTargeting the lowest 200 energy states (this takes ~5-15 seconds)...")
t0 = time.time()
evals, evecs = spla.eigsh(H_sparse, k=200, which='SA')
idx = np.argsort(evals)
evecs = evecs[:, idx]
print(f" -> States extracted in {time.time() - t0:.2f} seconds.")

s_values = np.linspace(0, 0.02, 10) 
n_values = [1, 50, 100, 150, 200]

err_W1 = {n: [] for n in n_values}; err_W1['worst'] = []
err_W2 = {n: [] for n in n_values}; err_W2['worst'] = []

print("\nConverting sparse matrices to dense for the Time Machine (expm)...")
H_dense = H_sparse.toarray()
H1_dense = H1_sparse.toarray()
H2_dense = H2_sparse.toarray()

print("\nSimulating Time Evolution Sequence (This will take a few minutes)...")
for i, s in enumerate(s_values):
    t_step = time.time()
    
    U = la.expm(-1j * s * H_dense)
    exp_H1 = la.expm(-1j * s * H1_dense)
    exp_H2 = la.expm(-1j * s * H2_dense)
    exp_H1_half = la.expm(-0.5j * s * H1_dense)
    
    W1 = exp_H1 @ exp_H2
    W2 = exp_H1_half @ exp_H2 @ exp_H1_half
    
    diff_W1 = U - W1
    diff_W2 = U - W2

    err_W1['worst'].append(la.norm(diff_W1, ord=2))
    err_W2['worst'].append(la.norm(diff_W2, ord=2))
    
    for n in n_values:
        V_n = evecs[:, :n]
        err_W1[n].append(la.norm(diff_W1 @ V_n, ord=2))
        err_W2[n].append(la.norm(diff_W2 @ V_n, ord=2))
        
    print(f" -> Step {i+1}/10 (s = {s:.4f}) finished in {time.time()-t_step:.2f}s")

print("\nGenerating Figure 1...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']

for idx, n in enumerate(n_values):
    ax1.plot(s_values, err_W1[n], label=f'n = {n}', color=colors[idx], linewidth=2)
ax1.plot(s_values, err_W1['worst'], 'k--', label='worst case', linewidth=2.5)
ax1.set_title('1st Order Trotter Error ($W_1(s)$) - $2\\times6$ Ladder')
ax1.set_ylabel('Error')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.7)

for idx, n in enumerate(n_values):
    ax2.plot(s_values, err_W2[n], label=f'n = {n}', color=colors[idx], linewidth=2)
ax2.plot(s_values, err_W2['worst'], 'k--', label='worst case', linewidth=2.5)
ax2.set_title('2nd Order Trotter Error ($W_2(s)$) - $2\\times6$ Ladder')
ax2.set_xlabel('Time Step ($s$)')
ax2.set_ylabel('Error')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()