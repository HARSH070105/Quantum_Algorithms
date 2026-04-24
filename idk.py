import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

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

N = 8
n = 2**N

H1 = np.zeros((n, n), dtype=complex)
H2 = np.zeros((n, n), dtype=complex)

# 1D Heisenberg model: H = - (XX + YY + ZZ)
for i in range(N - 1):
    interaction = - (get_interaction_term(N, i, i+1, X, X) + 
                     get_interaction_term(N, i, i+1, Y, Y) + 
                     get_interaction_term(N, i, i+1, Z, Z))
    
    # Split into H1 (even bonds) and H2 (odd bonds)
    if i % 2 == 0:
        H1 += interaction
    else:
        H2 += interaction

H = H1 + H2

print("Diagonalizing Hamiltonian...")
evals, evecs = la.eigh(H)
idx = np.argsort(evals)
evals = evals[idx]
evecs = evecs[:, idx]

s_values = np.linspace(0, 0.02, 20) 
n_values = [1, 50, 100, 150, 200]  # Low-energy subspace sizes

# Dictionary to store errors
err_W1 = {n: [] for n in n_values}; err_W1['worst'] = []
err_W2 = {n: [] for n in n_values}; err_W2['worst'] = []
err_W4 = {n: [] for n in n_values}; err_W4['worst'] = []

p1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
p0 = 1.0 - 4.0 * p1

def get_W2(time_step):
    return la.expm(-0.5j * time_step * H1) @ la.expm(-1j * time_step * H2) @ la.expm(-0.5j * time_step * H1)

print("Simulating Trotter Time Evolutions (1st, 2nd, and 4th Order)...")
for s in s_values:
    
    U = la.expm(-1j * s * H)
    
    W1 = la.expm(-1j * s * H1) @ la.expm(-1j * s * H2)
    W2 = get_W2(s)
    W4 = get_W2(p1 * s) @ get_W2(p1 * s) @ get_W2(p0 * s) @ get_W2(p1 * s) @ get_W2(p1 * s)
    
    diff_W1 = U - W1
    diff_W2 = U - W2
    diff_W4 = U - W4
    
    # Global worst-case error (spectral norm)
    err_W1['worst'].append(la.norm(diff_W1, ord=2))
    err_W2['worst'].append(la.norm(diff_W2, ord=2))
    err_W4['worst'].append(la.norm(diff_W4, ord=2))
    
    # Low-energy subspace errors
    for n in n_values:
        V_n = evecs[:, :n] # Restrict to lowest 'n' eigenvectors
        err_W1[n].append(la.norm(diff_W1 @ V_n, ord=2))
        err_W2[n].append(la.norm(diff_W2 @ V_n, ord=2))
        err_W4[n].append(la.norm(diff_W4 @ V_n, ord=2))


# print("Simulating Energy Leakage (Lemma 1)...")
# s_leak = 0.01 
# n_low = 5 
# Pi_low = evecs[:, :n_low] @ evecs[:, :n_low].conj().T

# leakage_errors = []
# gap_sizes = range(n_low + 1, n, 10) 
# U_H1 = la.expm(-1j * s_leak * H1)

# for n_high in gap_sizes:
#     Pi_high = evecs[:, n_high:] @ evecs[:, n_high:].conj().T
#     leakage = la.norm(Pi_high @ U_H1 @ Pi_low, ord=2)
#     leakage_errors.append(leakage)


fig, axs = plt.subplots(3, 1, figsize=(14, 10))

# Subplot 1: 1st Order W1
for n in n_values:
    axs[0].plot(s_values, err_W1[n], label=f'n = {n}')
axs[0].plot(s_values, err_W1['worst'], 'k--', label='worst case', linewidth=2)
axs[0].set_title('1st Order Trotter Error ($W_1(s)$)')
axs[0].set_ylabel('Spectral Norm Error')
axs[0].legend(); axs[0].grid(True, linestyle=':', alpha=0.7)

# Subplot 2: 2nd Order W2
for n in n_values:
    axs[1].plot(s_values, err_W2[n], label=f'n = {n}')
axs[1].plot(s_values, err_W2['worst'], 'k--', label='worst case', linewidth=2)
axs[1].set_title('2nd Order Trotter Error ($W_2(s)$)')
axs[1].legend(); axs[1].grid(True, linestyle=':', alpha=0.7)

# Subplot 3: 4th Order W4
for n in n_values:
    axs[2].plot(s_values, err_W4[n], label=f'n = {n}')
axs[2].plot(s_values, err_W4['worst'], 'k--', label='worst case', linewidth=2)
axs[2].set_title('4th Order Trotter Error ($W_4(s)$)')
axs[2].set_xlabel('Time Step (s)')
axs[2].set_ylabel('Spectral Norm Error')
axs[2].legend(); axs[2].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
