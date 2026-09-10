# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ============================================================
# 1. Parameters
# ============================================================

N = 10
t1 = 1
Lmbda = 1
U = 6

sz = N * N


# ============================================================
# 2. Layer-swap operator
#    Layer A <-> Layer B
# ============================================================

Sl_op = np.zeros((2 * sz, 2 * sz), dtype=complex)

for i in range(sz):
    Sl_op[i, sz + i] = 1.0
    Sl_op[sz + i, i] = 1.0


# ============================================================
# 3. Build two-layer Hamiltonian
# ============================================================

def build_H_2Layer(N, t1, U, Lmbda):

    sz = N * N
    H = np.zeros((2 * sz, 2 * sz), dtype=complex)

    for x in range(N):
        for y in range(N):

            i = x * N + y

            # ------------------------------------------------
            # Intralayer hopping + on-site interaction
            # ------------------------------------------------
            for offset in (0, sz):

                if x > 0:
                    H[
                        offset + i,
                        offset + (x - 1) * N + y
                    ] = -t1

                if x < N - 1:
                    H[
                        offset + i,
                        offset + (x + 1) * N + y
                    ] = -t1

                if y > 0:
                    H[
                        offset + i,
                        offset + x * N + (y - 1)
                    ] = -t1

                if y < N - 1:
                    H[
                        offset + i,
                        offset + x * N + (y + 1)
                    ] = -t1

                # Interaction along x = y
                if x == y:
                    H[
                        offset + i,
                        offset + i
                    ] += U

            # ------------------------------------------------
            # Cross-layer SOC coupling
            # ------------------------------------------------
            neighbours = [
                (1,  0,  1),
                (-1, 0, -1),
                (0,  1,  1),
                (0, -1, -1)
            ]

            for dx, dy, s in neighbours:

                nx = x + dx
                ny = y + dy

                if 0 <= nx < N and 0 <= ny < N:

                    j = nx * N + ny

                    H[i, sz + j] = Lmbda
                    H[sz + j, i] = Lmbda

                    H[sz + i, j] = Lmbda
                    H[j, sz + i] = Lmbda

    return H


# ============================================================
# 4. Solve Hamiltonian
# ============================================================

H = build_H_2Layer(
    N,
    t1,
    U,
    Lmbda
)

E, V = eigh(H.real)


# ============================================================
# 5. Calculate IPR
# ============================================================

IPR = np.sum(
    np.abs(V) ** 4,
    axis=0
)


# ============================================================
# 6. Sort states by energy
# ============================================================

idx = np.argsort(E.real)

E = E[idx]
IPR = IPR[idx]
V = V[:, idx]


# ============================================================
# 7. Select high-IPR states with E > 5
# ============================================================

valid_indices = np.where(
    E.real > 5
)[0]

# Sort IPR in descending order
sorted_relative_idx = np.argsort(
    IPR[valid_indices]
)[::-1]

descending_indices = valid_indices[
    sorted_relative_idx
]


# Highest IPR state
bpic_idx = descending_indices[0]

# 19th highest IPR state
bpic_idx2 = descending_indices[18]


# ============================================================
# 8. Print selected-state information
# ============================================================

print("Selected state 1:")
print("index =", bpic_idx)
print("E =", E[bpic_idx])
print("IPR =", IPR[bpic_idx])

print()

print("Selected state 2:")
print("index =", bpic_idx2)
print("E =", E[bpic_idx2])
print("IPR =", IPR[bpic_idx2])

print()

print(
    "Overlap =",
    np.dot(
        V[:, bpic_idx],
        V[:, bpic_idx2]
    )
)


# ============================================================
# 9. Check layer-exchange symmetry
# ============================================================

commutator = (
    Sl_op @ H
    - H @ Sl_op
)

print()

print(
    "||Sl H - H Sl|| =",
    np.linalg.norm(commutator)
)

print(
    "max|Sl H - H Sl| =",
    np.max(
        np.abs(commutator)
    )
)


# ============================================================
# 10. Plot style
# ============================================================

plt.rcParams.update({

    'font.family': 'sans-serif',

    'font.sans-serif': [
        'Arial',
        'Helvetica',
        'DejaVu Sans'
    ],

    'font.size': 9,

    'axes.labelsize': 9,
    'axes.titlesize': 9,

    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

    'legend.fontsize': 8,

    'axes.linewidth': 0.8,

    'axes.spines.top': False,
    'axes.spines.right': False,

    'xtick.direction': 'in',
    'ytick.direction': 'in',

    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    'xtick.major.size': 4,
    'ytick.major.size': 4,

    'xtick.minor.size': 2,
    'ytick.minor.size': 2,

    'grid.color': '0.7',
    'grid.linestyle': ':',
    'grid.alpha': 0.5,

    'figure.dpi': 100
})


# ============================================================
# Figure 1
# E - IPR spectrum colored by parity
# ============================================================

parity = np.sum(
    np.conj(V) * (Sl_op @ V),
    axis=0
).real


fig, ax = plt.subplots(
    figsize=(3.5, 3.5)
)


# ------------------------------------------------------------
# All eigenstates
# ------------------------------------------------------------

sc = ax.scatter(
    E,
    IPR,
    c=parity,
    cmap='bwr',
    vmin=-1,
    vmax=1,
    alpha=0.7,
    edgecolor='none',
    s=40
)


# ------------------------------------------------------------
# Highlight state 1
# ------------------------------------------------------------

ax.scatter(
    E[bpic_idx].real,
    IPR[bpic_idx],
    marker='H',
    s=95,
    facecolor='none',
    edgecolors='r',
    alpha=0.9
)


# ------------------------------------------------------------
# Highlight state 2
# ------------------------------------------------------------

ax.scatter(
    E[bpic_idx2].real,
    IPR[bpic_idx2],
    marker='H',
    s=95,
    facecolor='none',
    edgecolors='b',
    alpha=0.9
)


ax.set_xlabel('E')
ax.set_ylabel('IPR')

ax.grid(
    True,
    linestyle=':',
    alpha=0.5
)

plt.tight_layout(
    pad=0.5
)

plt.show()


# ============================================================
# Figure 2
# Wave functions of selected states
# ============================================================


# ============================================================
# 11. State 1
#     symmetric supermode
# ============================================================

V_raw = V[:, bpic_idx]

V_sym = (
    V_raw
    + Sl_op @ V_raw
)


# Normalize
V_sym = (
    V_sym
    / np.linalg.norm(V_sym)
)


# Layer A
psi_A = np.real(
    V_sym[:sz]
).reshape(
    N,
    N
)


# Layer B
psi_B = np.real(
    V_sym[sz:]
).reshape(
    N,
    N
)


# ============================================================
# 12. State 2
#     antisymmetric supermode
# ============================================================

V_raw2 = V[:, bpic_idx2]

V_sym2 = (
    V_raw2
    - Sl_op @ V_raw2
)


if np.linalg.norm(V_sym2) > 1e-10:

    V_sym2 = (
        V_sym2
        / np.linalg.norm(V_sym2)
    )


# Layer A
psi_C = np.real(
    V_sym2[:sz]
).reshape(
    N,
    N
)


# Layer B
psi_D = np.real(
    V_sym2[sz:]
).reshape(
    N,
    N
)


# ============================================================
# 13. Plot spatial profiles
# ============================================================

fig = plt.figure(
    figsize=(7.2, 6)
)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.3,
    wspace=0.3
)


ax_psiA = fig.add_subplot(
    gs[0, 0]
)

ax_psiB = fig.add_subplot(
    gs[0, 1]
)

ax_psiC = fig.add_subplot(
    gs[1, 0]
)

ax_psiD = fig.add_subplot(
    gs[1, 1]
)


# ============================================================
# State 1 common color scale
# ============================================================

vmax = max(
    np.max(
        np.abs(psi_A)
    ),
    np.max(
        np.abs(psi_B)
    )
)


im1 = ax_psiA.imshow(
    psi_A,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax,
    vmax=vmax
)

ax_psiA.set_title(
    r'Layer A'
)

fig.colorbar(
    im1,
    ax=ax_psiA,
    fraction=0.046,
    pad=0.04
)


im2 = ax_psiB.imshow(
    psi_B,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax,
    vmax=vmax
)

ax_psiB.set_title(
    r'Layer B'
)

fig.colorbar(
    im2,
    ax=ax_psiB,
    fraction=0.046,
    pad=0.04
)


# ============================================================
# State 2 common color scale
# ============================================================

vmax2 = max(
    np.max(
        np.abs(psi_C)
    ),
    np.max(
        np.abs(psi_D)
    )
)


im3 = ax_psiC.imshow(
    psi_C,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax2,
    vmax=vmax2
)

ax_psiC.set_title(
    r'Layer A'
)

fig.colorbar(
    im3,
    ax=ax_psiC,
    fraction=0.046,
    pad=0.04
)


im4 = ax_psiD.imshow(
    psi_D,
    cmap='RdBu',
    origin='lower',
    vmin=-vmax2,
    vmax=vmax2
)

ax_psiD.set_title(
    r'Layer B'
)

fig.colorbar(
    im4,
    ax=ax_psiD,
    fraction=0.046,
    pad=0.04
)


# ============================================================
# Remove ticks
# ============================================================

for ax in [
    ax_psiA,
    ax_psiB,
    ax_psiC,
    ax_psiD
]:

    ax.set_xticks([])
    ax.set_yticks([])


plt.show()