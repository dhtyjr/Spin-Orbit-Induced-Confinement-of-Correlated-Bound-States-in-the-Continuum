# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": False,
    "ytick.right": False,

    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,

    "legend.frameon": False,
    "legend.fontsize": 8,

    # =========================
    # Editable vector text export
    # =========================
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.unicode_minus": False
})


# =========================
# 2. Physics Parameters
# =========================
K = np.linspace(-np.pi, np.pi, 1000)

U = 3.0
t1 = 1.3
soc = 1.0


def E_plus(K, U, t1, coupling):
    return np.sqrt(U**2 + 16 * (t1 - coupling)**2 * np.cos(K/2)**2)


def E_minus(K, U, t1, coupling):
    return np.sqrt(U**2 + 16 * (t1 + coupling)**2 * np.cos(K/2)**2)



# =========================
# 3. Figure Setup
# =========================
fig, ax = plt.subplots(
    figsize=(3.5, 2.8),
    dpi=300
)


# Remove top/right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# =========================
# 4. Plot bands
# =========================

# Degenerate case
E_deg = E_plus(K, U, t1, 0)

ax.plot(
    K,
    E_deg,
    color='black',
    linestyle='--',
    linewidth=1.2,
    label='Without SOC',
    zorder=2
)


# SOC split case
E_p = E_plus(K, U, t1, soc)
E_m = E_minus(K, U, t1, soc)


ax.plot(
    K,
    E_p,
    color='#D81B60',
    label=r'$E_{\mathrm{doublon}}^{+}(K)$',
    zorder=3
)


ax.plot(
    K,
    E_m,
    color='#1E88E5',
    label=r'$E_{\mathrm{doublon}}^{-}(K)$',
    zorder=3
)


# Highlight splitting region
ax.fill_between(
    K,
    E_p,
    E_m,
    color='gray',
    alpha=0.15,
    zorder=1
)



# =========================
# 5. Splitting arrows at K=0
# =========================

E_deg_0 = E_plus(0, U, t1, 0)
E_p_0 = E_plus(0, U, t1, soc)
E_m_0 = E_minus(0, U, t1, soc)


ax.annotate(
    '',
    xy=(0, E_m_0 - 0.2),
    xytext=(0, E_deg_0 + 0.2),
    arrowprops=dict(
        arrowstyle='->',
        color='#1E88E5',
        lw=1.5,
        mutation_scale=12
    )
)


ax.annotate(
    '',
    xy=(0, E_p_0 + 0.2),
    xytext=(0, E_deg_0 - 0.2),
    arrowprops=dict(
        arrowstyle='->',
        color='#D81B60',
        lw=1.5,
        mutation_scale=12
    )
)



# Text labels

ax.text(
    0.15,
    (E_deg_0 + E_m_0)/2,
    'SOC',
    color='#1E88E5',
    va='center',
    fontsize=8
)


ax.text(
    0.15,
    (E_deg_0 + E_p_0)/2,
    'SOC',
    color='#D81B60',
    va='center',
    fontsize=8
)



# =========================
# 6. Axis formatting
# =========================

ax.set_xlabel(r'Center-of-mass momentum $K$')
ax.set_ylabel(r'$E(K)$')


ax.set_xlim(
    -np.pi,
    np.pi
)


ax.set_xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
)


ax.set_xticklabels(
    [
        r'$-\pi$',
        r'$-\pi/2$',
        r'$0$',
        r'$\pi/2$',
        r'$\pi$'
    ]
)


ax.yaxis.set_minor_locator(
    MultipleLocator(0.5)
)



# =========================
# 7. Legend
# =========================

ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0)
)


# remove tick labels if needed
ax.tick_params(
    labelbottom=False,
    labelleft=False
)



plt.tight_layout()
plt.show()