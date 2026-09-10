# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch



WATERFALL_CURVE_COLORS = [
    "#D71C2C",  
    "#FEAB88",  
    "#6FAFD2",  
    "#327DB7",  
]


plt.rcParams.update({
    "font.family": "Arial",

    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",

    "font.size": 11.5,
    "axes.labelsize": 15.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 12.5,

    "axes.linewidth": 1.35,
    "axes.spines.top": True,
    "axes.spines.right": True,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,

    "axes.unicode_minus": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.facecolor": "white",
    "axes.facecolor": "white",
})





# ==========================================
# 2. System Parameters & Hamiltonian Setup
# ==========================================
N = 10
t1 = 1


def build_H_2Layer(N, t1, U, Lmbda, add_theta=False):
    sz = N * N
    H = np.zeros((2 * sz, 2 * sz), dtype=complex)

    for x in range(N):
        for y in range(N):
            i = x * N + y

            # ----- 1. Kinetic (-t1) and on-site U for both layers -----
            for offset in (0, sz):
                if x > 0:
                    H[offset + i, offset + (x - 1) * N + y] = -t1
                if x < N - 1:
                    H[offset + i, offset + (x + 1) * N + y] = -t1
                if y > 0:
                    H[offset + i, offset + x * N + (y - 1)] = -t1
                if y < N - 1:
                    H[offset + i, offset + x * N + (y + 1)] = -t1
                if x == y:
                    H[offset + i, offset + i] += U

            # ----- 2. Spin-orbit coupling (cross-layer terms) -----
            neighbours = [
                (1, 0, 1),
                (-1, 0, -1),
                (0, 1, 1),
                (0, -1, -1),
            ]

            for dx, dy, s in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N:
                    j = nx * N + ny

                    H[i, sz + j] = Lmbda
                    H[sz + j, i] = Lmbda
                    H[sz + i, j] = Lmbda
                    H[j, sz + i] = Lmbda

    if add_theta:
        theta = 0.0
        for i in range(sz):
            H[i, i + sz] = theta

    return H


# ==========================================
# 3. Calculate the data used in A, B, C
#    Lmbda = 0, U = 9.82
#
# ==========================================
def calculate_top_row():
    Lmbda = 2.87 * 0
    U = 9.82
    sz = N * N

    H = build_H_2Layer(N, t1, U, Lmbda, add_theta=True)
    E, V = eigh(H.real)
    IPR = np.sum(np.abs(V) ** 4, axis=0)

    idx = np.argsort(E.real)
    E, IPR, V = E[idx], IPR[idx], V[:, idx]

    descending_indices = np.argsort(IPR)[::-1]
    bpic_idx = descending_indices[0]
    bpic_idx2 = descending_indices[19]

    print(E[bpic_idx])

    psi_A = np.real(V[:sz, bpic_idx]).reshape(N, N)
    psi_B = np.real(V[sz:, bpic_idx + 1]).reshape(N, N)

    psi_C = np.real(V[:sz, bpic_idx2]).reshape(N, N)
    psi_D = np.real(V[sz:, bpic_idx2 - 1]).reshape(N, N)

    threshold = 0.0
    psi_A[np.abs(psi_A) < threshold] = 0.0
    psi_B[np.abs(psi_B) < threshold] = 0.0
    psi_C[np.abs(psi_C) < threshold] = 0.0
    psi_D[np.abs(psi_D) < threshold] = 0.0

    return {
        "E": E,
        "IPR": IPR,
        "V": V,
        "sz": sz,
        "descending_indices": descending_indices,
        "bpic_idx": bpic_idx,
        "bpic_idx2": bpic_idx2,
        "psi_A": psi_A,
        "psi_B": psi_B,
        "psi_C": psi_C,
        "psi_D": psi_D,
    }


# ==========================================
# 4. Calculate the data used in D, E, F
#    Lmbda = 2.87, U = 9.82
#
# ==========================================
def calculate_bottom_row():
    Lmbda = 2.87
    U = 9.82
    sz = N * N

    H = build_H_2Layer(N, t1, U, Lmbda)
    E, V = eigh(H.real)
    IPR = np.sum(np.abs(V) ** 4, axis=0)

    idx = np.argsort(E.real)
    E, IPR, V = E[idx], IPR[idx], V[:, idx]

    descending_indices = np.argsort(IPR)[::-1]
    bpic_idx = descending_indices[0]
    bpic_idx2 = descending_indices[21]

    print(E[bpic_idx], np.dot(V[:, bpic_idx], V[:, bpic_idx2]))

    psi_A = (
        np.real(V[:sz, bpic_idx]).reshape(N, N)
        - np.real(V[:sz, bpic_idx + 1]).reshape(N, N)
    ) ** 1

    psi_B = (
        np.real(V[sz:, bpic_idx]).reshape(N, N)
        - np.real(V[sz:, bpic_idx + 1]).reshape(N, N)
    ) ** 1

    psi_C = np.real(V[:sz, bpic_idx2]).reshape(N, N)
    psi_D = np.real(V[sz:, bpic_idx2]).reshape(N, N)

    threshold = 0.0
    psi_A[np.abs(psi_A) < threshold] = 0.0
    psi_B[np.abs(psi_B) < threshold] = 0.0
    psi_C[np.abs(psi_C) < threshold] = 0.0
    psi_D[np.abs(psi_D) < threshold] = 0.0

    return {
        "E": E,
        "IPR": IPR,
        "V": V,
        "sz": sz,
        "descending_indices": descending_indices,
        "bpic_idx": bpic_idx,
        "bpic_idx2": bpic_idx2,
        "psi_A": psi_A,
        "psi_B": psi_B,
        "psi_C": psi_C,
        "psi_D": psi_D,
    }



def format_main_axis(axis):
    axis.grid(False)
    axis.minorticks_off()

    axis.patch.set_facecolor("none")
    axis.patch.set_alpha(0.0)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5.0,
        width=1.15,
        top=False,
        right=False,
        pad=5.0,
    )


def add_panel_label(axis, label, x=-0.16, y=1.055):
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=1000,
    )


# ==========================================
# 6. Plotting functions
# ==========================================
def plot_ipr(axis, data, panel_label):
    E = data["E"]
    IPR = data["IPR"]
    bpic_idx = data["bpic_idx"]
    bpic_idx2 = data["bpic_idx2"]

    axis.scatter(
        E.real,
        IPR,
        s=20,
        c="black",
        alpha=0.58,
        edgecolors="none",
        zorder=2,
    )

    axis.scatter(
        E[bpic_idx].real,
        IPR[bpic_idx],
        s=44,
        c="red",
        edgecolors="none",
        zorder=4,
    )

    axis.scatter(
        E[bpic_idx2].real,
        IPR[bpic_idx2],
        s=44,
        c="blue",
        edgecolors="none",
        zorder=4,
    )

    axis.set_xlabel(r"$E$")
    axis.set_ylabel("IPR")

    format_main_axis(axis)

    if panel_label == "A":
        axis.tick_params(axis="both", which="major", direction="out")
        axis.set_xticks([0, 5, 10])

    add_panel_label(axis, panel_label)


def plot_heatmap_group(figure, parent_spec, data, panel_label, label_y_figure=None):
    heatmap_colorbar_gap = 0.1

    inner_grid = parent_spec.subgridspec(
        2,
        7,
        width_ratios=[
            1.0, heatmap_colorbar_gap, 0.050,
            0.32,
            1.0, heatmap_colorbar_gap, 0.050,
        ],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.18,
    )

    heatmap_axes = []
    colorbar_axes = []

    for row_index in range(2):
        heatmap_axes.append(
            figure.add_subplot(inner_grid[row_index, 0])
        )

        colorbar_axes.append(
            figure.add_subplot(inner_grid[row_index, 2])
        )

        heatmap_axes.append(
            figure.add_subplot(inner_grid[row_index, 4])
        )

        colorbar_axes.append(
            figure.add_subplot(inner_grid[row_index, 6])
        )

    for axis_object in heatmap_axes + colorbar_axes:
        axis_object.patch.set_facecolor("none")
        axis_object.patch.set_alpha(0.0)

    spatial_maps = [
        data["psi_A"],
        data["psi_B"],
        data["psi_C"],
        data["psi_D"],
    ]

    layer_titles = [
        "Layer A",
        "Layer B",
        "Layer A",
        "Layer B",
    ]

    for index, (current_axis, current_colorbar_axis) in enumerate(
        zip(heatmap_axes, colorbar_axes)
    ):
        image = current_axis.imshow(
            spatial_maps[index],
            cmap="RdBu",
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )

        current_axis.set_title(
            layer_titles[index],
            pad=5,
            fontsize=12.5,
        )

        current_axis.set_xticks([])
        current_axis.set_yticks([])

        for spine in current_axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.05)

        colorbar = figure.colorbar(
            image,
            cax=current_colorbar_axis,
            orientation="vertical",
        )

        vmin, vmax = image.get_clim()
        colorbar.set_ticks([vmin, vmax])
        colorbar.set_ticklabels(["Min", "Max"])

        colorbar.ax.tick_params(
            labelsize=11.5,
            width=0.85,
            length=0.0,
            pad=2,
        )
        colorbar.outline.set_linewidth(0.9)

    # Panel label
    if label_y_figure is None:
        add_panel_label(
            heatmap_axes[0],
            panel_label,
            x=-0.17,
            y=1.12,
        )
    else:
        heatmap_pos = heatmap_axes[0].get_position()
        figure.text(
            heatmap_pos.x0 - 0.17 * heatmap_pos.width,
            label_y_figure,
            panel_label,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    return heatmap_axes


def get_selected_normalized_rho_data(data, bic_vals, energy_min=None, ipr_min=None):
    E = data["E"]
    IPR = data["IPR"]
    V = data["V"]
    sz = data["sz"]

    candidate_mask = np.ones(len(IPR), dtype=bool)
    if energy_min is not None:
        candidate_mask &= (E.real >= energy_min)
    if ipr_min is not None:
        candidate_mask &= (IPR >= ipr_min)

    candidate_indices = np.where(candidate_mask)[0]
    descending_indices = candidate_indices[
        np.argsort(IPR[candidate_indices])[::-1]
    ]

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r_vals = np.arange(-(N - 1), N)

    display_mask = (r_vals >= -4) & (r_vals <= 4)
    r_display = r_vals[display_mask]

    rho_bars = []
    ipr_labels = []

    for rank in bic_vals:
        bulk_idx = descending_indices[rank]

        psi_A = np.abs(V[:sz, bulk_idx].reshape(N, N))
        psi_B = np.abs(V[sz:, bulk_idx].reshape(N, N))
        rho_total = psi_A + psi_B

        rho_r = np.zeros(2 * N - 1)
        r_flat = (X - Y).flatten()
        rho_flat = rho_total.flatten()
        np.add.at(rho_r, r_flat + (N - 1), rho_flat)

        rho_selected = rho_r[display_mask]
        max_val = np.max(rho_selected)
        if max_val > 0:
            rho_selected = rho_selected / max_val

        rho_bars.append(rho_selected)
        ipr_labels.append(IPR[bulk_idx])

    return r_display, rho_bars, ipr_labels


def plot_rho_layered_bars(figure, parent_spec, data, panel_label, bic_vals, energy_min=None, ipr_min=None, label_y_figure=None):
    r_display, rho_bars, ipr_labels = get_selected_normalized_rho_data(
        data=data,
        bic_vals=bic_vals,
        energy_min=energy_min,
        ipr_min=ipr_min,
    )

    inner_grid = parent_spec.subgridspec(4, 1, hspace=0.0)

    axes = []
    for i in range(4):
        if i == 0:
            ax = figure.add_subplot(inner_grid[i, 0])
        else:
            ax = figure.add_subplot(inner_grid[i, 0], sharex=axes[0])
        axes.append(ax)

    colors = WATERFALL_CURVE_COLORS[:len(bic_vals)]

    outer_lw = 1.05
    inner_lw = 0.85
    outer_color = "black"
    inner_color = "0.42"

    def add_top_rounded_bar(ax, x_center, height, width, color, radius=0.08):
        if height <= 0:
            return

        left = x_center - width / 2
        right = x_center + width / 2
        bottom = 0.0
        top = height

        r = min(radius, width / 2, height)

        vertices = [
            (left, bottom),          
            (right, bottom),         
            (right, top - r),       
            (right, top),            
            (right - r, top),        
            (left + r, top),         
            (left, top),             
            (left, top - r),         
            (left, bottom),          
            (left, bottom),          
        ]

        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]

        patch = PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=1,
            zorder=3,
        )
        ax.add_patch(patch)

    for i, ax in enumerate(axes):
        bar_width = 0.78
        rounding = 0.08

        for x0, height in zip(r_display, rho_bars[i]):
            add_top_rounded_bar(
                ax=ax,
                x_center=x0,
                height=height,
                width=bar_width,
                color=colors[i],
                radius=rounding,
            )

        ax.set_ylim(0.0, 1.04)
        ax.set_xlim(-4.5, 4.5)
        ax.set_yticks([])
        ax.minorticks_off()
        ax.grid(False)
        ax.set_facecolor("white")

        for side in ("left", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(outer_color)
            ax.spines[side].set_linewidth(outer_lw)
            ax.spines[side].set_zorder(10)

        if i == 0:
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_color(outer_color)
            ax.spines["top"].set_linewidth(outer_lw)
            ax.spines["top"].set_zorder(10)
        else:
            ax.spines["top"].set_visible(False)

        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_zorder(10)
        if i == 3:
            ax.spines["bottom"].set_color(outer_color)
            ax.spines["bottom"].set_linewidth(outer_lw)
        else:
            ax.spines["bottom"].set_color(inner_color)
            ax.spines["bottom"].set_linewidth(inner_lw)

        if i < 3:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
        else:
            ax.set_xticks(np.arange(-4, 5, 1))
            ax.tick_params(
                axis="x",
                which="major",
                direction="out",
                length=5.0,
                width=1.15,
                bottom=True,
                top=False,
                pad=5.0,
                labelsize=12.5,
            )
            ax.set_xlabel(
                r"$r=x_2-x_1$",
                fontsize=15.5,
                labelpad=4.0,
            )

        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

        ax.text(
            0.965,
            0.78,
            rf"$\mathrm{{IPR}}={ipr_labels[i]:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=12,
        )

    top_pos = axes[0].get_position()
    bottom_pos = axes[-1].get_position()
    figure.text(
        top_pos.x0 - 0.012,
        0.5 * (top_pos.y1 + bottom_pos.y0),
        r"$\rho(r)$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=15.5,
    )

    # Panel label
    if label_y_figure is None:
        add_panel_label(
            axes[0],
            panel_label,
            x=-0.16,
            y=1.12,
        )
    else:
        cf_pos = axes[0].get_position()
        figure.text(
            cf_pos.x0 - 0.06 * cf_pos.width,
            label_y_figure,
            panel_label,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return axes

def plot_rho_top(figure, parent_spec, data, panel_label, label_y_figure=None):
    
    bic_vals = [0, 6, 12, 19]

    plot_rho_layered_bars(
        figure=figure,
        parent_spec=parent_spec,
        data=data,
        panel_label=panel_label,
        bic_vals=bic_vals,
        energy_min=5.0,
        ipr_min=None,
        label_y_figure=label_y_figure,
    )


def plot_rho_bottom(figure, parent_spec, data, panel_label, label_y_figure=None):
    bic_vals = [0, 4, 10, 19]

    plot_rho_layered_bars(
        figure=figure,
        parent_spec=parent_spec,
        data=data,
        panel_label=panel_label,
        bic_vals=bic_vals,
        energy_min=2.0,
        ipr_min=0.02,
        label_y_figure=label_y_figure,
    )


# ==========================================
# 7. Calculate both rows
# ==========================================
top_data = calculate_top_row()
bottom_data = calculate_bottom_row()


figure = plt.figure(
    figsize=(14.8, 8.8),
    dpi=160,
    facecolor="white",
)

outer_grid = figure.add_gridspec(
    2,
    5,
    left=0.060,
    right=0.985,
    bottom=0.085,
    top=0.950,

    width_ratios=[1.0, 0.12, 1.0, 0.22, 1.00],
    height_ratios=[1.0, 1.0],

    wspace=0.0,
    hspace=0.25,
)


# ------------------------------------------------------------
# Lmbda = 0
# ------------------------------------------------------------
axis_A = figure.add_subplot(
    outer_grid[0, 0]
)

plot_ipr(
    axis=axis_A,
    data=top_data,
    panel_label="A",
)

axis_A_pos = axis_A.get_position()
top_label_y = axis_A_pos.y0 + 1.055 * axis_A_pos.height

plot_heatmap_group(
    figure=figure,
    parent_spec=outer_grid[0, 2],
    data=top_data,
    panel_label="B",
    label_y_figure=top_label_y,
)

plot_rho_top(
    figure=figure,
    parent_spec=outer_grid[0, 4],
    data=top_data,
    panel_label="C",
    label_y_figure=top_label_y,
)


# ------------------------------------------------------------
# Lmbda = 2.87
# ------------------------------------------------------------
axis_D = figure.add_subplot(
    outer_grid[1, 0]
)

plot_ipr(
    axis=axis_D,
    data=bottom_data,
    panel_label="D",
)

axis_D_pos = axis_D.get_position()
bottom_label_y = axis_D_pos.y0 + 1.055 * axis_D_pos.height

plot_heatmap_group(
    figure=figure,
    parent_spec=outer_grid[1, 2],
    data=bottom_data,
    panel_label="E",
    label_y_figure=bottom_label_y,
)

plot_rho_bottom(
    figure=figure,
    parent_spec=outer_grid[1, 4],
    data=bottom_data,
    panel_label="F",
    label_y_figure=bottom_label_y,
)



plt.show()