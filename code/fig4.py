import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.colors import (
    Normalize,
    LinearSegmentedColormap,
    to_rgb
)
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CODE_DIR = Path(__file__).resolve().parent

ROOT_DIR = CODE_DIR.parent

FIG4_DATA_DIR = ROOT_DIR / "data" / "fig4"


DATA_DIRS = {
    "lowerA1": FIG4_DATA_DIR / "lowerA1",
    "lowerE5": FIG4_DATA_DIR / "lowerE5",
    "lowerB8": FIG4_DATA_DIR / "lowerB8",
    "upperE6": FIG4_DATA_DIR / "upperE6",
}


C_FIELD_DATA_DIR = (
    FIG4_DATA_DIR
    / "fig4_c"
)

C_FIELD_TARGET_FREQ = 3212.0


D_FIELD_DATA_DIR = (
    FIG4_DATA_DIR
    / "lowerB8"
)

D_FIELD_TARGET_FREQ = 3186.0


PRESSURE_DB_COL = 2
P_REF = 20e-6


FREQ_MIN = 2900
FREQ_MAX = 3400


PEAK_PROMINENCE = 0.002
PEAK_DISTANCE_HZ = 10
PEAK_MIN_HEIGHT = 0.01


SELECTED_IPR_RANKS = [1, 3, 4, 6]






MANUAL_PEAK_FREQS = {
    "lowerA1": [],
    "lowerE5":   [],
    "lowerB8":   [3186.0],
    "upperE6":   [],
}

EXCLUDE_PEAK_FREQS = {
    "lowerA1": [],
    "lowerE5":   [],
    "lowerB8":   [],
    "upperE6":   [],
}


plt.rcParams.update({

    "font.family": "Arial",


    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",


    "axes.unicode_minus": False,




    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,


    "axes.linewidth": 1.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,


})


CYLINDER_RADIUS = 0.28
CYLINDER_SEGMENTS = 32


CYLINDER_LIGHT_AZIMUTH_DEG = 315
CYLINDER_SIDE_SHADE_MIN = 0.58
CYLINDER_SIDE_SHADE_MAX = 0.92
CYLINDER_TOP_BRIGHTEN_MIX = 0.96

VIEW_ELEV = 25
VIEW_AZIM = -60
BOX_ASPECT = (1.0, 1.0, 0.58)


FIELD_3D_ZOOM = 1.15

ZMAX_SCALE = 1.08

OUTER_FRAME_COLOR = "black"
INNER_FRAME_COLOR = "0.84"

LAYER_TEXT_X = 4.4
LAYER_TEXT_Y = 1.0
LAYER_TEXT_Z_RATIO = 0.7
LAYER_TEXT_FONTSIZE = 15

AXIS_TICK_FONTSIZE = 15
XY_TICK_PAD = -2
Z_TICK_PAD = 1


LOWERA1_HIGHLIGHT_COLOR = "#B80824"


LOWERB8_HIGHLIGHT_COLOR = "#327DB7"


LAYER_MARKER_STYLES = {
    "upper": {
        "color": LOWERA1_HIGHLIGHT_COLOR,
        "marker": "o",
    },
    "lower": {
        "color": LOWERA1_HIGHLIGHT_COLOR,
        "marker": "o",
    },
}


LAYER_MARKER_X_OFFSET = -0.60
LAYER_MARKER_SIZE_3D = 48


COLUMNS = list("ABCDEFGHIJ")

site_pattern = re.compile(r"^(upper|lower)([A-Ja-j])(10|[1-9])$")


def parse_filename(file_path):


    stem = file_path.stem

    if "-" not in stem:
        return None

    _, receiver_name = stem.split("-", 1)

    match = site_pattern.match(receiver_name)
    if match is None:
        return None

    layer = match.group(1).lower()
    col_letter = match.group(2).upper()
    row_num = int(match.group(3))

    col_idx = COLUMNS.index(col_letter)
    row_idx = row_num - 1
    site_name = f"{layer}{col_letter}{row_num}"

    return layer, col_idx, row_idx, site_name


def fix_frequency_column(freq_raw):


    freq_raw = np.asarray(freq_raw, dtype=float)

    if len(freq_raw) >= 2 and freq_raw[1] < freq_raw[0]:
        f0 = freq_raw[0]
        freq = freq_raw.copy()
        freq[0] = f0
        freq[1:] = f0 + freq_raw[1:]
        return freq

    return freq_raw


def db_to_pa(p_db, p_ref=P_REF):
    # Convert sound-pressure level from dB to Pa.
    return p_ref * 10 ** (p_db / 20.0)


def read_full_spectrum(file_path):


    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python"
    )

    df = df.dropna()

    if df.shape[1] <= PRESSURE_DB_COL:
        raise ValueError(f"Insufficient number of columns; at least {PRESSURE_DB_COL + 1} columns are required.")

    freq_raw = df.iloc[:, 0].to_numpy(dtype=float)
    freq = fix_frequency_column(freq_raw)

    p_db = df.iloc[:, PRESSURE_DB_COL].to_numpy(dtype=float)
    p_pa = db_to_pa(p_db)

    return freq, p_pa


def read_value_at_target_freq(file_path, target_freq):


    freq, p_pa = read_full_spectrum(file_path)
    idx = np.argmin(np.abs(freq - target_freq))
    return freq[idx], p_pa[idx]


def format_axes(ax):
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.05)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        width=0.9,
        length=4.5,
        pad=3.0
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.45,
        color="0.86",
        alpha=0.75,
        zorder=0
    )


def find_peaks_simple(y, min_height=0.0, min_distance_points=1, prominence=0.0):
    y = np.asarray(y, dtype=float)
    candidate_indices = []

    for i in range(1, len(y) - 1):
        if not np.isfinite(y[i]):
            continue

        is_local_peak = (
            y[i] > y[i - 1]
            and y[i] >= y[i + 1]
            and y[i] >= min_height
        )

        if not is_local_peak:
            continue

        left_min = np.nanmin(y[:i + 1])
        right_min = np.nanmin(y[i:])
        simple_prominence = y[i] - max(left_min, right_min)

        if simple_prominence >= prominence:
            candidate_indices.append(i)

    if len(candidate_indices) == 0:
        return np.array([], dtype=int)

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    order = np.argsort(y[candidate_indices])[::-1]
    candidate_indices = candidate_indices[order]

    selected_indices = []

    for idx in candidate_indices:
        is_far_enough = all(abs(idx - old_idx) >= min_distance_points for old_idx in selected_indices)
        if is_far_enough:
            selected_indices.append(idx)

    return np.asarray(sorted(selected_indices), dtype=int)


def calculate_all_for_dir(data_dir, label):
    site_data = {}
    txt_files = sorted(data_dir.glob("*.txt"))
    if len(txt_files) == 0:
        raise RuntimeError(f"[{label}] No .txt files were found in directory: {data_dir}")


    for file_path in txt_files:
        receiver_info = parse_filename(file_path)

        if receiver_info is None:
            continue

        layer, col_idx, row_idx, site_name = receiver_info

        try:
            freq, p_pa = read_full_spectrum(file_path)
        except Exception:

            continue

        site_data[site_name] = {
            "freq": freq,
            "p_pa": p_pa,
            "layer": layer,
            "col_idx": col_idx,
            "row_idx": row_idx,
        }


    if len(site_data) == 0:
        raise RuntimeError(f"[{label}] No valid data were loaded.")


    site_range_data = {}
    valid_site_names = []

    for site_name, data in site_data.items():
        freq = data["freq"]
        p_pa = data["p_pa"]

        valid_mask = (
            (freq >= FREQ_MIN)
            & (freq <= FREQ_MAX)
            & np.isfinite(freq)
            & np.isfinite(p_pa)
        )

        if not np.any(valid_mask):

            continue

        site_range_data[site_name] = {
            "freq": freq[valid_mask],
            "p_pa": p_pa[valid_mask],
        }
        valid_site_names.append(site_name)

    if len(valid_site_names) == 0:
        raise RuntimeError(f"[{label}] No valid data were found within {FREQ_MIN}-{FREQ_MAX} Hz.")


    first_site = valid_site_names[0]
    freq_ref = site_range_data[first_site]["freq"]

    final_site_names = []

    for site_name in valid_site_names:
        current_freq = site_range_data[site_name]["freq"]

        if len(current_freq) != len(freq_ref):

            continue

        if not np.allclose(current_freq, freq_ref, rtol=0, atol=1e-6):

            continue

        final_site_names.append(site_name)

    if len(final_site_names) == 0:
        raise RuntimeError(f"[{label}] No data with a consistent frequency axis were found.")


    pressure_matrix = []
    site_meta = []

    for site_name in final_site_names:
        pressure_matrix.append(site_range_data[site_name]["p_pa"])

        # Use r = x2 - x1, with row_idx as x2 and col_idx as x1.
        site_meta.append({
            "site_name": site_name,
            "layer": site_data[site_name]["layer"],
            "col_idx": site_data[site_name]["col_idx"],
            "row_idx": site_data[site_name]["row_idx"],


            "relative_distance": (
                site_data[site_name]["row_idx"]
                - site_data[site_name]["col_idx"]
            ),
        })

    pressure_matrix = np.asarray(pressure_matrix, dtype=float)


    # Compute IPR over all valid measurement sites.
    sum_p2 = np.sum(pressure_matrix ** 2, axis=0)
    sum_p4 = np.sum(pressure_matrix ** 4, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ipr = sum_p4 / (sum_p2 ** 2)

    ipr[sum_p2 == 0] = np.nan

    valid_ipr_mask = np.isfinite(ipr)
    freq_range = freq_ref[valid_ipr_mask]
    ipr_range = ipr[valid_ipr_mask]

    if len(freq_range) == 0:
        raise RuntimeError(f"[{label}] No valid IPR data are available for peak detection.")

    if len(freq_range) >= 2:
        freq_step = float(np.median(np.diff(freq_range)))
    else:
        freq_step = 1.0

    min_distance_points = max(1, int(round(PEAK_DISTANCE_HZ / freq_step)))

    try:
        from scipy.signal import find_peaks
        peak_indices, _ = find_peaks(
            ipr_range,
            height=PEAK_MIN_HEIGHT,
            prominence=PEAK_PROMINENCE,
            distance=min_distance_points
        )
    except ImportError:

        peak_indices = find_peaks_simple(
            ipr_range,
            min_height=PEAK_MIN_HEIGHT,
            min_distance_points=min_distance_points,
            prominence=PEAK_PROMINENCE
        )


    manual_indices = []
    for manual_freq in MANUAL_PEAK_FREQS.get(label, []):
        nearest_idx = int(np.argmin(np.abs(freq_range - manual_freq)))
        manual_indices.append(nearest_idx)

    if len(manual_indices) > 0:
        peak_indices = np.unique(
            np.concatenate([peak_indices, np.asarray(manual_indices, dtype=int)])
        )


    exclude_freqs = EXCLUDE_PEAK_FREQS.get(label, [])
    if len(exclude_freqs) > 0 and len(peak_indices) > 0:
        peak_freqs_temp = freq_range[peak_indices]
        keep_mask = np.ones(len(peak_freqs_temp), dtype=bool)

        tolerance = max(abs(freq_step) * 0.51, 1e-6)

        for excluded_freq in exclude_freqs:
            keep_mask &= (np.abs(peak_freqs_temp - excluded_freq) > tolerance)

        peak_indices = peak_indices[keep_mask]

    peak_freqs = freq_range[peak_indices]
    peak_iprs = ipr_range[peak_indices]

    sort_indices = np.argsort(peak_freqs)
    peak_indices = peak_indices[sort_indices]
    peak_freqs = peak_freqs[sort_indices]
    peak_iprs = peak_iprs[sort_indices]


    original_valid_indices = np.where(valid_ipr_mask)[0]
    peak_column_indices = original_valid_indices[peak_indices]


    r_labels = np.arange(-9, 10)
    peak_rho_dict = {}

    for peak_freq, column_idx in zip(peak_freqs, peak_column_indices):
        p_at_peak = pressure_matrix[:, column_idx]

        # Normalize the upper and lower layers together before evaluating rho(r).
        norm_factor = np.sqrt(np.sum(np.abs(p_at_peak) ** 2))
        if not np.isfinite(norm_factor) or norm_factor == 0:

            continue

        p_normalized = p_at_peak / norm_factor
        rho_values = []

        for r_value in r_labels:
            rho_at_r = 0.0

            for site_idx, meta in enumerate(site_meta):
                if meta["relative_distance"] == r_value:

                    # rho(r) sums absolute amplitudes rather than squared amplitudes.
                    rho_at_r += np.abs(p_normalized[site_idx])

            rho_values.append(rho_at_r)

        peak_rho_dict[float(peak_freq)] = np.asarray(
            rho_values,
            dtype=float
        )

    return {
        "label": label,
        "freq": freq_range,
        "ipr": ipr_range,
        "peak_freqs": peak_freqs,
        "peak_iprs": peak_iprs,
        "r_labels": r_labels,
        "peak_rho_dict": peak_rho_dict,
    }


def load_pressure_map_for_target_freq(data_dir, target_freq, zero_source_site=None):


    pressure = {
        "upper": np.full((10, 10), np.nan),
        "lower": np.full((10, 10), np.nan),
    }

    selected_freq_record = []
    txt_files = sorted(data_dir.glob("*.txt"))

    for file_path in txt_files:
        receiver_info = parse_filename(file_path)

        if receiver_info is None:
            continue

        layer, col_idx, row_idx, _ = receiver_info

        try:
            selected_freq, p_pa = read_value_at_target_freq(file_path, target_freq)
        except Exception:

            continue

        pressure[layer][row_idx, col_idx] = p_pa
        selected_freq_record.append(selected_freq)


    if zero_source_site is not None:
        source_match = site_pattern.match(zero_source_site)

        if source_match is None:
            raise ValueError(
                f"zero_source_site={zero_source_site!r} does not match the site naming convention; "
                "for example, use 'lowerB8'."
            )

        source_layer = source_match.group(1).lower()
        source_col_letter = source_match.group(2).upper()
        source_row_num = int(source_match.group(3))

        source_col_idx = COLUMNS.index(source_col_letter)
        source_row_idx = source_row_num - 1

        original_source_pressure = pressure[source_layer][
            source_row_idx,
            source_col_idx
        ]

        if not np.isfinite(original_source_pressure):
            raise RuntimeError(
                f"Source position {zero_source_site} has no valid pressure value and cannot be set to zero."
            )

        # Set the selected source position to zero before normalization.
        pressure[source_layer][source_row_idx, source_col_idx] = 0.0


    all_values_raw = np.concatenate([
        pressure["upper"][np.isfinite(pressure["upper"])],
        pressure["lower"][np.isfinite(pressure["lower"])],
    ])

    if len(all_values_raw) == 0:
        raise RuntimeError("No valid pressure data were loaded for lowerA1.")

    # Normalize all valid upper- and lower-layer pressure values together.
    normalization_factor = np.sqrt(np.sum(all_values_raw ** 2))
    if normalization_factor == 0:
        raise RuntimeError("The normalization factor for lowerA1 is zero, so normalization cannot be performed.")

    pressure_norm = {
        "upper": pressure["upper"] / normalization_factor,
        "lower": pressure["lower"] / normalization_factor,
    }


    pressure_norm_sq = {
        "upper": np.abs(pressure_norm["upper"]) ** 2,
        "lower": np.abs(pressure_norm["lower"]) ** 2,
    }


    all_values_norm_sq = np.concatenate([
        pressure_norm_sq["upper"][np.isfinite(pressure_norm_sq["upper"])],
        pressure_norm_sq["lower"][np.isfinite(pressure_norm_sq["lower"])],
    ])

    dataset_vmax = np.max(all_values_norm_sq)
    dataset_zmax = dataset_vmax * ZMAX_SCALE

    if dataset_zmax <= 0:
        dataset_zmax = 1.0

    dataset_norm = Normalize(vmin=0, vmax=dataset_vmax)


    return {
        "pressure_norm": pressure_norm,
        "pressure_norm_sq": pressure_norm_sq,
        "vmax": dataset_vmax,
        "zmax": dataset_zmax,
        "norm": dataset_norm,
        "selected_freq_record": selected_freq_record,
        "target_freq": target_freq,
    }


def mix_with_white(color, mix=0.12):
    rgb = np.array(to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = (1 - mix) * white + mix * rgb
    return tuple(mixed)


def make_monochrome_cmap(base_color, cmap_name="mono"):
    light_color = mix_with_white(base_color, mix=0.12)
    cmap = LinearSegmentedColormap.from_list(
        cmap_name,
        [light_color, base_color]
    )
    return cmap


def make_cylinder_side_facecolors(base_color, theta):


    base_rgb = np.asarray(to_rgb(base_color), dtype=float)


    theta_mid = 0.5 * (theta[:-1] + theta[1:])


    light_azimuth_rad = np.deg2rad(CYLINDER_LIGHT_AZIMUTH_DEG)


    light_profile = 0.5 * (
        1.0 + np.cos(theta_mid - light_azimuth_rad)
    )


    brightness = (
        CYLINDER_SIDE_SHADE_MIN
        + (
            CYLINDER_SIDE_SHADE_MAX
            - CYLINDER_SIDE_SHADE_MIN
        ) * light_profile
    )

    facecolors = np.empty(
        (1, len(theta_mid), 4),
        dtype=float,
    )

    for idx, factor in enumerate(brightness):
        rgb = np.clip(
            base_rgb * factor,
            0.0,
            1.0,
        )

        facecolors[0, idx, :3] = rgb
        facecolors[0, idx, 3] = 1.0

    return facecolors


def draw_cylinder(
    ax,
    center_x,
    center_y,
    height,
    radius,
    color,
    n_segments=32,
):


    if not np.isfinite(height):
        return

    if height <= 0:
        return

    theta = np.linspace(
        0,
        2 * np.pi,
        n_segments + 1,
    )

    circle_x = (
        center_x
        + radius * np.cos(theta)
    )
    circle_y = (
        center_y
        + radius * np.sin(theta)
    )


    side_x = np.vstack([
        circle_x,
        circle_x,
    ])

    side_y = np.vstack([
        circle_y,
        circle_y,
    ])

    side_z = np.vstack([
        np.zeros_like(theta),
        np.full_like(theta, height),
    ])


    side_facecolors = make_cylinder_side_facecolors(
        color,
        theta,
    )

    side_surface = ax.plot_surface(
        side_x,
        side_y,
        side_z,
        facecolors=side_facecolors,
        linewidth=0,
        antialiased=True,


        shade=False,

        zorder=10,
    )

    side_surface.set_clip_on(False)
    side_surface.set_clip_path(None)


    top_vertices = [
        (
            center_x + radius * np.cos(angle),
            center_y + radius * np.sin(angle),
            height,
        )
        for angle in theta[:-1]
    ]


    top_color = mix_with_white(
        color,
        mix=CYLINDER_TOP_BRIGHTEN_MIX,
    )

    top_surface = Poly3DCollection(
        [top_vertices],
        facecolor=top_color,
        edgecolor="none",
        zorder=11,
    )

    top_surface.set_clip_on(False)
    top_surface.set_clip_path(None)

    ax.add_collection3d(top_surface)


def style_3d_axis(ax):


    ax.set_facecolor((1, 1, 1, 0))
    ax.patch.set_facecolor((1, 1, 1, 0))
    ax.patch.set_alpha(0)

    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis.line.set_color((1, 1, 1, 0))

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=AXIS_TICK_FONTSIZE,
        pad=XY_TICK_PAD,
        length=3,
        width=0.8
    )

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=AXIS_TICK_FONTSIZE,
        pad=XY_TICK_PAD,
        length=3,
        width=0.8
    )

    ax.tick_params(
        axis="z",
        which="major",
        labelsize=AXIS_TICK_FONTSIZE,
        pad=Z_TICK_PAD,
        length=3,
        width=0.8
    )


def disable_3d_axes_clipping(ax):


    for artist in ax.get_children():
        try:
            artist.set_clip_on(False)
        except (AttributeError, TypeError):
            pass

        try:
            artist.set_clip_path(None)
        except (AttributeError, TypeError):
            pass


    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for tick in axis.get_major_ticks():
            for artist in (
                tick.label1,
                tick.label2,
                tick.tick1line,
                tick.tick2line,
                tick.gridline,
            ):
                try:
                    artist.set_clip_on(False)
                    artist.set_clip_path(None)
                except (AttributeError, TypeError):
                    pass


    ax.patch.set_visible(False)


def _get_box_points(ax):

    x0, x1 = sorted(ax.get_xlim3d())
    y0, y1 = sorted(ax.get_ylim3d())
    z0, z1 = sorted(ax.get_zlim3d())

    a = (x0, y0, z0)
    b = (x1, y0, z0)
    c = (x0, y1, z0)
    d = (x1, y1, z0)

    e = (x0, y0, z1)
    f = (x1, y0, z1)
    g = (x0, y1, z1)
    h = (x1, y1, z1)

    return a, b, c, d, e, f, g, h


def _draw_box_edge(ax, p1, p2, color, linewidth, zorder):

    line = ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="round",
        zorder=zorder
    )[0]
    line.set_clip_on(False)
    line.set_clip_path(None)
    return line


def draw_inner_box_edges_under(ax):


    points = _get_box_points(ax)
    a, b, c, e = points[0], points[1], points[2], points[4]

    _draw_box_edge(ax, a, b, INNER_FRAME_COLOR, 0.75, zorder=0)
    _draw_box_edge(ax, a, c, INNER_FRAME_COLOR, 0.75, zorder=0)
    _draw_box_edge(ax, a, e, INNER_FRAME_COLOR, 0.75, zorder=0)


def draw_outer_box_edges_over(ax):


    points = _get_box_points(ax)
    b, c, d, e, f, g = (points[i] for i in (1, 2, 3, 4, 5, 6))

    _draw_box_edge(ax, b, d, OUTER_FRAME_COLOR, 1.05, zorder=30)
    _draw_box_edge(ax, b, f, OUTER_FRAME_COLOR, 1.05, zorder=30)
    _draw_box_edge(ax, c, d, OUTER_FRAME_COLOR, 1.05, zorder=30)
    _draw_box_edge(ax, c, g, OUTER_FRAME_COLOR, 1.05, zorder=30)
    _draw_box_edge(ax, e, f, OUTER_FRAME_COLOR, 1.05, zorder=30)
    _draw_box_edge(ax, e, g, OUTER_FRAME_COLOR, 1.05, zorder=30)


def add_layer_text_inside(
    ax,
    text_string,
    zmax,
    layer,
    marker_color=None,
):

    style = LAYER_MARKER_STYLES[layer]
    current_marker_color = (
        style["color"]
        if marker_color is None
        else marker_color
    )
    z_position = zmax * LAYER_TEXT_Z_RATIO


    layer_marker = ax.scatter(
        [LAYER_TEXT_X + LAYER_MARKER_X_OFFSET],
        [LAYER_TEXT_Y],
        [z_position],
        s=LAYER_MARKER_SIZE_3D,
        marker="o",
        facecolors=current_marker_color,
        edgecolors=current_marker_color,
        linewidths=1.2,
        depthshade=False,
        zorder=51,
    )
    layer_marker.set_clip_on(False)
    layer_marker.set_clip_path(None)

    ax.text(
        LAYER_TEXT_X,
        LAYER_TEXT_Y,
        z_position,
        text_string,
        fontsize=LAYER_TEXT_FONTSIZE,
        fontweight="normal",
        ha="left",
        va="center",
        zorder=52,
        clip_on=False,
    )




LOWERB8_SPECIAL_FREQ = 3186.0


HEXAGON_LOWERA1 = LOWERA1_HIGHLIGHT_COLOR
HEXAGON_LOWERB8 = LOWERB8_HIGHLIGHT_COLOR


LOWERB8_BASE_COLOR = LOWERB8_HIGHLIGHT_COLOR


SELECTED_COLORS = [
    "#D71C2C",
    "#FEAB88",
    "#6FAFD2",
    "#327DB7",
]


results = {}

for label, data_dir in DATA_DIRS.items():
    results[label] = calculate_all_for_dir(
        data_dir=data_dir,
        label=label,
    )


all_selected_peaks = []

for label, result in results.items():
    for peak_no, (peak_freq, peak_ipr) in enumerate(
        zip(result["peak_freqs"], result["peak_iprs"]),
        start=1,
    ):
        peak_freq_float = float(peak_freq)

        if peak_freq_float not in result["peak_rho_dict"]:
            continue

        all_selected_peaks.append({
            "label": label,
            "peak_no": peak_no,
            "peak_freq": peak_freq_float,
            "peak_ipr": float(peak_ipr),
            "r_labels": result["r_labels"],
            "rho_values": result["peak_rho_dict"][peak_freq_float],
        })

if len(all_selected_peaks) == 0:
    raise RuntimeError("No IPR peaks are available for plotting.")

all_selected_peaks_sorted = sorted(
    all_selected_peaks,
    key=lambda item: item["peak_ipr"],
    reverse=True,
)


colored_peaks = [
    all_selected_peaks_sorted[rank - 1]
    for rank in SELECTED_IPR_RANKS
    if rank <= len(all_selected_peaks_sorted)
]

colored_peak_color_map = {
    (item["label"], item["peak_freq"]): SELECTED_COLORS[index]
    for index, item in enumerate(colored_peaks)
}

maximum_ipr_peak = all_selected_peaks_sorted[0]


lowerb8_candidates = [
    item
    for item in all_selected_peaks
    if item["label"] == "lowerB8"
]

if len(lowerb8_candidates) == 0:
    raise RuntimeError("No IPR peak was found for lowerB8.")

lowerb8_special_peak = min(
    lowerb8_candidates,
    key=lambda item: abs(item["peak_freq"] - LOWERB8_SPECIAL_FREQ),
)

if abs(lowerb8_special_peak["peak_freq"] - LOWERB8_SPECIAL_FREQ) > 1.0:
    raise RuntimeError(
        "No peak near 3186 Hz was found in lowerB8. "
        "Check whether MANUAL_PEAK_FREQS contains 3186.0."
    )


lowera1_map = load_pressure_map_for_target_freq(
    data_dir=C_FIELD_DATA_DIR,
    target_freq=C_FIELD_TARGET_FREQ,
)


lowerb8_map = load_pressure_map_for_target_freq(
    data_dir=D_FIELD_DATA_DIR,
    target_freq=D_FIELD_TARGET_FREQ,
    zero_source_site="lowerB8",
)

lowera1_cmap = make_monochrome_cmap(
    base_color=LOWERA1_HIGHLIGHT_COLOR,
    cmap_name="lowerA1_mono",
)

lowerb8_cmap = make_monochrome_cmap(
    base_color=LOWERB8_BASE_COLOR,
    cmap_name="lowerB8_mono",
)




def add_panel_label_2d(ax, label, x=-0.13, y=1.04, fontsize=15):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        zorder=100,
    )


def add_dashed_hexagon(
    ax,
    x,
    y,
    color,
    radius_points=10.0,
    linewidth=1.7,
):


    from matplotlib.patches import Polygon

    figure = ax.figure
    figure.canvas.draw()

    center_display = ax.transData.transform((x, y))
    radius_pixels = radius_points * figure.dpi / 72.0


    angles = np.deg2rad(np.arange(90, 450, 60))

    vertices_display = np.column_stack([
        center_display[0] + radius_pixels * np.cos(angles),
        center_display[1] + radius_pixels * np.sin(angles),
    ])

    vertices_data = ax.transData.inverted().transform(vertices_display)

    patch = Polygon(
        vertices_data,
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=(0, (3.2, 2.0)),
        joinstyle="miter",
        capstyle="butt",
        zorder=20,
        clip_on=False,
    )

    ax.add_patch(patch)
    return patch


def draw_field_axis(
    ax,
    field_map,
    layer,
    cmap,
    layer_title,
    marker_color=None,
):

    grid = field_map["pressure_norm_sq"][layer]
    dataset_norm = field_map["norm"]
    dataset_vmax = field_map["vmax"]
    dataset_zmax = field_map["zmax"]

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    tick_positions = [1, 5, 10]
    tick_labels = ["1", "5", "10"]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_zticks(np.linspace(0, dataset_vmax, 4))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_zlim(0, dataset_zmax)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.invert_yaxis()
    ax.set_box_aspect(BOX_ASPECT, zoom=FIELD_3D_ZOOM)

    style_3d_axis(ax)
    draw_inner_box_edges_under(ax)

    for row_idx in range(10):
        for col_idx in range(10):
            value = grid[row_idx, col_idx]

            if not np.isfinite(value):
                continue

            draw_cylinder(
                ax=ax,
                center_x=col_idx + 1,
                center_y=row_idx + 1,
                height=value,
                radius=CYLINDER_RADIUS,
                color=cmap(dataset_norm(value)),
                n_segments=CYLINDER_SEGMENTS,
            )

    draw_outer_box_edges_over(ax)
    add_layer_text_inside(
        ax,
        layer_title,
        dataset_zmax,
        layer,
        marker_color=marker_color,
    )


    disable_3d_axes_clipping(ax)


def add_shared_field_colorbar(
    fig,
    position,
    field_map,
    cmap,
    orientation="horizontal",
):
    scalar_mappable = ScalarMappable(
        norm=field_map["norm"],
        cmap=cmap,
    )
    scalar_mappable.set_array([])

    cbar_ax = fig.add_axes(position)
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=cbar_ax,
        orientation=orientation,
    )

    colorbar.set_ticks([0, field_map["norm"].vmax])
    colorbar.set_ticklabels(["Min", "Max"])
    colorbar.ax.tick_params(
        labelsize=15,
        width=0.9,
        length=0,
        pad=3,
    )
    colorbar.outline.set_linewidth(0.9)

    if orientation == "horizontal":
        colorbar.set_label(r"$|P|^2$", fontsize=15, labelpad=3)
    else:
        cbar_ax.set_title(r"$|P|^2$", fontsize=16, pad=5)

    return colorbar


def plot_peak_rho_layered_bars(
    figure,
    rect,
    peak_items,
    peak_color_map,
):
    if len(peak_items) == 0:
        raise RuntimeError("No peaks are available for the layered bar plot in panel B.")


    r_all = np.asarray(peak_items[0]["r_labels"], dtype=float)
    display_mask = (r_all >= -4) & (r_all <= 4)
    r_display = r_all[display_mask]

    rho_bars = []
    ipr_values = []
    bar_colors = []

    for item in peak_items:
        current_r = np.asarray(item["r_labels"], dtype=float)

        if len(current_r) != len(r_all) or not np.allclose(
            current_r,
            r_all,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError("The r coordinates are inconsistent across peaks in panel B.")

        rho_selected = np.asarray(
            item["rho_values"],
            dtype=float,
        )[display_mask]


        max_val = np.nanmax(rho_selected)
        if np.isfinite(max_val) and max_val > 0:
            rho_selected = rho_selected / max_val

        rho_bars.append(rho_selected)
        ipr_values.append(float(item["peak_ipr"]))

        key = (
            item["label"],
            item["peak_freq"],
        )
        bar_colors.append(
            peak_color_map[key]
        )


    left, bottom, width, height = rect
    n_layers = len(rho_bars)
    layer_height = height / n_layers

    axes = []

    for i in range(n_layers):
        current_bottom = bottom + (n_layers - 1 - i) * layer_height

        if i == 0:
            ax = figure.add_axes(
                [left, current_bottom, width, layer_height]
            )
        else:
            ax = figure.add_axes(
                [left, current_bottom, width, layer_height],
                sharex=axes[0],
            )

        axes.append(ax)


    outer_lw = 1.05
    inner_lw = 0.85
    outer_color = "black"
    inner_color = "0.42"

    def add_top_rounded_bar(
        ax,
        x_center,
        height_value,
        bar_width,
        color,
        radius=0.08,
    ):

        if not np.isfinite(height_value) or height_value <= 0:
            return

        left_bar = x_center - bar_width / 2
        right_bar = x_center + bar_width / 2
        bottom_bar = 0.0
        top_bar = float(height_value)

        r = min(
            radius,
            bar_width / 2,
            top_bar,
        )

        vertices = [
            (left_bar, bottom_bar),
            (right_bar, bottom_bar),
            (right_bar, top_bar - r),
            (right_bar, top_bar),
            (right_bar - r, top_bar),
            (left_bar + r, top_bar),
            (left_bar, top_bar),
            (left_bar, top_bar - r),
            (left_bar, bottom_bar),
            (left_bar, bottom_bar),
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
            alpha=1.0,
            zorder=3,
        )
        ax.add_patch(patch)


    for i, ax in enumerate(axes):
        bar_width = 0.78
        rounding = 0.08

        for x0, height_value in zip(
            r_display,
            rho_bars[i],
        ):
            add_top_rounded_bar(
                ax=ax,
                x_center=x0,
                height_value=height_value,
                bar_width=bar_width,
                color=bar_colors[i],
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

        if i == n_layers - 1:
            ax.spines["bottom"].set_color(outer_color)
            ax.spines["bottom"].set_linewidth(outer_lw)
        else:
            ax.spines["bottom"].set_color(inner_color)
            ax.spines["bottom"].set_linewidth(inner_lw)


        if i < n_layers - 1:
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
                length=4.5,
                width=0.9,
                bottom=True,
                top=False,
                pad=3.0,
                labelsize=15,
            )
            ax.set_xlabel(
                r"$r=x_2-x_1$",
                fontsize=15,
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
            0.76,
            rf"$\mathrm{{IPR}}={ipr_values[i]:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11.5,
            color="0.12",
            zorder=20,
        )


    figure.text(
        left - 0.018,
        bottom + 0.50 * height,
        r"$\rho(r)$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
    )

    return axes


FIG_WIDTH = 12.2
FIG_HEIGHT = 12.4
FIG_DPI = 160


AB_LEFT = 0.050
AB_BOTTOM = 0.785
AB_WIDTH = 0.370
AB_HEIGHT = 0.2
AB_GAP = 0.045

A_LEFT = AB_LEFT
B_LEFT = A_LEFT + AB_WIDTH + AB_GAP


FIELD_LEFT = -0.030
FIELD_WIDTH = 0.450
FIELD_HEIGHT = 0.240

FIELD_CBAR_GAP_LEFT = 0.008
FIELD_CBAR_WIDTH = 0.016
FIELD_CBAR_GAP_RIGHT = 0.008
FIELD_CBAR_HEIGHT = 0.130

FIELD_CBAR_LEFT = FIELD_LEFT + FIELD_WIDTH + FIELD_CBAR_GAP_LEFT
FIELD_RIGHT = (
    FIELD_CBAR_LEFT
    + FIELD_CBAR_WIDTH
    + FIELD_CBAR_GAP_RIGHT
)


C_BOTTOM = 0.475
C_LEFT_A = FIELD_LEFT
C_LEFT_B = FIELD_RIGHT
C_CBAR_BOTTOM = C_BOTTOM + (FIELD_HEIGHT - FIELD_CBAR_HEIGHT) / 2


E_BOTTOM = 0.230
E_LEFT_A = FIELD_LEFT
E_LEFT_B = FIELD_RIGHT
E_CBAR_BOTTOM = E_BOTTOM + (FIELD_HEIGHT - FIELD_CBAR_HEIGHT) / 2


PANEL_LABEL_FONTSIZE = 17
PANEL_LABEL_2D_X = -0.13
PANEL_LABEL_2D_Y = 1.06


PANEL_LABEL_3D_Y_OFFSET = -0.025


LEFT_PANEL_LABEL_X = A_LEFT + PANEL_LABEL_2D_X * AB_WIDTH

C_LABEL_X = LEFT_PANEL_LABEL_X
C_LABEL_Y = C_BOTTOM + FIELD_HEIGHT + PANEL_LABEL_3D_Y_OFFSET

E_LABEL_X = LEFT_PANEL_LABEL_X
E_LABEL_Y = E_BOTTOM + FIELD_HEIGHT + PANEL_LABEL_3D_Y_OFFSET


fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=FIG_DPI,
    facecolor="white",
)


ax1 = fig.add_axes([
    A_LEFT,
    AB_BOTTOM,
    AB_WIDTH,
    AB_HEIGHT,
])


ax3 = fig.add_axes(
    [C_LEFT_A, C_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT],
    projection="3d",
)
ax4 = fig.add_axes(
    [C_LEFT_B, C_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT],
    projection="3d",
)


ax7 = fig.add_axes(
    [E_LEFT_A, E_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT],
    projection="3d",
)
ax8 = fig.add_axes(
    [E_LEFT_B, E_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT],
    projection="3d",
)


format_axes(ax1)


for item in all_selected_peaks:
    ax1.scatter(
        item["peak_freq"] / 1000.0,
        item["peak_ipr"],
        s=24,
        marker="o",
        facecolor="black",
        edgecolor="black",
        linewidth=0.40,
        alpha=0.88,
        zorder=2,
    )


for item in colored_peaks:
    key = (item["label"], item["peak_freq"])
    color = colored_peak_color_map[key]

    ax1.scatter(
        item["peak_freq"] / 1000.0,
        item["peak_ipr"],
        s=52,
        marker="o",
        facecolor=color,
        edgecolor="black",
        linewidth=0.65,
        alpha=1.0,
        zorder=5,
    )


ax1.set_xlabel("Frequency (kHz)", labelpad=4)
ax1.set_ylabel("IPR", labelpad=5)
ax1.set_xlim(FREQ_MIN / 1000.0, FREQ_MAX / 1000.0)
ax1.set_xticks(
    np.arange(
        FREQ_MIN / 1000.0,
        FREQ_MAX / 1000.0 + 0.001,
        0.1,
    )
)


ax1.set_ylim(0, 0.50)
ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


add_dashed_hexagon(
    ax=ax1,
    x=maximum_ipr_peak["peak_freq"] / 1000.0,
    y=maximum_ipr_peak["peak_ipr"],
    color=HEXAGON_LOWERA1,
    radius_points=10.5,
)


add_dashed_hexagon(
    ax=ax1,
    x=lowerb8_special_peak["peak_freq"] / 1000.0,
    y=lowerb8_special_peak["peak_ipr"],
    color=HEXAGON_LOWERB8,
    radius_points=10.5,
)

add_panel_label_2d(
    ax1,
    "A",
    x=PANEL_LABEL_2D_X,
    y=PANEL_LABEL_2D_Y,
    fontsize=PANEL_LABEL_FONTSIZE,
)


ax2_layers = plot_peak_rho_layered_bars(
    figure=fig,
    rect=[
        B_LEFT,
        AB_BOTTOM,
        AB_WIDTH,
        AB_HEIGHT,
    ],
    peak_items=colored_peaks,
    peak_color_map=colored_peak_color_map,
)


add_panel_label_2d(
    ax2_layers[0],
    "B",
    x=PANEL_LABEL_2D_X + 0.06,
    y=PANEL_LABEL_2D_Y,
    fontsize=PANEL_LABEL_FONTSIZE,
)


draw_field_axis(
    ax=ax3,
    field_map=lowera1_map,
    layer="upper",
    cmap=lowera1_cmap,
    layer_title="Layer A",
    marker_color=LOWERA1_HIGHLIGHT_COLOR,
)

draw_field_axis(
    ax=ax4,
    field_map=lowera1_map,
    layer="lower",
    cmap=lowera1_cmap,
    layer_title="Layer B",
    marker_color=LOWERA1_HIGHLIGHT_COLOR,
)

add_shared_field_colorbar(
    fig=fig,
    position=[
        FIELD_CBAR_LEFT,
        C_CBAR_BOTTOM,
        FIELD_CBAR_WIDTH,
        FIELD_CBAR_HEIGHT,
    ],
    field_map=lowera1_map,
    cmap=lowera1_cmap,
    orientation="vertical",
)

fig.text(
    C_LABEL_X,
    C_LABEL_Y,
    "C",
    fontsize=PANEL_LABEL_FONTSIZE,
    fontweight="bold",
)


draw_field_axis(
    ax=ax7,
    field_map=lowerb8_map,
    layer="upper",
    cmap=lowerb8_cmap,
    layer_title="Layer A",
    marker_color=LOWERB8_HIGHLIGHT_COLOR,
)

draw_field_axis(
    ax=ax8,
    field_map=lowerb8_map,
    layer="lower",
    cmap=lowerb8_cmap,
    layer_title="Layer B",
    marker_color=LOWERB8_HIGHLIGHT_COLOR,
)

add_shared_field_colorbar(
    fig=fig,
    position=[
        FIELD_CBAR_LEFT,
        E_CBAR_BOTTOM,
        FIELD_CBAR_WIDTH,
        FIELD_CBAR_HEIGHT,
    ],
    field_map=lowerb8_map,
    cmap=lowerb8_cmap,
    orientation="vertical",
)

fig.text(
    E_LABEL_X,
    E_LABEL_Y,
    "D",
    fontsize=PANEL_LABEL_FONTSIZE,
    fontweight="bold",
)


plt.show()
