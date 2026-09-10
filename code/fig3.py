import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import (
    Normalize,
    LinearSegmentedColormap,
    to_rgb,
)
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter,
)
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



CODE_DIR = Path(__file__).resolve().parent

ROOT_DIR = CODE_DIR.parent

DATA_DIR = ROOT_DIR / "data" / "fig3"


LOWER_DOS_DIR = DATA_DIR
UPPER_DOS_DIR = DATA_DIR


FIELD_DATASETS = {
    "A1": {
        "data_dir": DATA_DIR / "upperJ10",
        "coord_label": "(1,1)",
        "base_color": "#B80824",
        "title_marker": "o",
        "target_freq": 3202.0,

        "layer_b_zmax": 0.5,

        "zero_source": False,

        "source_layer": "lower",
        "source_site": "A1",
    },

    "B8": {
        "data_dir": DATA_DIR / "lowerB8",
        "coord_label": "(2,8)",
        "base_color": "#327DB7",
        "title_marker": "o",
        "target_freq": 3202.0,

        "layer_b_zmax": None,

        "zero_source": True,

        "source_layer": "lower",
        "source_site": "B8",
    },
}


PRESSURE_DB_COL = 2
P_REF = 20e-6


FREQ_MIN = 2900
FREQ_MAX = 3400


DOS_SELECTED_SITES = [
    {
        "site": "J10",
        "layer": "upper",
        "coord": "(10,10)",
    },

    {
        "site": "B8",
        "layer": "lower",
        "coord": "(2,8)",
    },
]


DOS_COLOR_RED = "#B80824"
DOS_COLOR_BLUE = "#327DB7"


DOS_RED_ALPHA = 0.88
DOS_BLUE_ALPHA = 0.88


DOS_MARKER_FREQ = 3202.0


DOS_MARKER_COLOR_A1 = "#D71C2C"
DOS_MARKER_COLOR_B8 = "#6FAFD2"


DOS_MARKER_SIZE_A1 = 15
DOS_MARKER_SIZE_B8 = 17


DOS_MARKER_EDGE_COLOR = "black"
DOS_MARKER_EDGE_WIDTH = 1.4


DOS_MARKER_SHAPE_A1 = "o"
DOS_MARKER_SHAPE_B8 = "o"


DOS_FRAME_LINE_WIDTH = 1.7


DOS_PANEL_LABEL = None


DOS_FIGSIZE = (10.0, 9)
FIELD_FIGSIZE = (19.0, 5.2)


DOS_AX_POSITION = [
    0.145,
    0.155,
    0.815,
    0.800,
]


GROUP_LEFTS = [
    0.035,
    0.525,
]


FIELD_SUB_AX_WIDTH = 0.180
FIELD_PAIR_GAP = 0.020
GROUP_WIDTH = 2 * FIELD_SUB_AX_WIDTH + FIELD_PAIR_GAP


FIELD_ROW_BOTTOM = 0.150
FIELD_AX_HEIGHT = 0.670


FIELD_TITLE_Y = 0.815
FIELD_TITLE_FONTSIZE = 18
FIELD_TITLE_MARKER_SIZE = 12


FIELD_CBAR_LENGTH = 0.280
FIELD_CBAR_THICKNESS = 0.012


FIELD_CBAR_OFFSET_X = 0.03


FIELD_CBAR_TITLE_Y_OFFSET = 0.010
FIELD_CBAR_TITLE_FONTSIZE = 18


LAYER_TEXT_X = 4.35
LAYER_TEXT_Y = 1.0
LAYER_TEXT_Z_RATIO = 0.58
LAYER_TEXT_FONTSIZE = 16


FIELD_AXIS_TICK_FONTSIZE = 18
FIELD_XY_TICK_PAD = -2
FIELD_Z_TICK_PAD = 5


CYLINDER_RADIUS = 0.28
CYLINDER_SEGMENTS = 32


CYLINDER_LIGHT_AZIMUTH_DEG = 315
CYLINDER_SIDE_SHADE_MIN = 0.58
CYLINDER_SIDE_SHADE_MAX = 0.92
CYLINDER_TOP_BRIGHTEN_MIX = 0.96


VIEW_ELEV = 25
VIEW_AZIM = -60

BOX_ASPECT = (
    1.0,
    1.0,
    0.58,
)


ZMAX_SCALE = 1.08


OUTER_FRAME_COLOR = "black"
INNER_FRAME_COLOR = "0.84"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],

    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",

    "axes.unicode_minus": False,

    "figure.facecolor": "white",
    "axes.facecolor": "white",


    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


COLUMNS = list("ABCDEFGHIJ")
ROWS = list(range(1, 11))

SITE_PATTERN = re.compile(
    r"^(upper|lower)([A-Ja-j])(10|[1-9])$"
)


def fix_frequency_column(freq_raw):
    freq_raw = np.asarray(
        freq_raw,
        dtype=float,
    )

    if (
        len(freq_raw) >= 2
        and freq_raw[1] < freq_raw[0]
    ):
        f0 = freq_raw[0]

        freq = freq_raw.copy()

        freq[0] = f0
        freq[1:] = f0 + freq_raw[1:]

        return freq

    return freq_raw


# Convert SPL from dB to pressure in Pa.
def db_to_pa(
    p_db,
    p_ref=P_REF,
):
    return (
        p_ref
        * 10 ** (p_db / 20)
    )


def site_to_filename(site):
    return f"{site}-{site}.txt"


def site_to_indices(site):
    col_letter = site[0].upper()
    row_num = int(site[1:])

    col_idx = COLUMNS.index(
        col_letter
    )

    row_idx = row_num - 1

    return (
        row_idx,
        col_idx,
    )


def read_dos_site(file_path):
    data = np.loadtxt(
        file_path
    )

    if (
        data.ndim != 2
        or data.shape[1] <= PRESSURE_DB_COL
    ):
        raise ValueError(
            f"{file_path.name} 的数据列数不足，"
            f"至少需要 {PRESSURE_DB_COL + 1} 列。"
        )

    freq_all = fix_frequency_column(
        data[:, 0]
    )

    p_db_all = data[
        :,
        PRESSURE_DB_COL
    ]

    mask = (
        (freq_all >= FREQ_MIN)
        & (freq_all <= FREQ_MAX)
    )

    freq = freq_all[mask]
    p_db = p_db_all[mask]

    if len(freq) == 0:
        raise ValueError(
            f"{file_path.name} 在 "
            f"{FREQ_MIN}-{FREQ_MAX} Hz "
            "范围内没有数据。"
        )


    p_pa = db_to_pa(
        p_db
    )


    p_pa2 = (
        p_pa ** 2
    )


    # Normalize each DOS curve by its integrated |P|^2 in the selected band.
    norm_factor = np.sum(
        p_pa2
    )

    if (
        not np.isfinite(norm_factor)
        or norm_factor <= 0
    ):
        raise ValueError(
            f"{file_path.name} 的 Pa² 总和无效，"
            "无法归一化。"
        )

    return (
        freq,
        p_pa2 / norm_factor,
    )


def load_dos_data():
    selected_files = []

    for item in DOS_SELECTED_SITES:

        site = item["site"]
        layer = item["layer"]

        if layer == "lower":

            file_path = (
                LOWER_DOS_DIR
                / site_to_filename(site)
            )

        elif layer == "upper":

            file_path = (
                UPPER_DOS_DIR
                / site_to_filename(site)
            )

        else:

            raise ValueError(
                f"未知 layer：{layer}"
            )

        selected_files.append({
            "site": site,
            "layer": layer,
            "coord": item["coord"],
            "path": file_path,
        })


    missing_files = []

    for item in selected_files:

        if not item["path"].exists():

            missing_files.append(
                item
            )


    if missing_files:

        missing_text = "\n".join(
            f"  {item['layer']}：{item['path']}"
            for item in missing_files
        )

        raise FileNotFoundError(
            "以下 DOS 文件没有找到：\n"
            f"{missing_text}"
        )


    curves = []
    freq_ref = None

    for item in selected_files:

        freq, p_norm = read_dos_site(
            item["path"]
        )

        if freq_ref is None:

            freq_ref = freq

        else:

            if (
                len(freq) != len(freq_ref)
                or not np.allclose(
                    freq,
                    freq_ref
                )
            ):
                raise ValueError(
                    "DOS 频率轴不一致：\n"
                    f"{item['path']}"
                )

        curves.append({
            "site": item["site"],
            "layer": item["layer"],
            "coord": item["coord"],
            "freq": freq,
            "p_norm": p_norm,
        })


    global_max = max(
        np.max(
            curve["p_norm"]
        )
        for curve in curves
    )

    if (
        not np.isfinite(global_max)
        or global_max <= 0
    ):
        raise ValueError(
            "DOS 的整体最大值无效。"
        )


    for curve in curves:

        curve["p_norm"] = (
            curve["p_norm"]
            / global_max
        )


    curve_map = {
        (
            curve["layer"],
            curve["site"],
        ): curve
        for curve in curves
    }

    curve_a1 = curve_map[
        ("upper", "J10")
    ]

    curve_b8 = curve_map[
        ("lower", "B8")
    ]


    freq_hz = (
        curve_a1["freq"]
    )

    freq_khz = (
        freq_hz
        / 1000.0
    )


    y_a1 = (
        curve_a1["p_norm"].copy()
    )

    y_b8 = (
        curve_b8["p_norm"].copy()
    )


    idx_a1 = int(
        np.argmin(
            np.abs(
                freq_hz
                - DOS_MARKER_FREQ
            )
        )
    )

    idx_b8 = int(
        np.argmin(
            np.abs(
                freq_hz
                - DOS_MARKER_FREQ
            )
        )
    )


    markers = {
        "A1": {
            "x_hz":
                float(freq_hz[idx_a1]),

            "x_khz":
                float(freq_khz[idx_a1]),

            "y":
                float(y_a1[idx_a1]),
        },

        "B8": {
            "x_hz":
                float(freq_hz[idx_b8]),

            "x_khz":
                float(freq_khz[idx_b8]),

            "y":
                float(y_b8[idx_b8]),
        },
    }


    return {
        "freq_hz": freq_hz,
        "freq_khz": freq_khz,

        "y_a1": y_a1,
        "y_b8": y_b8,

        "markers": markers,
        "curves": curves,
    }


def draw_dos_panel(
    ax,
    dos_data,
):
    freq_khz = (
        dos_data["freq_khz"]
    )

    y_a1 = (
        dos_data["y_a1"]
    )

    y_b8 = (
        dos_data["y_b8"]
    )

    markers = (
        dos_data["markers"]
    )


    ax.fill_between(
        freq_khz,
        0,
        y_a1,

        facecolor=DOS_COLOR_RED,
        alpha=DOS_RED_ALPHA,

        linewidth=0,
        edgecolor="none",

        zorder=1,
    )


    ax.fill_between(
        freq_khz,
        0,
        y_b8,

        facecolor=DOS_COLOR_BLUE,
        alpha=DOS_BLUE_ALPHA,

        linewidth=0,
        edgecolor="none",

        zorder=2,
    )


    ax.plot(
        markers["A1"]["x_khz"],
        markers["A1"]["y"],

        marker=DOS_MARKER_SHAPE_A1,
        markersize=DOS_MARKER_SIZE_A1,

        markerfacecolor=
            DOS_MARKER_COLOR_A1,

        markeredgecolor=
            DOS_MARKER_EDGE_COLOR,

        markeredgewidth=
            DOS_MARKER_EDGE_WIDTH,

        linestyle="None",
        clip_on=False,

        zorder=10,
    )


    ax.plot(
        markers["B8"]["x_khz"],
        markers["B8"]["y"],

        marker=DOS_MARKER_SHAPE_B8,
        markersize=DOS_MARKER_SIZE_B8,

        markerfacecolor=
            DOS_MARKER_COLOR_B8,

        markeredgecolor=
            DOS_MARKER_EDGE_COLOR,

        markeredgewidth=
            DOS_MARKER_EDGE_WIDTH,

        linestyle="None",
        clip_on=False,

        zorder=11,
    )


    ax.set_xlim(
        FREQ_MIN / 1000.0,
        FREQ_MAX / 1000.0,
    )

    ax.set_ylim(
        0,
        1.05,
    )


    ax.set_xlabel(
        r"Frequency (kHz)",
        fontsize=30,
        labelpad=8,
    )

    ax.set_ylabel(
        r"$|P|^2$ (a.u.)",
        fontsize=30,
        labelpad=15,
    )


    ax.xaxis.set_major_locator(
        MultipleLocator(0.1)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )


    ax.yaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )


    ax.minorticks_off()


    ax.tick_params(
        axis="both",
        which="major",

        direction="out",

        length=7,
        width=1.6,
        pad=9,

        bottom=True,
        left=True,
        top=False,
        right=False,

        labelsize=23,
    )


    for spine in ax.spines.values():

        spine.set_visible(
            True
        )

        spine.set_linewidth(
            DOS_FRAME_LINE_WIDTH
        )

        spine.set_color(
            "black"
        )


    ax.grid(
        False
    )


    if DOS_PANEL_LABEL is not None:

        ax.text(
            0.025,
            0.96,

            DOS_PANEL_LABEL,

            transform=ax.transAxes,

            ha="left",
            va="top",

            fontsize=29,
            fontweight="bold",
        )


    legend_handles = [
        Patch(
            facecolor=DOS_COLOR_RED,
            edgecolor="none",
            alpha=DOS_RED_ALPHA,
            label="Doublon",
        ),

        Patch(
            facecolor=DOS_COLOR_BLUE,
            edgecolor="none",
            alpha=DOS_BLUE_ALPHA,
            label="Bulk",
        ),
    ]


    ax.legend(
        handles=legend_handles,

        loc="upper left",

        frameon=False,
        ncol=1,

        handlelength=2.4,
        handleheight=1.0,

        handletextpad=0.8,
        labelspacing=0.65,

        borderaxespad=0.8,

        fontsize=17,
    )


def parse_field_filename(
    file_path
):
    stem = file_path.stem

    if "-" not in stem:
        return None


    _, receiver_name = (
        stem.split(
            "-",
            1,
        )
    )


    match = SITE_PATTERN.match(
        receiver_name
    )

    if match is None:
        return None


    layer = (
        match.group(1)
        .lower()
    )

    col_letter = (
        match.group(2)
        .upper()
    )

    row_num = int(
        match.group(3)
    )


    col_idx = (
        COLUMNS.index(
            col_letter
        )
    )

    row_idx = (
        row_num - 1
    )


    return (
        layer,
        col_idx,
        row_idx,
        receiver_name,
    )


def read_one_field_file(
    file_path,
    target_freq,
):
    df = pd.read_csv(
        file_path,

        sep=r"\s+",

        header=None,
        comment="#",

        engine="python",
    )


    if (
        df.ndim != 2
        or df.shape[1] <= PRESSURE_DB_COL
    ):
        raise ValueError(
            f"{file_path.name} "
            "的数据列数不足。"
        )


    freq_raw = (
        df.iloc[:, 0]
        .to_numpy(
            dtype=float
        )
    )


    freq = (
        fix_frequency_column(
            freq_raw
        )
    )


    p_db = (
        df.iloc[
            :,
            PRESSURE_DB_COL,
        ]
        .to_numpy(
            dtype=float
        )
    )


    p_pa = (
        db_to_pa(
            p_db
        )
    )


    idx = int(
        np.argmin(
            np.abs(
                freq
                - target_freq
            )
        )
    )


    return (
        float(freq[idx]),
        float(p_pa[idx]),
        float(p_db[idx]),
    )


def mix_with_white(
    color,
    mix=0.12,
):
    rgb = np.array(
        to_rgb(
            color
        )
    )

    white = np.ones(
        3
    )

    return tuple(
        (1 - mix) * white
        + mix * rgb
    )


def make_monochrome_cmap(
    base_color,
    cmap_name,
):
    light_color = (
        mix_with_white(
            base_color,
            mix=0.12,
        )
    )

    return (
        LinearSegmentedColormap
        .from_list(
            cmap_name,
            [
                light_color,
                base_color,
            ],
        )
    )


def load_field_dataset(
    dataset_name,
    data_dir,
    target_freq,
    zero_source=False,
    source_layer=None,
    source_site=None,
):

    pressure = {
        "upper":
            np.full(
                (10, 10),
                np.nan,
            ),

        "lower":
            np.full(
                (10, 10),
                np.nan,
            ),
    }


    if not data_dir.exists():

        raise FileNotFoundError(
            f"[{dataset_name}] "
            f"文件夹不存在："
            f"{data_dir}"
        )


    txt_files = sorted(
        data_dir.glob(
            "*.txt"
        )
    )


    if len(txt_files) == 0:

        raise FileNotFoundError(
            f"[{dataset_name}] "
            f"文件夹中没有 txt："
            f"{data_dir}"
        )


    for file_path in txt_files:

        receiver_info = (
            parse_field_filename(
                file_path
            )
        )


        if receiver_info is None:

            continue


        (
            layer,
            col_idx,
            row_idx,
            _,
        ) = receiver_info


        try:

            (
                _,
                p_pa,
                _,
            ) = read_one_field_file(
                file_path,
                target_freq,
            )


        except Exception:

            continue


        pressure[
            layer
        ][
            row_idx,
            col_idx,
        ] = p_pa


    for layer in [
        "upper",
        "lower",
    ]:

        if np.any(
            ~np.isfinite(
                pressure[layer]
            )
        ):
            raise ValueError(
                f"[{dataset_name}] Missing field data in {layer} layer."
            )

    # Zero the selected source point before normalizing the full field.
    if zero_source:

        if (
            source_layer is None
            or source_site is None
        ):
            raise ValueError(
                f"[{dataset_name}] "
                "zero_source=True，"
                "但没有指定 source_layer/source_site。"
            )


        source_row_idx, source_col_idx = (
            site_to_indices(
                source_site
            )
        )


        pressure[
            source_layer
        ][
            source_row_idx,
            source_col_idx,
        ] = 0.0


    all_values_raw = np.concatenate([
        pressure["upper"][
            np.isfinite(
                pressure["upper"]
            )
        ],

        pressure["lower"][
            np.isfinite(
                pressure["lower"]
            )
        ],
    ])


    if len(all_values_raw) == 0:

        raise RuntimeError(
            f"[{dataset_name}] "
            "没有读取到有效声压。"
        )


    # Normalize upper and lower layers together with one L2 norm.
    normalization_factor = np.sqrt(
        np.sum(
            all_values_raw ** 2
        )
    )


    if (
        not np.isfinite(
            normalization_factor
        )
        or normalization_factor <= 0
    ):
        raise RuntimeError(
            f"[{dataset_name}] "
            "归一化因子无效。"
        )


    pressure_norm = {
        "upper":
            pressure["upper"]
            / normalization_factor,

        "lower":
            pressure["lower"]
            / normalization_factor,
    }


    all_values_norm = np.concatenate([
        pressure_norm["upper"][
            np.isfinite(
                pressure_norm["upper"]
            )
        ],

        pressure_norm["lower"][
            np.isfinite(
                pressure_norm["lower"]
            )
        ],
    ])


    normalization_check = np.sum(
        all_values_norm ** 2
    )


    vmax = float(
        np.max(
            all_values_norm
        )
    )


    zmax = (
        vmax
        * ZMAX_SCALE
    )


    if zmax <= 0:
        zmax = 1.0


    norm = Normalize(
        vmin=0,
        vmax=vmax,
    )


    if zero_source:

        source_row_idx, source_col_idx = (
            site_to_indices(
                source_site
            )
        )


    return {
        "pressure_norm":
            pressure_norm,

        "normalization_factor":
            normalization_factor,

        "vmax":
            vmax,

        "zmax":
            zmax,

        "norm":
            norm,
    }


def make_cylinder_side_facecolors(
    base_color,
    theta,
):
    base_rgb = np.asarray(
        to_rgb(base_color),
        dtype=float,
    )


    theta_mid = 0.5 * (
        theta[:-1]
        + theta[1:]
    )


    light_azimuth_rad = np.deg2rad(
        CYLINDER_LIGHT_AZIMUTH_DEG
    )


    light_profile = 0.5 * (
        1.0
        + np.cos(
            theta_mid
            - light_azimuth_rad
        )
    )

    brightness = (
        CYLINDER_SIDE_SHADE_MIN
        + (
            CYLINDER_SIDE_SHADE_MAX
            - CYLINDER_SIDE_SHADE_MIN
        )
        * light_profile
    )

    facecolors = np.empty(
        (
            1,
            len(theta_mid),
            4,
        ),
        dtype=float,
    )

    for idx, factor in enumerate(
        brightness
    ):
        rgb = np.clip(
            base_rgb * factor,
            0.0,
            1.0,
        )

        facecolors[
            0,
            idx,
            :3,
        ] = rgb

        facecolors[
            0,
            idx,
            3,
        ] = 1.0

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
    if (
        not np.isfinite(height)
        or height <= 0
    ):
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
        np.full_like(
            theta,
            height,
        ),
    ])

    side_facecolors = (
        make_cylinder_side_facecolors(
            color,
            theta,
        )
    )

    side_surface = ax.plot_surface(
        side_x,
        side_y,
        side_z,

        facecolors=
            side_facecolors,

        linewidth=0,
        antialiased=True,


        shade=False,

        zorder=10,
    )


    side_surface.set_clip_on(
        False
    )

    side_surface.set_clip_path(
        None
    )


    top_vertices = [
        (
            center_x
            + radius * np.cos(angle),

            center_y
            + radius * np.sin(angle),

            height,
        )
        for angle in theta[:-1]
    ]


    top_color = mix_with_white(
        color,
        mix=
            CYLINDER_TOP_BRIGHTEN_MIX,
    )

    top_surface = Poly3DCollection(
        [
            top_vertices
        ],

        facecolor=
            top_color,

        edgecolor="none",

        zorder=11,
    )

    top_surface.set_clip_on(
        False
    )

    top_surface.set_clip_path(
        None
    )

    ax.add_collection3d(
        top_surface
    )

def style_3d_axis(ax):
    ax.set_facecolor(
        (1, 1, 1, 0)
    )

    ax.patch.set_facecolor(
        (1, 1, 1, 0)
    )

    ax.patch.set_alpha(
        0
    )


    ax.grid(
        False
    )


    for axis in [
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ]:

        axis.pane.fill = False

        axis.pane.set_edgecolor(
            (1, 1, 1, 0)
        )

        axis._axinfo[
            "grid"
        ][
            "linewidth"
        ] = 0

        axis._axinfo[
            "grid"
        ][
            "color"
        ] = (
            1,
            1,
            1,
            0,
        )

        axis.line.set_color(
            (1, 1, 1, 0)
        )


    ax.tick_params(
        axis="x",
        which="major",

        labelsize=
            FIELD_AXIS_TICK_FONTSIZE,

        pad=
            FIELD_XY_TICK_PAD,

        length=3,
        width=0.8,
    )


    ax.tick_params(
        axis="y",
        which="major",

        labelsize=
            FIELD_AXIS_TICK_FONTSIZE,

        pad=
            FIELD_XY_TICK_PAD,

        length=3,
        width=0.8,
    )


    ax.tick_params(
        axis="z",
        which="major",

        labelsize=
            FIELD_AXIS_TICK_FONTSIZE,

        pad=
            FIELD_Z_TICK_PAD,

        length=3,
        width=0.8,
    )


def draw_custom_box(
    ax,
    draw_gray_edges=True,
    draw_black_edges=True,
):
    x0, x1 = sorted(
        ax.get_xlim3d()
    )

    y0, y1 = sorted(
        ax.get_ylim3d()
    )

    z0, z1 = sorted(
        ax.get_zlim3d()
    )


    point_a = (
        x0,
        y0,
        z0,
    )

    point_b = (
        x1,
        y0,
        z0,
    )

    point_c = (
        x0,
        y1,
        z0,
    )

    point_d = (
        x1,
        y1,
        z0,
    )

    point_e = (
        x0,
        y0,
        z1,
    )

    point_f = (
        x1,
        y0,
        z1,
    )

    point_g = (
        x0,
        y1,
        z1,
    )

    point_h = (
        x1,
        y1,
        z1,
    )


    black_style = {
        "color":
            OUTER_FRAME_COLOR,

        "linewidth":
            1.05,

        "linestyle":
            "-",

        "solid_capstyle":
            "round",
    }


    gray_style = {
        "color":
            INNER_FRAME_COLOR,

        "linewidth":
            0.75,

        "linestyle":
            "-",

        "solid_capstyle":
            "round",
    }


    def draw_edge(
        point_1,
        point_2,
        line_style,
    ):
        ax.plot(
            [
                point_1[0],
                point_2[0],
            ],

            [
                point_1[1],
                point_2[1],
            ],

            [
                point_1[2],
                point_2[2],
            ],

            **line_style,
        )


    if draw_gray_edges:

        draw_edge(
            point_a,
            point_b,
            gray_style,
        )

        draw_edge(
            point_a,
            point_c,
            gray_style,
        )

        draw_edge(
            point_a,
            point_e,
            gray_style,
        )


    if draw_black_edges:

        draw_edge(
            point_b,
            point_d,
            black_style,
        )

        draw_edge(
            point_b,
            point_f,
            black_style,
        )

        draw_edge(
            point_c,
            point_d,
            black_style,
        )

        draw_edge(
            point_c,
            point_g,
            black_style,
        )

        draw_edge(
            point_e,
            point_f,
            black_style,
        )

        draw_edge(
            point_e,
            point_g,
            black_style,
        )


def add_layer_text(
    ax,
    text_string,
    zmax,
):
    ax.text(
        LAYER_TEXT_X,
        LAYER_TEXT_Y,

        zmax
        * LAYER_TEXT_Z_RATIO,

        text_string,

        fontsize=
            LAYER_TEXT_FONTSIZE,

        fontweight="bold",

        ha="left",
        va="center",

        zorder=250,
    )


def draw_field_axis(
    ax,
    grid,
    cmap,
    norm,
    vmax,
    zmax,
    layer_title,
    fixed_layer_zmax=None,
):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


    tick_positions = [
        1,
        5,
        10,
    ]

    tick_labels = [
        "1",
        "5",
        "10",
    ]


    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels
    )


    ax.set_yticks(
        tick_positions
    )

    ax.set_yticklabels(
        tick_labels
    )


    layer_vmax = float(
        np.nanmax(grid)
    )

    if fixed_layer_zmax is not None:
        layer_vmax = float(fixed_layer_zmax)

    layer_zmax = (
        layer_vmax
        * ZMAX_SCALE
    )

    if layer_zmax <= 0:
        layer_zmax = 1.0


    if fixed_layer_zmax is not None:
        ax.set_zticks(
            [0.0, 0.2, 0.4, layer_vmax]
        )
    else:
        ax.set_zticks(
            np.linspace(
                0,
                layer_vmax,
                3,
            )
        )

    ax.zaxis.set_major_formatter(
        FormatStrFormatter(
            "%.1f"
        )
    )


    ax.set_xlim(
        0.5,
        10.5,
    )

    ax.set_ylim(
        0.5,
        10.5,
    )

    ax.set_zlim(
        0,
        layer_zmax,
    )


    ax.view_init(
        elev=VIEW_ELEV,
        azim=VIEW_AZIM,
    )


    ax.invert_yaxis()


    ax.set_box_aspect(
        BOX_ASPECT
    )


    style_3d_axis(
        ax
    )


    draw_custom_box(
        ax,

        draw_gray_edges=True,
        draw_black_edges=False,
    )


    for row_idx in range(10):

        for col_idx in range(10):

            value = grid[
                row_idx,
                col_idx,
            ]


            if not np.isfinite(
                value
            ):
                continue


            center_x = (
                col_idx + 1
            )

            center_y = (
                row_idx + 1
            )


            cylinder_color = (
                cmap(
                    norm(
                        value
                    )
                )
            )


            draw_cylinder(
                ax=ax,

                center_x=
                    center_x,

                center_y=
                    center_y,

                height=
                    value,

                radius=
                    CYLINDER_RADIUS,

                color=
                    cylinder_color,

                n_segments=
                    CYLINDER_SEGMENTS,
            )


    draw_custom_box(
        ax,

        draw_gray_edges=False,
        draw_black_edges=True,
    )


    add_layer_text(
        ax,
        layer_title,
        layer_zmax,
    )


def add_field_title(
    fig,
    center_x,
    title_y,
    frequency,
    coord_label,
    marker_color,
    marker_shape,
):
    marker_x = (
        center_x
        - 0.032
    )

    text_x = (
        center_x
        - 0.010
    )


    marker = Line2D(
        [marker_x],
        [title_y],

        marker=
            marker_shape,

        linestyle=
            "none",

        markersize=
            FIELD_TITLE_MARKER_SIZE,

        markerfacecolor=
            marker_color,

        markeredgecolor=
            "black",

        markeredgewidth=
            1.0,

        transform=
            fig.transFigure,

        clip_on=False,

        zorder=300,
    )


    fig.add_artist(
        marker
    )


    fig.text(
        text_x,
        title_y,

        rf"$f = {frequency:.0f}\ \mathrm{{Hz}}$",

        fontsize=
            FIELD_TITLE_FONTSIZE,

        ha="left",
        va="center",
    )


def add_vertical_colorbar(
    fig,
    position,
    cmap,
    norm,
):
    scalar_mappable = (
        ScalarMappable(
            norm=norm,
            cmap=cmap,
        )
    )


    scalar_mappable.set_array(
        []
    )


    cax = fig.add_axes(
        position
    )


    colorbar = fig.colorbar(
        scalar_mappable,

        cax=cax,

        orientation=
            "vertical",
    )


    colorbar.set_ticks([
        0,
        norm.vmax,
    ])


    colorbar.set_ticklabels([
        "Min",
        "Max",
    ])


    colorbar.ax.tick_params(
        labelsize=15,
        width=0.9,
        length=2.5,
        pad=3,
    )


    colorbar.outline.set_linewidth(
        1.0
    )


    text_x = (
        position[0]
        + position[2] / 2
    )

    text_y = (
        position[1]
        + position[3]
        + FIELD_CBAR_TITLE_Y_OFFSET
    )


    fig.text(
        text_x,
        text_y,

        r"$|P|$",

        fontsize=
            FIELD_CBAR_TITLE_FONTSIZE,

        ha="center",
        va="bottom",
    )


    return colorbar


def draw_field_group(
    fig,
    left,
    row_bottom,
    title_y,
    result,
    config,
):

    center_x = (
        left
        + GROUP_WIDTH / 2
    )


    ax_upper = fig.add_axes(
        [
            left,
            row_bottom,
            FIELD_SUB_AX_WIDTH,
            FIELD_AX_HEIGHT,
        ],
        projection="3d",
    )


    ax_lower = fig.add_axes(
        [
            left
            + FIELD_SUB_AX_WIDTH
            + FIELD_PAIR_GAP,

            row_bottom,
            FIELD_SUB_AX_WIDTH,
            FIELD_AX_HEIGHT,
        ],
        projection="3d",
    )


    draw_field_axis(
        ax=ax_upper,

        grid=result[
            "pressure_norm"
        ][
            "upper"
        ],

        cmap=result[
            "cmap"
        ],

        norm=result[
            "norm"
        ],

        vmax=result[
            "vmax"
        ],

        zmax=result[
            "zmax"
        ],

        layer_title=
            "Layer A",

        fixed_layer_zmax=None,
    )


    draw_field_axis(
        ax=ax_lower,

        grid=result[
            "pressure_norm"
        ][
            "lower"
        ],

        cmap=result[
            "cmap"
        ],

        norm=result[
            "norm"
        ],

        vmax=result[
            "vmax"
        ],

        zmax=result[
            "zmax"
        ],

        layer_title=
            "Layer B",

        fixed_layer_zmax=config.get(
            "layer_b_zmax",
            None,
        ),
    )


    add_field_title(
        fig=fig,

        center_x=center_x,

        title_y=title_y,

        frequency=result[
            "target_freq"
        ],

        coord_label=config.get(
            "coord_label",
            "",
        ),

        marker_color=config[
            "base_color"
        ],

        marker_shape=config[
            "title_marker"
        ],
    )


    group_center_y = (
        row_bottom
        + FIELD_AX_HEIGHT / 2
    )


    cbar_position = [
        left
        + GROUP_WIDTH
        + FIELD_CBAR_OFFSET_X,

        group_center_y
        - FIELD_CBAR_LENGTH / 2,

        FIELD_CBAR_THICKNESS,

        FIELD_CBAR_LENGTH,
    ]


    add_vertical_colorbar(
        fig=fig,

        position=
            cbar_position,

        cmap=result[
            "cmap"
        ],

        norm=result[
            "norm"
        ],
    )


def main():


    dos_data = load_dos_data()


    field_results = {}

    for dataset_name, config in FIELD_DATASETS.items():

        target_freq = config["target_freq"]

        result = load_field_dataset(
            dataset_name=dataset_name,
            data_dir=config["data_dir"],
            target_freq=target_freq,
            zero_source=config["zero_source"],
            source_layer=config["source_layer"],
            source_site=config["source_site"],
        )

        result["target_freq"] = target_freq
        result["cmap"] = make_monochrome_cmap(
            base_color=config["base_color"],
            cmap_name=f"{dataset_name}_mono",
        )

        field_results[dataset_name] = result


    fig_dos = plt.figure(
        figsize=DOS_FIGSIZE,
        facecolor="white",
    )

    ax_dos = fig_dos.add_axes(DOS_AX_POSITION)
    draw_dos_panel(ax_dos, dos_data)


    fig_field = plt.figure(
        figsize=FIELD_FIGSIZE,
        facecolor="white",
    )

    field_order = [
        "A1",
        "B8",
    ]

    for group_index, dataset_name in enumerate(field_order):

        draw_field_group(
            fig=fig_field,
            left=GROUP_LEFTS[group_index],
            row_bottom=FIELD_ROW_BOTTOM,
            title_y=FIELD_TITLE_Y,
            result=field_results[dataset_name],
            config=FIELD_DATASETS[dataset_name],
        )


    plt.show()


if __name__ == "__main__":
    main()
