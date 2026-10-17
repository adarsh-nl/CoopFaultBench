# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""ONE visual language for every released figure.

THREE RULES, in priority order.

1. THE SAME COLOUR MEANS THE SAME MODEL (or the same injector) IN EVERY
   FIGURE IT APPEARS IN. Two palettes are defined -- one keyed by model, one
   by injector -- and nothing else may assign a colour.

2. COLOUR IS NEVER THE ONLY DISCRIMINATOR. Every series also carries a
   distinct marker, linestyle and linewidth, so the figure survives greyscale
   printing and the common forms of colour blindness.

3. ONE SCALE. Font sizes, linewidths and legend style come from here, not
   from per-figure rcParams.

The model palette is Okabe-Ito, which is designed for deuteranopia,
protanopia and tritanopia and stays distinguishable when desaturated. The
injector palette adds two Okabe-Ito hues and pairs the two LiDAR-weather
injectors as a dark/light blue so they read as a family.

``figure_style()`` installs the scale inside a ``matplotlib.rc_context``
starting from matplotlib's built-in defaults, so the output does not depend on
a user's matplotlibrc or on which figure family was drawn first.
"""
from __future__ import annotations

import contextlib

import matplotlib

# ---------------------------------------------------------------- palettes
# Okabe-Ito. Grey is reserved for reference lines, never for a series.
_OI = {
    'blue':      '#0072B2',
    'skyblue':   '#56B4E9',
    'green':     '#009E73',
    'vermilion': '#D55E00',
    'orange':    '#E69F00',
    'purple':    '#CC79A7',
    'yellow':    '#F0E442',
    'black':     '#000000',
}

REFERENCE_GREY = '#444444'
GRID_GREY = '#DDDDDD'

# One colour per model, everywhere.
OP_COLOR = {
    'cobevt':     _OI['blue'],
    'v2xvit':     _OI['vermilion'],
    'where2comm': _OI['green'],
    'cosdh':      _OI['purple'],
}
# Marker, linestyle, linewidth -- the greyscale-safe discriminators.
OP_MARKER = {'cobevt': 'o', 'v2xvit': 's', 'where2comm': '^', 'cosdh': 'D'}
OP_LS = {'cobevt': '-', 'v2xvit': (0, (1, 1.2)), 'where2comm': (0, (4, 1.5)),
         'cosdh': (0, (5, 1.2, 1, 1.2))}
OP_LW = {'cobevt': 1.5, 'v2xvit': 1.2, 'where2comm': 1.5, 'cosdh': 1.2}
OP_LABEL = {'cobevt': 'CoBEVT', 'v2xvit': 'V2X-ViT',
            'where2comm': 'Where2comm', 'cosdh': 'CoSDH'}

# One colour per injector, everywhere.
INJ_COLOR = {
    'lidar_snow':       _OI['blue'],
    'lidar_fog':        _OI['skyblue'],
    'points_reduce':    _OI['green'],
    'pose_error':       _OI['vermilion'],
    'latency':          _OI['orange'],
    'missing_modality': _OI['purple'],
    'agent_drop':       _OI['black'],
}
INJ_MARKER = {'lidar_snow': 'o', 'lidar_fog': 'v', 'points_reduce': 's',
              'pose_error': 'D', 'latency': '^', 'missing_modality': 'P',
              'agent_drop': 'X'}
INJ_LS = {'lidar_snow': '-', 'lidar_fog': (0, (4, 1.5)),
          'points_reduce': (0, (1, 1.2)), 'pose_error': (0, (5, 1.2, 1, 1.2)),
          'latency': (0, (3, 1, 1, 1, 1, 1)), 'missing_modality': '-',
          'agent_drop': (0, (2, 1))}
INJ_LABEL = {'lidar_snow': 'LiDAR snow', 'lidar_fog': 'LiDAR fog',
             'points_reduce': 'Points reduce', 'pose_error': 'Pose error',
             'latency': 'Comm. latency',
             'missing_modality': 'Missing LiDAR', 'agent_drop': 'Agent drop'}

TIER_ALPHA = {'mild': 0.45, 'moderate': 0.72, 'severe': 1.0}

# ------------------------------------------------------------------ scales
COL_W = 3.25          # single column, inches
FULL_W = 6.9          # double column, inches

FS = {                # one font scale
    'base': 8, 'label': 8, 'title': 8, 'panel': 7.5,
    'tick': 7, 'legend': 7, 'annot': 6.5, 'small': 6,
}
LW = {                # one linewidth scale
    'axes': 0.6, 'series': 1.5, 'thin': 1.0, 'reference': 0.8, 'edge': 0.5,
}

# Extra rcParams used only by the grid figure family (agent drop by stratum,
# severe-tier ordering, all-model degradation curves): no top/right spines,
# thin grid, slightly larger default markers.
GRID_FAMILY_RC = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linewidth': 0.4, 'lines.markersize': 3.5,
}


def base_rc(dpi=400):
    """The one scale as an rcParams dict."""
    return {
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': FS['base'],
        'axes.labelsize': FS['label'], 'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'axes.linewidth': LW['axes'],
        'xtick.major.width': LW['axes'], 'ytick.major.width': LW['axes'],
        'figure.dpi': dpi, 'savefig.dpi': dpi,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'legend.frameon': False, 'legend.handlelength': 1.8,
        'legend.columnspacing': 1.1, 'legend.handletextpad': 0.6,
        'axes.grid': False,
    }


@contextlib.contextmanager
def figure_style(extra=None):
    """Context in which a figure family is built AND saved: matplotlib's
    built-in defaults, then the one scale, then ``extra`` (if any). The
    caller's rcParams are restored on exit."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(base_rc())
        if extra:
            matplotlib.rcParams.update(extra)
        yield


def op_style(op, **over):
    """Every kwarg needed to draw one model's series, consistently."""
    d = {'color': OP_COLOR[op], 'marker': OP_MARKER[op],
         'linestyle': OP_LS[op], 'linewidth': OP_LW[op],
         'markersize': 3.2, 'label': OP_LABEL[op]}
    d.update(over)
    return d


def inj_style(inj, **over):
    """Every kwarg needed to draw one injector's series, consistently."""
    d = {'color': INJ_COLOR[inj], 'marker': INJ_MARKER[inj],
         'linestyle': INJ_LS[inj], 'linewidth': LW['series'],
         'markersize': 3.0, 'label': INJ_LABEL[inj]}
    d.update(over)
    return d


def save_pdf(fig, path):
    """Write one PDF with the CreationDate stamp suppressed, so regeneration
    is byte-identical for identical inputs and library versions."""
    fig.savefig(path, metadata={'CreationDate': None})
    return path
