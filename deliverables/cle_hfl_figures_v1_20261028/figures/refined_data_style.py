"""Style for the separately versioned Figure 4/5 revision candidates.

Does not modify figure_common.py or any previously rendered figure.
TeX fonts match the Figure 1/2 TikZ and Figure 3 refined candidates.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ['PATH'] = 'D:/texlive/2026/bin/windows' + os.pathsep + os.environ.get('PATH', '')
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parents[3] / 'local_runs' / 'mplconfig_figure3_refined')

import matplotlib as mpl

ACCENT = '#2F6A7F'
NEUTRAL = '#69747C'


def setup_style() -> None:
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'font.size': 8,
        'text.usetex': True, 'text.latex.preamble': r'\usepackage{lmodern}',
        'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'axes.edgecolor': '#CBD2D8', 'axes.linewidth': 0.7,
        'text.color': '#20252B', 'axes.labelcolor': '#20252B',
        'xtick.color': '#66727D', 'ytick.color': '#20252B',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'CLE-HFL-refined-data-20261030', 'savefig.facecolor': 'white',
    })
