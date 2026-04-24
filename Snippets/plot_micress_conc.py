"""Plot a MICRESS concentration field (.conc1) using micpy.

Usage:
    python plot_micress_conc.py <conc1_file> [geof_file] [output.png]

Examples:
    python plot_micress_conc.py data.conc1
    python plot_micress_conc.py data.conc1 geometry.geof
    python plot_micress_conc.py data.conc1 geometry.geof result.png

Install:
    pip install micress-micpy matplotlib
"""

import sys
from pathlib import Path
from micpy import bin, geo


def plot_conc(
    conc_path: str,
    geof_path: str = None,
    output_path: str = "output.png",
):
    with bin.File(conc_path) as f:
        if geof_path is not None:
            geometry = geo.read(geof_path, type=geo.Type.BASIC)
            f.shape = tuple(geometry["shape"][::-1])
            f.spacing = tuple(geometry["spacing"][::-1])

        field = f.read(-1)
        fig, ax, cbar = field.plot()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <conc1_file> [geof_file] [output.png]")
        sys.exit(1)

    conc = sys.argv[1]
    geof = None
    out = "output.png"

    for arg in sys.argv[2:]:
        if arg.lower().endswith((".geof", ".geof1")):
            geof = arg
        else:
            out = arg

    plot_conc(conc, geof, out)
