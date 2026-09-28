"""
VARNIKA — Evidence plot: encoder-processor-decoder architecture diagram
(for Slide 2 "conceptual GNN contribution" / Slide 3 "spherical mesh/GNN diagram")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUT_DIR = BASE_DIR / "outputs"


def box(ax, xy, w, h, text, color):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    linewidth=1.5, edgecolor="#1f4e8c", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ((0.3, 1.2), 2.0, 1.6, "Input\n5 features x 66 nodes\nx T_in past hours", "#e8f0fb"),
        ((2.8, 1.2), 2.0, 1.6, "Encoder\nper-node MLP\n-> hidden embedding", "#cfe0f7"),
        ((5.3, 1.2), 2.2, 1.6, "Processor\nGCNConv (mesh from\n01_graph_construction)\n+ GRUCell per node,\nrepeated over T_in", "#9fc2ec"),
        ((8.0, 2.1), 2.4, 1.0, "Decoder A\npredicted next-step\nfeatures (N x 5)", "#f7e0cf"),
        ((8.0, 0.7), 2.4, 1.0, "Decoder B\nthreat embedding\n(N x embed_dim) ->\nhanded to tracker/precursor", "#f7cfcf"),
    ]
    for xy, w, h, text, color in stages:
        box(ax, xy, w, h, text, color)

    arrow_pairs = [
        ((2.3, 2.0), (2.8, 2.0)),
        ((4.8, 2.0), (5.3, 2.0)),
        ((7.5, 2.2), (8.0, 2.6)),
        ((7.5, 1.8), (8.0, 1.2)),
    ]
    for (x0, y0), (x1, y1) in arrow_pairs:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#1f4e8c", lw=1.6))

    ax.set_title("Spatio-temporal GNN prototype — encoder / processor / decoder\n"
                 "(architecture in the lineage of GraphCast / MeshGraphNets, scaled to a 66-node regional mesh)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/gnn_architecture_diagram.png", dpi=160)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR}/gnn_architecture_diagram.png")


if __name__ == "__main__":
    main()
