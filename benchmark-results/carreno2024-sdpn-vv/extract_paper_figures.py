"""Reference curves of Carreno et al. (2024) Figs. 4, 6 and 7, read from the PDF.

    python extract_paper_figures.py PAPER.pdf data/paper-figures.json   # needs PyMuPDF

The paper's plots are matplotlib vector graphics, so the curves are the drawn
paths themselves, not a pixel digitization. Each curve is a stroked path in a
known colour; in Fig. 7b the strokes were converted to filled outlines, whose
centre line is taken bin by bin (0.05 cm). Fig. 4 shows the paper's SDP3, the
SDP3 of Nazari et al. and SP7 as markers; their centres are kept as points,
except the legend samples (x < 3.2 cm, y < 0.65).

Data coordinates come from a least-squares line through the tick marks (1.45 pt
strokes in RGB 0.13 gray). The tick labels are glyph outlines in the PDF, so
their values are transcribed below from the rendered page. The recorded
uncertainty is the fit residual plus half the drawn line width, in data units.
"""

import json
import sys

import numpy as np
import pymupdf

BLUE, ORANGE, GREEN, RED = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16)
TICK = (0.13, 0.13, 0.13)
DIAG = "distance along the diagonal (cm)"

# page (0-based), region (pt), transcribed tick labels, curves {name: colour},
# markers {name: colour}, legend region to drop (pt).
FIGS = {
    "fig4": dict(page=7, box=(60, 455, 290, 640), x=[0, 2, 4, 6, 8, 10],
                 y=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75],
                 curves={"OpenMOC": RED},
                 markers={"SDP3": BLUE, "SDP3 (Nazari et al.)": ORANGE, "SP7": GREEN},
                 xlabel="x (cm) at y = 4.5 cm", ylabel="scalar flux (AU)"),
    "fig6a": dict(page=8, box=(110, 380, 295, 510), x=[0, 5, 10, 15, 20, 25, 30],
                  y=[50, 60, 70, 80, 90],
                  curves={"OpenMC": BLUE, "SDP1": ORANGE, "SDP2": GREEN, "SDP3": RED},
                  xlabel=DIAG, ylabel="group 1 scalar flux (AU)"),
    "fig6b": dict(page=8, box=(305, 380, 490, 510), x=[0, 5, 10, 15, 20, 25, 30],
                  y=[10, 20, 30, 40, 50],
                  curves={"OpenMC": BLUE, "SDP1": ORANGE, "SDP2": GREEN, "SDP3": RED},
                  xlabel=DIAG, ylabel="group 7 scalar flux (AU)"),
    "fig7a": dict(page=8, box=(110, 580, 295, 712), x=[0, 5, 10, 15, 20, 25, 30],
                  y=[10, 15, 20, 25, 30, 35, 40, 45, 50],
                  curves={"OpenMC": BLUE, "SDP1": ORANGE, "SDP2": GREEN, "SDP3": RED},
                  xlabel=DIAG, ylabel="group 1 scalar flux (AU)"),
    "fig7b": dict(page=8, box=(305, 565, 490, 700), x=[0, 5, 10, 15, 20, 25, 30],
                  y=[2, 4, 6, 8, 10, 12, 14, 16],
                  curves={"OpenMC": BLUE, "SDP1": ORANGE, "SDP2": GREEN, "SDP3": RED},
                  xlabel=DIAG, ylabel="group 7 scalar flux (AU)"),
}


def _rgb(c):
    return tuple(round(v, 2) for v in c) if c else None


def _inside(r, box):
    return box[0] <= r.x0 and r.x1 <= box[2] and box[1] <= r.y0 and r.y1 <= box[3]


def _points(items):
    pts = []
    for it in items:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "c":
            pts += [it[1], it[4]]
    return np.array([(p.x, p.y) for p in pts])


def _fit(pos, vals):
    """value = a + b * pos, with the largest residual."""
    b, a = np.polyfit(pos, vals, 1)
    return a, b, float(np.max(np.abs(a + b * np.asarray(pos) - vals)))


def extract(doc, spec):
    page = doc[spec["page"]]
    box = spec["box"]
    drawings = [d for d in page.get_drawings() if _inside(d["rect"], box)]
    xt = sorted({round(d["rect"].x0, 3) for d in drawings if _rgb(d.get("color")) == TICK
                 and d["rect"].width < 1e-3 and 1.0 < d["rect"].height < 2.0})
    yt = sorted({round(d["rect"].y0, 3) for d in drawings if _rgb(d.get("color")) == TICK
                 and d["rect"].height < 1e-3 and 1.0 < d["rect"].width < 2.0}, reverse=True)
    if len(xt) != len(spec["x"]) or len(yt) != len(spec["y"]):
        raise RuntimeError(f"tick count mismatch: {len(xt)} x, {len(yt)} y")
    ax, bx, rx = _fit(xt, spec["x"])
    ay, by, ry = _fit(yt, spec["y"])
    out = dict(xlabel=spec["xlabel"], ylabel=spec["ylabel"],
               calibration=dict(x=[ax, bx, rx], y=[ay, by, ry]), curves={}, markers={})
    widths = []
    for name, colour in spec["curves"].items():
        strokes = [d for d in drawings if _rgb(d.get("color")) == colour and d["type"] == "s"
                   and len(d["items"]) > 2]
        fills = [d for d in drawings if _rgb(d.get("fill")) == colour and d["type"] == "f"
                 and len(d["items"]) > 2]
        if strokes:
            pts = np.vstack([_points(d["items"]) for d in strokes])
            widths += [d.get("width") or 0.0 for d in strokes]
            x, y = ax + bx * pts[:, 0], ay + by * pts[:, 1]
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
            kind = "stroke"
        else:
            pts = np.vstack([_points(d["items"]) for d in fills])
            x, y = ax + bx * pts[:, 0], ay + by * pts[:, 1]
            edges = np.arange(x.min(), x.max() + 0.05, 0.05)
            idx = np.digitize(x, edges)
            xs, ys = [], []
            for k in np.unique(idx):
                sel = idx == k
                xs.append(x[sel].mean())
                ys.append(0.5 * (y[sel].min() + y[sel].max()))
            x, y = np.array(xs), np.array(ys)
            kind = "filled outline, bin centre line"
        out["curves"][name] = dict(x=np.round(x, 5).tolist(), y=np.round(y, 5).tolist(), source=kind)
    for name, colour in spec.get("markers", {}).items():
        marks = [d for d in drawings if (_rgb(d.get("fill")) == colour or _rgb(d.get("color")) == colour)
                 and d["rect"].width < 6 and d["rect"].height < 6]
        centres = np.array([((d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2)
                            for d in marks])
        # A marker may be drawn as a fill and an edge: one point per centre.
        centres = np.unique(np.round(centres, 2), axis=0)
        x, y = ax + bx * centres[:, 0], ay + by * centres[:, 1]
        keep = ~((x < 3.2) & (y < 0.65))    # the legend box
        order = np.argsort(x[keep])
        out["markers"][name] = dict(x=np.round(x[keep][order], 5).tolist(),
                                    y=np.round(y[keep][order], 5).tolist())
    half = 0.5 * max(widths) if widths else 0.413
    out["uncertainty"] = dict(x=rx + half * abs(bx), y=ry + half * abs(by),
                              note="tick-fit residual plus half the line width")
    return out


def main(pdf, output):
    doc = pymupdf.open(pdf)
    data = dict(source=dict(paper="Carreno et al., Ann. Nucl. Energy 207 (2024) 110675",
                            file=pdf.split("/")[-1], method=__doc__.split("\n\n")[1]),
                figures={k: extract(doc, v) for k, v in FIGS.items()})
    with open(output, "w") as fh:
        json.dump(data, fh, indent=1)
    for k, f in data["figures"].items():
        print(k, {n: len(c["x"]) for n, c in f["curves"].items()},
              {n: len(c["x"]) for n, c in f["markers"].items()},
              "unc x %.3g y %.3g" % (f["uncertainty"]["x"], f["uncertainty"]["y"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
