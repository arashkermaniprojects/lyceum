"""Canonical-topic SVG generators.

Each generator returns a tuple ``(svg_body, width, height, params_used)``
and accepts a ``seed`` so the regen-on-rejection loop can perturb data.
Output is hand-rolled SVG (no matplotlib) — small, deterministic, fast.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .render import SVGCanvas, Axes, PALETTE, _esc


@dataclass
class GenResult:
    svg_body: str
    width: float
    height: float
    params: dict
    title: str


# ---------------------------------------------------------------------------
# 1. Overfitting curve — train error monotone↓, test error U-shape
# ---------------------------------------------------------------------------

def overfitting_curve(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 480.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=0.5, x_hi=10.5, y_lo=0.0, y_hi=1.05,
        x_label="model complexity (e.g. polynomial degree)",
        y_label="error",
        title="Overfitting: training vs. test error",
    )
    ax.draw_frame(x_ticks=5, y_ticks=4)
    xs = [d for d in range(1, 11)]
    # Training error: monotonically decreasing toward zero.
    train = [0.85 * math.exp(-0.45 * (d - 1)) + rng.uniform(-0.01, 0.01)
             for d in xs]
    # Test error: U-shaped (high underfit on left, rises right).
    test = [0.85 * math.exp(-0.6 * (d - 1)) + 0.04 * (d - 3) ** 2
            + rng.uniform(-0.015, 0.015) for d in xs]
    ax.polyline(zip(xs, train), color=PALETTE["train"], width=2.4,
                label="training error", label_pos=(7.2, train[6] + 0.05))
    ax.polyline(zip(xs, test), color=PALETTE["test"], width=2.4,
                label="test error", label_pos=(7.0, test[6] + 0.06))
    ax.scatter(zip(xs, train), color=PALETTE["train"])
    ax.scatter(zip(xs, test), color=PALETTE["test"])
    # Mark the sweet-spot complexity.
    best_x = xs[min(range(len(test)), key=lambda i: test[i])]
    ax.vline(best_x, color="#9e9e9e", label="sweet spot")
    # Region labels.
    canvas.add(
        f'<text x="{ax.to_px(2):.1f}" y="{ax.to_py(0.92):.1f}" '
        f'fill="{PALETTE["muted"]}" font-size="11" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">underfit</text>'
    )
    canvas.add(
        f'<text x="{ax.to_px(8.5):.1f}" y="{ax.to_py(0.92):.1f}" '
        f'fill="{PALETTE["muted"]}" font-size="11" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">overfit</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "best_complexity": best_x},
        title="Overfitting curve",
    )


# ---------------------------------------------------------------------------
# 2. Bias–variance decomposition — bias², variance, total error
# ---------------------------------------------------------------------------

def bias_variance(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 480.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=0.5, x_hi=10.5, y_lo=0.0, y_hi=1.05,
        x_label="model complexity",
        y_label="error",
        title="Bias–variance decomposition",
    )
    ax.draw_frame()
    xs = list(range(1, 11))
    bias = [0.95 * math.exp(-0.55 * (d - 1)) for d in xs]
    var = [0.06 * (d - 1) + 0.01 * (d - 1) ** 2 for d in xs]
    irred = 0.05
    total = [b + v + irred for b, v in zip(bias, var)]
    ax.polyline(zip(xs, bias), color=PALETTE["bias"], width=2.2,
                label="bias²", label_pos=(7.5, bias[6] + 0.04))
    ax.polyline(zip(xs, var), color=PALETTE["variance"], width=2.2,
                label="variance", label_pos=(7.5, var[6] + 0.04))
    ax.polyline(zip(xs, total), color=PALETTE["total"], width=2.6,
                label="total error", label_pos=(2.0, total[1] + 0.06))
    # Irreducible error reference line.
    ax.polyline([(0.5, irred), (10.5, irred)], color=PALETTE["muted"],
                dash="3 3", width=1.0)
    canvas.add(
        f'<text x="{ax.to_px(8.4):.1f}" y="{ax.to_py(irred + 0.02):.1f}" '
        f'fill="{PALETTE["muted"]}" font-size="11" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">irreducible σ²</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Bias-variance decomposition",
    )


# ---------------------------------------------------------------------------
# 3. ROC curve — true vs. false positive rate
# ---------------------------------------------------------------------------

def roc_curve(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 360.0, 320.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0,
        x_label="false positive rate",
        y_label="true positive rate",
        title="ROC curve",
    )
    ax.draw_frame(x_ticks=5, y_ticks=5)
    # Diagonal random-guess reference.
    ax.polyline([(0, 0), (1, 1)], color=PALETTE["muted"],
                dash="4 3", width=1.2)
    # ROC for a decent classifier: TPR = 1 - (1-FPR)^k.
    k = 3.0 + rng.uniform(-0.4, 0.4)
    pts = [(t, 1 - (1 - t) ** k) for t in [i / 40 for i in range(0, 41)]]
    ax.polyline(pts, color=PALETTE["train"], width=2.6)
    # AUC ≈ k / (k+1) for this family.
    auc = k / (k + 1)
    canvas.add(
        f'<text x="{ax.to_px(0.55):.1f}" y="{ax.to_py(0.20):.1f}" '
        f'fill="{PALETTE["train"]}" font-size="13" font-weight="600" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">AUC = {auc:.2f}</text>'
    )
    canvas.add(
        f'<text x="{ax.to_px(0.55):.1f}" y="{ax.to_py(0.55):.1f}" '
        f'fill="{PALETTE["muted"]}" font-size="11" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">random guess</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "auc": round(auc, 3)},
        title="ROC curve",
    )


# ---------------------------------------------------------------------------
# 4. k-fold cross-validation — visual of K folds
# ---------------------------------------------------------------------------

def kfold_split(*, seed: int = 0, k: int = 5) -> GenResult:
    rng = random.Random(seed)
    W, H = 520.0, 240.0
    canvas = SVGCanvas(W, H)
    canvas.add(
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">k-fold cross-validation '
        f'(k = {k})</text>'
    )
    pad_l, pad_r, top, bottom = 60.0, 16.0, 40.0, 24.0
    track_w = W - pad_l - pad_r
    row_h = (H - top - bottom) / k
    for i in range(k):
        y = top + i * row_h
        canvas.add(
            f'<text x="{pad_l - 8:.1f}" y="{y + row_h*0.65:.1f}" '
            f'text-anchor="end" font-size="11" fill="{PALETTE["muted"]}" '
            f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">iter {i+1}</text>'
        )
        for j in range(k):
            cell_x = pad_l + (j / k) * track_w
            cell_w = track_w / k - 2
            is_test = (j == i)
            color = PALETTE["test"] if is_test else PALETTE["train"]
            label = "test" if is_test else "train"
            canvas.add(
                f'<rect x="{cell_x:.1f}" y="{y + 4:.1f}" '
                f'width="{cell_w:.1f}" height="{row_h - 8:.1f}" rx="3" '
                f'fill="{color}" opacity="0.85" '
                f'stroke="{PALETTE["axis"]}" stroke-width="0.6"/>'
            )
            canvas.add(
                f'<text x="{cell_x + cell_w/2:.1f}" '
                f'y="{y + row_h/2 + 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#fff" '
                f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">{label}</text>'
            )
    # Legend.
    lx, ly = pad_l, H - 14.0
    canvas.add(
        f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="12" height="9" '
        f'fill="{PALETTE["train"]}"/>'
        f'<text x="{lx + 16:.1f}" y="{ly - 1:.1f}" font-size="11" '
        f'fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">training fold</text>'
    )
    canvas.add(
        f'<rect x="{lx + 110:.1f}" y="{ly - 9:.1f}" width="12" height="9" '
        f'fill="{PALETTE["test"]}"/>'
        f'<text x="{lx + 126:.1f}" y="{ly - 1:.1f}" font-size="11" '
        f'fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">held-out test fold</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "k": k},
        title=f"{k}-fold cross-validation",
    )


# ---------------------------------------------------------------------------
# 5. Gradient descent on a quadratic loss
# ---------------------------------------------------------------------------

def gradient_descent(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 420.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=-2.5, x_hi=2.5, y_lo=-1.8, y_hi=2.0,
        x_label="θ₁", y_label="θ₂",
        title="Gradient descent (level sets of loss)",
    )
    ax.draw_frame(x_ticks=5, y_ticks=4)
    # Draw loss level sets (concentric ellipses around origin).
    for r in [0.4, 0.8, 1.2, 1.6, 2.0]:
        rx = r
        ry = r * 0.75
        canvas.add(
            f'<ellipse cx="{ax.to_px(0):.2f}" cy="{ax.to_py(0):.2f}" '
            f'rx="{(ax.to_px(rx) - ax.to_px(0)):.2f}" '
            f'ry="{(ax.to_py(0) - ax.to_py(ry)):.2f}" fill="none" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1.2"/>'
        )
    # Descent path: θ_{t+1} = θ_t - η ∇f(θ_t), with f = 0.5(x²/a + y²/b).
    a, b = 1.0, (1.0 / 0.75) ** 2
    eta = 0.20 + rng.uniform(-0.02, 0.02)
    x, y = -2.1 + rng.uniform(-0.05, 0.05), 1.6 + rng.uniform(-0.05, 0.05)
    pts = [(x, y)]
    for _ in range(18):
        gx = x / a
        gy = y / b
        x -= eta * gx
        y -= eta * gy
        pts.append((x, y))
        if abs(x) + abs(y) < 1e-3:
            break
    ax.polyline(pts, color=PALETTE["test"], width=2.0)
    ax.scatter(pts, color=PALETTE["test"], r=2.5)
    # Mark the minimum.
    canvas.add(
        f'<circle cx="{ax.to_px(0):.2f}" cy="{ax.to_py(0):.2f}" '
        f'r="3.5" fill="{PALETTE["bias"]}" stroke="#fff"/>'
    )
    canvas.add(
        f'<text x="{ax.to_px(0):.1f}" y="{ax.to_py(0) - 8:.1f}" '
        f'text-anchor="middle" font-size="11" fill="{PALETTE["bias"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">minimum</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "lr": round(eta, 3), "steps": len(pts)},
        title="Gradient descent",
    )


# ---------------------------------------------------------------------------
# 6. Learning curve — error vs training-set size
# ---------------------------------------------------------------------------

def learning_curve(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 480.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=10, x_hi=1000, y_lo=0.0, y_hi=1.0,
        x_label="training set size n",
        y_label="error",
        title="Learning curve",
    )
    ax.draw_frame(x_ticks=5, y_ticks=4)
    sizes = [10, 25, 50, 100, 200, 400, 700, 1000]
    train = [max(0.05, 0.10 + 0.5 * math.exp(-n / 80) + rng.uniform(-0.02, 0.02))
             for n in sizes]
    test = [0.55 - 0.42 * (1 - math.exp(-n / 200)) + rng.uniform(-0.015, 0.015)
            for n in sizes]
    ax.polyline(zip(sizes, train), color=PALETTE["train"], width=2.4,
                label="training error", label_pos=(700, train[-1] - 0.06))
    ax.polyline(zip(sizes, test), color=PALETTE["test"], width=2.4,
                label="validation error", label_pos=(550, test[-1] + 0.06))
    ax.scatter(zip(sizes, train), color=PALETTE["train"])
    ax.scatter(zip(sizes, test), color=PALETTE["test"])
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Learning curve",
    )


# ---------------------------------------------------------------------------
# 7. Regularization path — coefficients vs λ
# ---------------------------------------------------------------------------

def regularization_path(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 480.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=0.0, x_hi=1.0, y_lo=-1.0, y_hi=1.5,
        x_label="λ (regularization strength)",
        y_label="coefficient value",
        title="Regularization path (lasso/ridge)",
    )
    ax.draw_frame(x_ticks=5, y_ticks=5)
    # 6 coefficients shrinking toward zero at different rates.
    initial = [1.3, 0.9, 0.6, -0.7, -0.4, 0.2]
    rates = [3.5, 2.0, 4.0, 2.5, 5.5, 7.0]
    grid = [i / 40 for i in range(41)]
    colors = [PALETTE["train"], PALETTE["test"], PALETTE["bias"],
              PALETTE["variance"], PALETTE["total"], "#00838f"]
    for j, (b0, r) in enumerate(zip(initial, rates)):
        path = [(L, b0 * math.exp(-r * L) + rng.uniform(-0.005, 0.005))
                for L in grid]
        ax.polyline(path, color=colors[j], width=1.8,
                    label=f"β{j+1}", label_pos=(0.04, b0 + 0.05))
    # Zero reference.
    ax.polyline([(0, 0), (1, 0)], color=PALETTE["muted"], dash="3 2",
                width=0.8)
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Regularization path",
    )


# ---------------------------------------------------------------------------
# 9. Multilayer perceptron with forward + backward arrows (back-propagation)
# ---------------------------------------------------------------------------

def back_propagation(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 540.0, 320.0
    canvas = SVGCanvas(W, H)
    canvas.add(
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">Back-propagation '
        f'(forward → loss → gradient ←)</text>'
    )
    layer_sizes = [3, 4, 4, 2]
    layer_labels = ["input", "hidden 1", "hidden 2", "output"]
    pad_l, pad_r, top, bot = 70.0, 80.0, 56.0, 24.0
    plot_w = W - pad_l - pad_r
    plot_h = H - top - bot
    n_layers = len(layer_sizes)
    layer_x = [pad_l + (plot_w * i) / (n_layers - 1)
               for i in range(n_layers)]
    node_pos: list[list[tuple[float, float]]] = []
    for li, n in enumerate(layer_sizes):
        col_y = []
        gap = plot_h / max(1, n + 1)
        for i in range(n):
            y = top + gap * (i + 1)
            col_y.append((layer_x[li], y))
        node_pos.append(col_y)
    # Edges: every node in layer L → every node in L+1.
    for li in range(n_layers - 1):
        for x1, y1 in node_pos[li]:
            for x2, y2 in node_pos[li + 1]:
                canvas.add(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                    f'x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#cfd8dc" stroke-width="0.8"/>'
                )
    # Forward arrow on top of layer 1→2 (illustrative).
    fx1, _ = node_pos[0][0]
    fx2, _ = node_pos[1][0]
    canvas.add(
        f'<defs>'
        f'<marker id="bpfwd" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{PALETTE["train"]}"/></marker>'
        f'<marker id="bpbwd" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{PALETTE["test"]}"/></marker>'
        f'</defs>'
    )
    # Forward arrow above the network.
    canvas.add(
        f'<line x1="{pad_l - 6:.1f}" y1="{top - 16:.1f}" '
        f'x2="{W - pad_r + 6:.1f}" y2="{top - 16:.1f}" '
        f'stroke="{PALETTE["train"]}" stroke-width="2.4" '
        f'marker-end="url(#bpfwd)"/>'
    )
    canvas.add(
        f'<text x="{(pad_l + W - pad_r)/2:.1f}" y="{top - 22:.1f}" '
        f'text-anchor="middle" font-size="12" fill="{PALETTE["train"]}" '
        f'font-weight="600" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">forward pass</text>'
    )
    # Backward arrow below the network.
    canvas.add(
        f'<line x1="{W - pad_r + 6:.1f}" y1="{H - bot + 6:.1f}" '
        f'x2="{pad_l - 6:.1f}" y2="{H - bot + 6:.1f}" '
        f'stroke="{PALETTE["test"]}" stroke-width="2.4" '
        f'marker-end="url(#bpbwd)"/>'
    )
    canvas.add(
        f'<text x="{(pad_l + W - pad_r)/2:.1f}" y="{H - bot + 18:.1f}" '
        f'text-anchor="middle" font-size="12" fill="{PALETTE["test"]}" '
        f'font-weight="600" font-family="DejaVu Sans, ui-sans-serif, sans-serif">'
        f'backward gradient ∂L/∂w</text>'
    )
    # Nodes.
    for li, col in enumerate(node_pos):
        for x, y in col:
            canvas.add(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" '
                f'fill="#fff" stroke="{PALETTE["axis"]}" stroke-width="1.4"/>'
            )
        canvas.add(
            f'<text x="{col[0][0]:.1f}" y="{H - bot - 4:.1f}" '
            f'text-anchor="middle" font-size="11" fill="{PALETTE["muted"]}" '
            f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">{layer_labels[li]}</text>'
        )
    # Loss node at far right.
    out_x = layer_x[-1] + 50
    out_y = (top + (H - bot)) / 2
    canvas.add(
        f'<rect x="{out_x - 22:.1f}" y="{out_y - 16:.1f}" '
        f'width="44" height="32" rx="6" fill="#fff3e0" '
        f'stroke="{PALETTE["variance"]}" stroke-width="1.6"/>'
        f'<text x="{out_x:.1f}" y="{out_y + 5:.1f}" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="{PALETTE["variance"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">L</text>'
    )
    for x, y in node_pos[-1]:
        canvas.add(
            f'<line x1="{x:.1f}" y1="{y:.1f}" '
            f'x2="{out_x - 22:.1f}" y2="{out_y:.1f}" '
            f'stroke="{PALETTE["axis"]}" stroke-width="1"/>'
        )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "layers": layer_sizes},
        title="Back-propagation",
    )


# ---------------------------------------------------------------------------
# 10. Activation functions — sigmoid, tanh, relu
# ---------------------------------------------------------------------------

def activation_functions(*, seed: int = 0) -> GenResult:
    W, H = 480.0, 280.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=-4, x_hi=4, y_lo=-1.4, y_hi=1.4,
        x_label="z", y_label="σ(z)",
        title="Activation functions",
    )
    ax.draw_frame(x_ticks=4, y_ticks=4)
    xs = [(-4 + 8 * i / 80) for i in range(81)]
    sig = [(x, 2 / (1 + math.exp(-x)) - 1) for x in xs]   # rescaled to [-1,1]
    tanh = [(x, math.tanh(x)) for x in xs]
    relu = [(x, max(0.0, x) / 4 * 1.4 - 0.7) for x in xs]
    # Above three are scaled to fit the [-1.4, 1.4] frame.
    ax.polyline(sig, color=PALETTE["train"], width=2.2,
                label="sigmoid", label_pos=(2.0, 0.4))
    ax.polyline(tanh, color=PALETTE["test"], width=2.2,
                label="tanh", label_pos=(2.0, 0.95))
    ax.polyline(relu, color=PALETTE["bias"], width=2.2,
                label="ReLU", label_pos=(2.0, -0.2))
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Activation functions",
    )


# ---------------------------------------------------------------------------
# 11. Weight matrix in a neural network — grid of weights with W·x labels
# ---------------------------------------------------------------------------

def weight_matrix(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 540.0, 320.0
    canvas = SVGCanvas(W, H)
    canvas.add(
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">Weight matrix W: '
        f'connections from one layer to the next</text>'
    )
    # Left layer (input nodes) and right layer (output nodes).
    n_in, n_out = 4, 3
    pad_l = 60.0
    pad_r = 200.0
    top, bot = 50.0, 30.0
    plot_top = top
    plot_bot = H - bot
    in_x = pad_l
    out_x = W - pad_r
    in_y = [plot_top + (plot_bot - plot_top) * (i + 1) / (n_in + 1)
            for i in range(n_in)]
    out_y = [plot_top + (plot_bot - plot_top) * (i + 1) / (n_out + 1)
             for i in range(n_out)]
    # Edges with weight values.
    weights = []
    for j, oy in enumerate(out_y):
        row = []
        for i, iy in enumerate(in_y):
            w = round(rng.uniform(-1.2, 1.2), 2)
            row.append(w)
            color = PALETTE["train"] if w >= 0 else PALETTE["test"]
            stroke_w = 0.5 + min(2.0, abs(w) * 1.6)
            canvas.add(
                f'<line x1="{in_x:.1f}" y1="{iy:.1f}" '
                f'x2="{out_x:.1f}" y2="{oy:.1f}" '
                f'stroke="{color}" stroke-width="{stroke_w:.2f}" '
                f'opacity="0.85"/>'
            )
        weights.append(row)
    # Nodes.
    for x, ys, label in [(in_x, in_y, "x"), (out_x, out_y, "z")]:
        for k, y in enumerate(ys):
            canvas.add(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" '
                f'fill="#fff" stroke="{PALETTE["axis"]}" stroke-width="1.4"/>'
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{PALETTE["ink"]}" '
                f'font-family="ui-monospace,monospace">{label}{k+1}</text>'
            )
    # Layer captions.
    canvas.add(
        f'<text x="{in_x:.1f}" y="{plot_top - 12:.1f}" '
        f'text-anchor="middle" font-size="11" fill="{PALETTE["muted"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">input layer (n={n_in})</text>'
    )
    canvas.add(
        f'<text x="{out_x:.1f}" y="{plot_top - 12:.1f}" '
        f'text-anchor="middle" font-size="11" fill="{PALETTE["muted"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">output (n={n_out})</text>'
    )
    # Weight matrix display on the right.
    mat_x = out_x + 50.0
    mat_y = plot_top + 8.0
    cell_w, cell_h = 38.0, 24.0
    bracket_x = mat_x - 6
    bracket_h = cell_h * n_out + 8
    canvas.add(
        f'<path d="M {bracket_x:.1f},{mat_y - 4:.1f} '
        f'L {bracket_x - 8:.1f},{mat_y - 4:.1f} '
        f'L {bracket_x - 8:.1f},{mat_y + bracket_h:.1f} '
        f'L {bracket_x:.1f},{mat_y + bracket_h:.1f}" '
        f'fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.4"/>'
    )
    rb = mat_x + cell_w * n_in + 4
    canvas.add(
        f'<path d="M {rb:.1f},{mat_y - 4:.1f} '
        f'L {rb + 8:.1f},{mat_y - 4:.1f} '
        f'L {rb + 8:.1f},{mat_y + bracket_h:.1f} '
        f'L {rb:.1f},{mat_y + bracket_h:.1f}" '
        f'fill="none" stroke="{PALETTE["axis"]}" stroke-width="1.4"/>'
    )
    for j in range(n_out):
        for i in range(n_in):
            cx = mat_x + cell_w * i + cell_w / 2
            cy = mat_y + cell_h * j + cell_h / 2 + 4
            v = weights[j][i]
            color = PALETTE["train"] if v >= 0 else PALETTE["test"]
            canvas.add(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{color}" font-weight="600" '
                f'font-family="ui-monospace,monospace">{v:+.2f}</text>'
            )
    canvas.add(
        f'<text x="{mat_x + cell_w * n_in / 2:.1f}" '
        f'y="{mat_y + bracket_h + 16:.1f}" text-anchor="middle" '
        f'font-size="12" fill="{PALETTE["ink"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">'
        f'W ({n_out} × {n_in})</text>'
    )
    canvas.add(
        f'<text x="{mat_x + cell_w * n_in / 2:.1f}" '
        f'y="{mat_y + bracket_h + 30:.1f}" text-anchor="middle" '
        f'font-size="11" fill="{PALETTE["muted"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">z = W x + b</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed, "shape": (n_out, n_in)},
        title="Weight matrix",
    )


# ---------------------------------------------------------------------------
# 12. Loss curve over parameters — R(θ) bowl with a marked minimum
# ---------------------------------------------------------------------------

def loss_function(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 480.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=-3.0, x_hi=3.0, y_lo=0.0, y_hi=4.5,
        x_label="parameter θ",
        y_label="loss R(θ)",
        title="Loss function R(θ)",
    )
    ax.draw_frame(x_ticks=4, y_ticks=4)
    xs = [-3.0 + 6.0 * i / 80 for i in range(81)]
    pts = [(x, 0.5 * x * x + 0.4) for x in xs]
    ax.polyline(pts, color=PALETTE["test"], width=2.4)
    # Show a couple of "bad" parameter points and one minimum.
    ax.scatter([(-2.4, 0.5 * 2.4 ** 2 + 0.4),
                (1.8, 0.5 * 1.8 ** 2 + 0.4)],
               color=PALETTE["muted"], r=3.5)
    ax.scatter([(0.0, 0.4)], color=PALETTE["bias"], r=4.5)
    canvas.add(
        f'<text x="{ax.to_px(0):.1f}" y="{ax.to_py(0.4) - 10:.1f}" '
        f'text-anchor="middle" font-size="11" '
        f'fill="{PALETTE["bias"]}" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">θ* (minimum)</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Loss function R(θ)",
    )


# ---------------------------------------------------------------------------
# 8. Decision boundary — 2-class scatter + classifier boundary
# ---------------------------------------------------------------------------

def decision_boundary(*, seed: int = 0) -> GenResult:
    rng = random.Random(seed)
    W, H = 360.0, 300.0
    canvas = SVGCanvas(W, H)
    ax = Axes(
        canvas, x_lo=-3.5, x_hi=3.5, y_lo=-3.0, y_hi=3.0,
        x_label="x₁", y_label="x₂",
        title="Decision boundary",
    )
    ax.draw_frame(x_ticks=5, y_ticks=5)
    # Two Gaussian blobs.
    for cx, cy, color in [(-1.2, -0.6, PALETTE["train"]),
                           (1.2, 0.6, PALETTE["test"])]:
        for _ in range(28):
            x = cx + rng.gauss(0, 0.8)
            y = cy + rng.gauss(0, 0.8)
            canvas.add(
                f'<circle cx="{ax.to_px(x):.2f}" cy="{ax.to_py(y):.2f}" '
                f'r="3" fill="{color}" stroke="#fff" stroke-width="0.5"/>'
            )
    # Linear boundary x₁ + x₂ = 0 → x₂ = -x₁.
    pts = [(-3.5, 3.5), (3.5, -3.5)]
    ax.polyline(pts, color=PALETTE["axis"], width=2.0)
    canvas.add(
        f'<text x="{ax.to_px(-2.7):.1f}" y="{ax.to_py(2.5):.1f}" '
        f'fill="{PALETTE["axis"]}" font-size="11" '
        f'font-family="DejaVu Sans, ui-sans-serif, sans-serif">decision boundary</text>'
    )
    return GenResult(
        svg_body=canvas.render_body(), width=W, height=H,
        params={"seed": seed},
        title="Decision boundary",
    )
