"""
Loss function visuals (2-panel figure):

1) Comparison loss:
   - Randomly generate pairwise predictions and ground-truth labels.
   - Show them side-by-side (pred vs true) and highlight mistakes.

2) Ordering loss:
   - Show true ordering (left) vs predicted ordering (right).
   - Draw lines connecting each item from its true position to predicted position.
   - Ordering loss corresponds to sum_i |pred_pos(i) - true_pos(i)|.
"""



from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
TOP_MARGIN = 0.7   # smaller = more space for titles (0.85–0.9 is good)


def save_loss_demo_figure(
    out_path: str = "outputs/losses/loss_functions_demo.png",
    n_pairs: int = 18,
    n_items: int = 10,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # -------------------------
    # (A) Comparison loss demo
    # -------------------------
    # y_true, y_pred in {0,1}; 1 means "A more concise than B"
    y_true = rng.integers(0, 2, size=n_pairs)
    y_pred = y_true.copy()

    # Inject some mistakes (~25%)
    n_flip = max(1, int(0.25 * n_pairs))
    flip_idx = rng.choice(n_pairs, size=n_flip, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    mistakes = (y_pred != y_true)
    acc = 1.0 - mistakes.mean()
    L_comp = 1.0 - acc

    # Create readable labels like "Pair 1", etc.
    pair_labels = [f"P{i+1}" for i in range(n_pairs)]

    # -------------------------
    # (B) Ordering loss demo
    # -------------------------
    # Items labeled 1..n_items
    items = np.arange(1, n_items + 1)

    # True ordering: random permutation
    true_order = rng.permutation(items)

    # Pred ordering: start from true and apply a few swaps to create errors
    pred_order = true_order.copy()
    n_swaps = max(1, n_items // 4)
    for _ in range(n_swaps):
        a, b = rng.choice(n_items, size=2, replace=False)
        pred_order[a], pred_order[b] = pred_order[b], pred_order[a]

    # Compute positions (rank indices) to show L_ord
    true_pos = {item: i for i, item in enumerate(true_order)}
    pred_pos = {item: i for i, item in enumerate(pred_order)}
    abs_disp = np.array([abs(pred_pos[item] - true_pos[item]) for item in items])
    L_ord = int(abs_disp.sum())

    # -------------------------
    # Plotting
    # -------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(12, 6))

    # ===== Left panel: comparison loss =====
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, n_pairs - 0.5)
    ax1.axis("off")

    # Column headers
    ax1.text(0.5, n_pairs - 0.1, "Pred", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.text(1.5, n_pairs - 0.1, "True", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.text(-0.35, n_pairs - 0.1, "Pair", ha="left", va="bottom", fontsize=11, fontweight="bold")

    # Draw rows as little rectangles with text
    # Highlight mistakes by coloring the Pred cell
    for i in range(n_pairs):
        y = n_pairs - 1 - i  # top to bottom

        # Pair label
        ax1.text(-0.35, y, pair_labels[i], ha="left", va="center", fontsize=10)

        # Pred cell
        pred_face = "#ffcccc" if mistakes[i] else "#ccffcc"  # red-ish for mistake, green-ish for correct
        ax1.add_patch(plt.Rectangle((0.0, y - 0.4), 1.0, 0.8, facecolor=pred_face, edgecolor="black", linewidth=0.8))
        ax1.text(0.5, y, str(int(y_pred[i])), ha="center", va="center", fontsize=11, fontweight="bold")

        # True cell (neutral)
        ax1.add_patch(plt.Rectangle((1.0, y - 0.4), 1.0, 0.8, facecolor="#f2f2f2", edgecolor="black", linewidth=0.8))
        ax1.text(1.5, y, str(int(y_true[i])), ha="center", va="center", fontsize=11, fontweight="bold")

    ax1.text(
        0.5,
        -0.9,
        f"Mistakes: {int(mistakes.sum())}/{n_pairs}  |  Accuracy: {acc:.3f}",
        ha="center",
        va="top",
        fontsize=10,
    )

    # ===== Right panel: ordering loss =====
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f"Ordering Loss: {L_ord}")
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.5, n_items - 0.5)
    ax2.axis("off")

    # positions vertically: top = rank 1
    y_positions = np.arange(n_items)[::-1]

    # Draw left (true) and right (pred) columns with item labels
    x_true, x_pred = 0.0, 1.0

    ax2.text(x_true, n_items - 0.1, "True order", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.text(x_pred, n_items - 0.1, "Pred order", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Map item -> y coordinate in each column
    true_y = {item: y_positions[i] for i, item in enumerate(true_order)}
    pred_y = {item: y_positions[i] for i, item in enumerate(pred_order)}

    # draw nodes (as small boxes) and connecting lines
    for i, item in enumerate(items):
        yt = true_y[item]
        yp = pred_y[item]

        # True box
        ax2.add_patch(plt.Rectangle((x_true - 0.12, yt - 0.35), 0.24, 0.7, facecolor="#f2f2f2", edgecolor="black", linewidth=0.8))
        ax2.text(x_true, yt, str(item), ha="center", va="center", fontsize=11, fontweight="bold")

        # Pred box
        ax2.add_patch(plt.Rectangle((x_pred - 0.12, yp - 0.35), 0.24, 0.7, facecolor="#f2f2f2", edgecolor="black", linewidth=0.8))
        ax2.text(x_pred, yp, str(item), ha="center", va="center", fontsize=11, fontweight="bold")

        # Line from true position to predicted position
        # (Thicker lines for larger displacement makes the idea pop)
        disp = abs(pred_pos[item] - true_pos[item])
        lw = 1.0 + 0.35 * disp
        ax2.plot([x_true + 0.12, x_pred - 0.12], [yt, yp], linewidth=lw, alpha=0.85)

    # Also print the two order arrays below for clarity
    true_str = "True: " + "  ".join(map(str, true_order.tolist()))
    pred_str = "Pred: " + "  ".join(map(str, pred_order.tolist()))
    ax2.text(0.5, -0.9, true_str, ha="center", va="top", fontsize=10)
    ax2.text(0.5, -1.35, pred_str, ha="center", va="top", fontsize=10)

    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    return out_path


if __name__ == "__main__":
    path = save_loss_demo_figure(
        out_path="outputs/losses/loss_functions_demo.png",
        n_pairs=18,
        n_items=10,
        seed=0,
    )
    print("Saved:", os.path.abspath(path))
