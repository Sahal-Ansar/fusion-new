"""
Flat-vector illustrations in the spirit of Google Alegria (geometric, flat
colors, abstract figures, limited palette, no outlines).

Outputs (in ./assets/):
    illust_hero_car.png       - title-slide hero illustration (autonomous car w/ sensors)
    illust_problem_ghosts.png - the "moving car ghosting" metaphor
    illust_pipeline_icons.png - 4 simple stage icons (camera / events / flow / lidar)
    illust_thanks.png         - closing illustration
    icon_imu.png / icon_dl.png / icon_ours.png - related-work column icons

All vector-drawn with matplotlib patches; no external SVGs needed.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Rectangle, Polygon,
                                Wedge, Ellipse, FancyArrowPatch, Arc)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Alegria-leaning palette (flat, slightly muted)
A_TEAL    = "#3DB3A6"
A_CORAL   = "#F26D5B"
A_MUSTARD = "#F2C14E"
A_NAVY    = "#1C3D5A"
A_BLUSH   = "#F4B8B0"
A_SKY     = "#A0D8EF"
A_CREAM   = "#F8F1E5"
A_OLIVE   = "#7B9E54"
A_GREY    = "#6B6B6B"
A_DARK    = "#2C2C2C"


def _new_canvas(w=10, h=5.6, bg="white"):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax


def _save(fig, name, dpi=200):
    out = ASSETS / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out.name}")


# =========================================================================
# A flat car silhouette (used in several illustrations)
# =========================================================================
def _draw_car(ax, cx, cy, scale=1.0, body=A_CORAL, window=A_NAVY,
              wheel=A_DARK, alpha=1.0, sensors=False):
    s = scale
    # body (rounded rectangle)
    body_w, body_h = 3.0 * s, 0.95 * s
    ax.add_patch(FancyBboxPatch(
        (cx - body_w / 2, cy - body_h / 2), body_w, body_h,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        facecolor=body, edgecolor="none", alpha=alpha, zorder=2))
    # cabin (trapezoid via Polygon)
    cabin = np.array([
        [cx - 0.95 * s, cy + body_h / 2 - 0.02],
        [cx - 0.55 * s, cy + body_h / 2 + 0.55 * s],
        [cx + 0.65 * s, cy + body_h / 2 + 0.55 * s],
        [cx + 1.05 * s, cy + body_h / 2 - 0.02],
    ])
    ax.add_patch(Polygon(cabin, facecolor=body, edgecolor="none", alpha=alpha, zorder=2))
    # window
    win = np.array([
        [cx - 0.78 * s, cy + body_h / 2 + 0.05],
        [cx - 0.45 * s, cy + body_h / 2 + 0.42 * s],
        [cx + 0.55 * s, cy + body_h / 2 + 0.42 * s],
        [cx + 0.88 * s, cy + body_h / 2 + 0.05],
    ])
    ax.add_patch(Polygon(win, facecolor=window, edgecolor="none", alpha=alpha, zorder=3))
    # wheels
    wh_r = 0.32 * s
    for wx in (cx - 0.95 * s, cx + 0.95 * s):
        ax.add_patch(Circle((wx, cy - body_h / 2), wh_r, facecolor=wheel,
                            edgecolor="none", alpha=alpha, zorder=4))
        ax.add_patch(Circle((wx, cy - body_h / 2), wh_r * 0.45, facecolor="white",
                            edgecolor="none", alpha=alpha, zorder=5))
    # optional roof-mounted sensor stack
    if sensors:
        sx = cx + 0.05 * s
        sy = cy + body_h / 2 + 0.55 * s
        # cylindrical LiDAR
        ax.add_patch(Rectangle((sx - 0.18 * s, sy), 0.36 * s, 0.22 * s,
                               facecolor=A_NAVY, edgecolor="none", zorder=6))
        ax.add_patch(Ellipse((sx, sy + 0.22 * s), 0.36 * s, 0.10 * s,
                             facecolor=A_TEAL, edgecolor="none", zorder=7))
        ax.add_patch(Ellipse((sx, sy), 0.36 * s, 0.10 * s,
                             facecolor=A_NAVY, edgecolor="none", zorder=7))
        # spinning arrow
        ax.add_patch(Arc((sx, sy + 0.32 * s), 0.55 * s, 0.18 * s, angle=0,
                         theta1=20, theta2=320, color=A_DARK, lw=1.5))


# =========================================================================
# 1. Hero illustration: car with sensors emitting beams
# =========================================================================
def make_hero_car():
    fig, ax = _new_canvas(11, 5.4, bg="white")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-2.4, 3.2)

    # ground line
    ax.add_patch(Rectangle((-5.5, -2.4), 11, 0.5, facecolor=A_CREAM,
                           edgecolor="none", zorder=0))
    # decorative dots in background
    rng = np.random.default_rng(7)
    for _ in range(28):
        x = rng.uniform(-5.4, 5.4)
        y = rng.uniform(0.2, 3.0)
        r = rng.uniform(0.04, 0.12)
        ax.add_patch(Circle((x, y), r, facecolor=A_BLUSH, alpha=0.45, edgecolor="none", zorder=0))

    _draw_car(ax, 0, -0.7, scale=1.25, body=A_CORAL, window=A_NAVY, sensors=True)

    # LiDAR beams
    beam_origin = (0.06, 0.85)
    for ang_deg, length, color in [
        (160, 4.6, A_TEAL), (140, 4.2, A_TEAL), (120, 3.8, A_MUSTARD),
        (60, 3.8, A_MUSTARD), (40, 4.2, A_TEAL), (20, 4.6, A_TEAL),
    ]:
        ang = np.deg2rad(ang_deg)
        ex = beam_origin[0] + length * np.cos(ang)
        ey = beam_origin[1] + length * np.sin(ang)
        ax.plot([beam_origin[0], ex], [beam_origin[1], ey],
                color=color, lw=2.2, alpha=0.75, zorder=1, solid_capstyle="round")
        ax.add_patch(Circle((ex, ey), 0.10, facecolor=color, edgecolor="none", zorder=2))

    # camera FOV cone
    cam_origin = (0.06, 0.95)
    for ang in (75, 105):
        ang_r = np.deg2rad(ang)
        ex = cam_origin[0] + 5.0 * np.cos(ang_r)
        ey = cam_origin[1] + 5.0 * np.sin(ang_r)
        ax.plot([cam_origin[0], ex], [cam_origin[1], ey], color=A_NAVY,
                lw=1.0, alpha=0.35, ls="--", zorder=1)

    _save(fig, "illust_hero_car.png")


# =========================================================================
# 2. The "ghosting" metaphor: same car, three offset versions
# =========================================================================
def make_problem_ghosts():
    fig, ax = _new_canvas(11, 4.4, bg="white")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 2)

    # ground
    ax.add_patch(Rectangle((-6, -2), 12, 0.45, facecolor=A_CREAM,
                           edgecolor="none", zorder=0))

    for cx, alpha, body in [(-2.6, 0.35, A_BLUSH),
                            (-0.4, 0.65, A_BLUSH),
                            (1.8, 1.00, A_CORAL)]:
        _draw_car(ax, cx, -0.55, scale=0.95, body=body, window=A_NAVY, alpha=alpha)

    # Motion arrow
    ax.add_patch(FancyArrowPatch(
        (-3.2, 1.05), (3.0, 1.05),
        arrowstyle="-|>,head_length=10,head_width=8",
        color=A_NAVY, lw=2.2))
    ax.text(-0.1, 1.45, "what the LiDAR \"sees\" depends on when each beam fires",
            ha="center", fontsize=11, color=A_DARK)

    # Labels under cars
    for cx, label in [(-2.6, "t = 0"), (-0.4, "t = 50 ms"), (1.8, "t = 100 ms")]:
        ax.text(cx, -1.55, label, ha="center", fontsize=9.5, color=A_GREY)

    _save(fig, "illust_problem_ghosts.png")


# =========================================================================
# 3. Pipeline icons (4 small flat icons in a row)
# =========================================================================
def make_pipeline_icons():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    fig.patch.set_facecolor("white")

    titles = ["LiDAR project", "Simulate events", "Optical flow", "Event-gated fix"]

    # 1. LiDAR icon: cylinder with beams
    ax = axes[0]
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.add_patch(FancyBboxPatch((-0.55, -0.4), 1.1, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.20",
                                facecolor=A_NAVY, edgecolor="none"))
    ax.add_patch(Ellipse((0, 0.4), 1.1, 0.32, facecolor=A_TEAL, edgecolor="none"))
    for ang in range(20, 160, 25):
        a = np.deg2rad(ang)
        ax.plot([0, 1.7 * np.cos(a)], [0.4, 0.4 + 1.55 * np.sin(a)],
                color=A_TEAL, lw=2, alpha=0.7, solid_capstyle="round")

    # 2. Events icon: dotted speckle on a frame
    ax = axes[1]
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.add_patch(FancyBboxPatch((-1.4, -1.0), 2.8, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_DARK, edgecolor="none"))
    rng = np.random.default_rng(2)
    for _ in range(70):
        x = rng.uniform(-1.3, 1.3); y = rng.uniform(-0.9, 0.9)
        c = "white" if rng.random() > 0.4 else A_SKY
        ax.add_patch(Circle((x, y), rng.uniform(0.04, 0.09),
                            facecolor=c, edgecolor="none"))

    # 3. Flow icon: arrows
    ax = axes[2]
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.add_patch(FancyBboxPatch((-1.4, -1.0), 2.8, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_CREAM, edgecolor="none"))
    grid = [(-0.95, 0.55), (-0.05, 0.55), (0.85, 0.55),
            (-0.95, -0.05), (-0.05, -0.05), (0.85, -0.05),
            (-0.95, -0.65), (-0.05, -0.65), (0.85, -0.65)]
    for x, y in grid:
        ax.add_patch(FancyArrowPatch((x, y), (x + 0.6, y + 0.18),
                                     arrowstyle="-|>,head_length=5,head_width=4",
                                     color=A_CORAL, lw=1.6))

    # 4. Event-gated fix icon: gear + check
    ax = axes[3]
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    # mask
    ax.add_patch(FancyBboxPatch((-1.4, -1.0), 2.8, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_CREAM, edgecolor="none"))
    # red point -> green point
    ax.add_patch(Circle((-0.7, 0.0), 0.18, facecolor=A_CORAL, edgecolor="none"))
    ax.add_patch(FancyArrowPatch((-0.5, 0.0), (0.5, 0.0),
                                 arrowstyle="-|>,head_length=8,head_width=6",
                                 color=A_NAVY, lw=2))
    ax.add_patch(Circle((0.75, 0.0), 0.18, facecolor=A_OLIVE, edgecolor="none"))
    # tiny mask box behind the arrow
    ax.add_patch(FancyBboxPatch((-0.55, -0.45), 1.35, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_MUSTARD, edgecolor="none", alpha=0.25))

    for ax, title in zip(axes, titles):
        ax.text(0, -1.55, title, ha="center", fontsize=11.5, color=A_DARK,
                fontweight="bold")

    fig.tight_layout()
    _save(fig, "illust_pipeline_icons.png")


# =========================================================================
# 4. Related-work column icons (3 separate small files)
# =========================================================================
def _icon_canvas():
    fig, ax = _new_canvas(3.2, 3.2, bg="white")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    return fig, ax


def make_icon_imu():
    fig, ax = _icon_canvas()
    # gyro / chip
    ax.add_patch(FancyBboxPatch((-1.1, -1.1), 2.2, 2.2,
                                boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=A_NAVY, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((-0.6, -0.6), 1.2, 1.2,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_TEAL, edgecolor="none"))
    ax.text(0, 0.0, "IMU", ha="center", va="center", fontsize=18,
            fontweight="bold", color="white")
    # corner pins
    for x in (-1.1, 1.1):
        for y in (-0.6, 0.0, 0.6):
            ax.add_patch(Rectangle((x - 0.18, y - 0.07), 0.18, 0.14,
                                   facecolor=A_MUSTARD, edgecolor="none"))
    _save(fig, "icon_imu.png")


def make_icon_dl():
    fig, ax = _icon_canvas()
    # neural net nodes
    layers = [(-1.2, [-0.9, -0.3, 0.3, 0.9]),
              (0.0,  [-0.7, 0.0, 0.7]),
              (1.2,  [-0.5, 0.5])]
    coords = []
    for x, ys in layers:
        coords.append([(x, y) for y in ys])
    # edges
    for li in range(len(coords) - 1):
        for a in coords[li]:
            for b in coords[li + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=A_GREY, lw=0.7, alpha=0.5)
    # nodes
    for li, layer in enumerate(coords):
        for (x, y) in layer:
            c = [A_CORAL, A_MUSTARD, A_TEAL][li]
            ax.add_patch(Circle((x, y), 0.16, facecolor=c, edgecolor="none"))
    _save(fig, "icon_dl.png")


def make_icon_ours():
    fig, ax = _icon_canvas()
    # camera body
    ax.add_patch(FancyBboxPatch((-1.2, -0.7), 2.4, 1.4,
                                boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=A_CORAL, edgecolor="none"))
    # viewfinder bump
    ax.add_patch(FancyBboxPatch((-0.4, 0.65), 0.8, 0.35,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=A_CORAL, edgecolor="none"))
    # lens
    ax.add_patch(Circle((0, 0), 0.55, facecolor=A_NAVY, edgecolor="none"))
    ax.add_patch(Circle((0, 0), 0.40, facecolor=A_TEAL, edgecolor="none"))
    ax.add_patch(Circle((-0.12, 0.12), 0.13, facecolor="white", edgecolor="none", alpha=0.7))
    # event spark
    ax.add_patch(Circle((0.95, 0.85), 0.13, facecolor=A_MUSTARD, edgecolor="none"))
    ax.add_patch(Circle((-0.95, 0.85), 0.10, facecolor=A_MUSTARD, edgecolor="none", alpha=0.7))
    _save(fig, "icon_ours.png")


# =========================================================================
# 5. Closing illustration: car + check mark
# =========================================================================
def make_thanks():
    fig, ax = _new_canvas(10, 5.0, bg="white")
    ax.set_xlim(-5, 5); ax.set_ylim(-2.5, 2.7)

    # ground
    ax.add_patch(Rectangle((-5, -2.5), 10, 0.45, facecolor=A_CREAM,
                           edgecolor="none", zorder=0))

    _draw_car(ax, -1.5, -0.7, scale=1.1, body=A_TEAL, window=A_NAVY, sensors=True)

    # Big check mark in a circle
    ax.add_patch(Circle((2.6, 0.4), 1.05, facecolor=A_OLIVE, edgecolor="none"))
    ax.plot([2.05, 2.5, 3.2], [0.45, -0.05, 0.85], color="white", lw=5,
            solid_capstyle="round", solid_joinstyle="round")

    ax.text(0, 2.15, "Thank you", ha="center", fontsize=22,
            fontweight="bold", color=A_NAVY)
    _save(fig, "illust_thanks.png")


def main():
    print("[gen_vector_illustrations] generating Alegria-style vectors...")
    make_hero_car()
    make_problem_ghosts()
    make_pipeline_icons()
    make_icon_imu()
    make_icon_dl()
    make_icon_ours()
    make_thanks()
    print("[gen_vector_illustrations] done.")


if __name__ == "__main__":
    main()
