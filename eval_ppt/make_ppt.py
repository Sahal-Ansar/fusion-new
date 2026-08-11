"""
Build the eval-2 presentation slide deck (./eval_presentation.pptx).

Design system
-------------
* 16:9, 13.333 x 7.5 in
* Single key idea per slide; lots of whitespace
* Palette: Google Blue / Red / Yellow / Green + neutral greys
* Title family: Google Sans (fallback Calibri Light)
* Body family: Roboto (fallback Calibri)
* All images sourced from ./assets/ + ../research_paper_v3/*.png

15 slides, ~30 s each => ~7.5 minutes of voice-over headroom.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PAPER_FIGS = HERE.parent / "research_paper_v3"

OUT = HERE / "eval_presentation.pptx"

# ------------- palette -------------
BLUE   = RGBColor(0x1A, 0x73, 0xE8)
RED    = RGBColor(0xEA, 0x43, 0x35)
YELLOW = RGBColor(0xFB, 0xBC, 0x04)
GREEN  = RGBColor(0x34, 0xA8, 0x53)
GREY   = RGBColor(0x5F, 0x63, 0x68)
DARK   = RGBColor(0x20, 0x21, 0x24)
LIGHT  = RGBColor(0xF1, 0xF3, 0xF4)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

TITLE_FONT = "Google Sans"
BODY_FONT  = "Roboto"
TITLE_FALLBACK = "Calibri Light"
BODY_FALLBACK  = "Calibri"


# =========================================================================
# Helpers
# =========================================================================
def add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=DARK,
             font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.15):
    """Add a textbox; `text` may contain '\\n' for multi-line."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return tb


def add_image(slide, path, x, y, w=None, h=None):
    if w is not None and h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                        width=Inches(w), height=Inches(h))
    if w is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                        width=Inches(w))
    if h is not None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                        height=Inches(h))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y))


def add_rect(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def add_blank(prs):
    blank = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(blank)


def add_header(slide, title, kicker=None, kicker_color=BLUE):
    """Standard slide header: small colored kicker + large title."""
    if kicker:
        add_text(slide, 0.6, 0.45, 12, 0.35, kicker.upper(),
                 size=11, bold=True, color=kicker_color, font=BODY_FONT)
        add_text(slide, 0.6, 0.78, 12, 0.85, title,
                 size=30, bold=True, color=DARK, font=TITLE_FONT)
    else:
        add_text(slide, 0.6, 0.55, 12, 0.85, title,
                 size=30, bold=True, color=DARK, font=TITLE_FONT)


def add_footer(slide, page, total=15, section=None):
    """Footer with section label and slide number."""
    if section:
        add_text(slide, 0.6, 7.05, 8, 0.3, section.upper(),
                 size=9, bold=True, color=GREY, font=BODY_FONT)
    add_text(slide, 12.0, 7.05, 1.2, 0.3, f"{page} / {total}",
             size=9, color=GREY, font=BODY_FONT, align=PP_ALIGN.RIGHT)


def add_accent_bar(slide, x, y, w=0.4, h=0.06, color=BLUE):
    add_rect(slide, x, y, w, h, fill=color)


# =========================================================================
# Deck
# =========================================================================
def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    TOTAL = 15
    pg = 0

    # ====================================================================
    # SLIDE 1 - TITLE
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    # left text column
    add_accent_bar(s, 0.6, 1.55, w=0.5, h=0.08, color=BLUE)
    add_text(s, 0.6, 1.75, 6.8, 0.4, "EVALUATION 2  -  HONORS PROJECT",
             size=12, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 0.6, 2.20, 7.0, 1.8,
             "Event-Guided Motion\nCompensation for LiDAR\nTemporal Distortion",
             size=36, bold=True, color=DARK, font=TITLE_FONT, line_spacing=1.05)
    add_text(s, 0.6, 4.65, 7.0, 0.4,
             "A lightweight, image-space pipeline for autonomous driving",
             size=15, color=GREY, font=BODY_FONT)
    add_text(s, 0.6, 5.4, 7.0, 0.4, "Sahal Ansar Theparambil",
             size=14, bold=True, color=DARK, font=BODY_FONT)
    add_text(s, 0.6, 5.78, 7.0, 0.3,
             "Department of Engineering  -  IIIT Sri City",
             size=11, color=GREY, font=BODY_FONT)
    # right hero
    add_image(s, ASSETS / "illust_hero_car.png", 7.6, 1.9, w=5.5)
    add_footer(s, pg, TOTAL, "Title")

    # ====================================================================
    # SLIDE 2 - INTRODUCTION / PROBLEM
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "A spinning LiDAR doesn't take a snapshot.", kicker="The problem")
    add_image(s, ASSETS / "sweep_diagram.png", 0.6, 2.05, w=8.2)
    # right callout
    add_rect(s, 9.1, 2.05, 3.7, 4.6, fill=LIGHT)
    add_text(s, 9.35, 2.25, 3.3, 0.4, "ONE SCAN, 100 ms",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 9.35, 2.65, 3.3, 0.5, "64 beams.\nOne rotation.",
             size=18, bold=True, color=DARK, font=TITLE_FONT, line_spacing=1.1)
    add_text(s, 9.35, 3.85, 3.3, 2.5,
             "Each beam fires at a\ndifferent moment.\n\n"
             "At 60 km/h the ego car\ntravels 1.67 m between\nthe first and last beam.\n\n"
             "Result: projected points\nno longer line up with\nthe image.",
             size=11, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Introduction")

    # ====================================================================
    # SLIDE 3 - WHY IT MATTERS
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Misalignment compounds at every fusion step.",
               kicker="Why this matters")
    add_image(s, ASSETS / "illust_problem_ghosts.png", 0.6, 1.9, w=8.0)
    # right column - three small stats
    stat_x = 9.0
    stat_y_start = 2.1
    add_text(s, stat_x, stat_y_start, 4.0, 0.4, "100 ms",
             size=44, bold=True, color=RED, font=TITLE_FONT)
    add_text(s, stat_x, stat_y_start + 0.95, 4.0, 0.3,
             "scan duration on a 10 Hz Velodyne",
             size=11, color=GREY, font=BODY_FONT)

    add_text(s, stat_x, stat_y_start + 1.65, 4.0, 0.4, "1.67 m",
             size=44, bold=True, color=YELLOW, font=TITLE_FONT)
    add_text(s, stat_x, stat_y_start + 2.6, 4.0, 0.3,
             "ego displacement at highway speed",
             size=11, color=GREY, font=BODY_FONT)

    add_text(s, stat_x, stat_y_start + 3.2, 4.0, 0.4, "+",
             size=44, bold=True, color=DARK, font=TITLE_FONT)
    add_text(s, stat_x, stat_y_start + 4.0, 4.0, 0.3,
             "moving cars and people inside the scene",
             size=11, color=GREY, font=BODY_FONT)
    add_footer(s, pg, TOTAL, "Introduction")

    # ====================================================================
    # SLIDE 4 - VISUAL PROOF
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "You can see the mismatch in a single KITTI frame.",
               kicker="Visual proof")
    add_image(s, ASSETS / "misalignment_arrow.png", 0.85, 1.85, w=8.4)
    add_text(s, 9.6, 2.1, 3.3, 0.4, "WHAT YOU'RE LOOKING AT",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 9.6, 2.55, 3.3, 4.0,
             "Red dots: LiDAR points\nprojected with static\ncalibration.\n\n"
             "Yellow arrows: where\ndense optical flow says\nthose points should sit\n"
             "in the next frame.\n\n"
             "Static calibration cannot\nclose this gap on its own.",
             size=12, color=DARK, font=BODY_FONT, line_spacing=1.3)
    add_footer(s, pg, TOTAL, "Introduction")

    # ====================================================================
    # SLIDE 5 - RELATED WORK
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Existing fixes need hardware or GPUs.",
               kicker="Related work")
    # 3 columns
    col_w = 3.9
    col_y = 2.0
    col_h = 4.6
    cols = [
        ("icon_imu.png",  "IMU de-skew",
         "Integrates ego-motion\nbetween per-point\ntimestamps.\n\n"
         "Needs raw timestamps and tight inertial coupling - unavailable in\nKITTI-sync.",
         BLUE),
        ("icon_dl.png",   "Deep fusion",
         "Transformer / CNN models for cross-modal alignment.\n\n"
         "Strong benchmarks but\nrequire GPU inference\nand training data.",
         RED),
        ("icon_ours.png", "Ours (classical)",
         "Image-space optical flow,\nevent-gated, no DL,\nno IMU, no hardware.\n\n"
         "Runs on a laptop CPU and\ncovers both ego- and\nscene-motion.",
         GREEN),
    ]
    for i, (icon, title, body, color) in enumerate(cols):
        x = 0.6 + i * (col_w + 0.3)
        add_rect(s, x, col_y, col_w, col_h, fill=LIGHT)
        add_image(s, ASSETS / icon, x + col_w / 2 - 0.7, col_y + 0.2, w=1.4)
        add_accent_bar(s, x + 0.3, col_y + 1.85, w=0.35, h=0.07, color=color)
        add_text(s, x + 0.3, col_y + 2.0, col_w - 0.6, 0.45, title,
                 size=18, bold=True, color=DARK, font=TITLE_FONT)
        add_text(s, x + 0.3, col_y + 2.55, col_w - 0.6, 2.0, body,
                 size=11, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Related work")

    # ====================================================================
    # SLIDE 6 - FIRST EVALUATION RECAP
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Where the first evaluation left off.",
               kicker="Recap")
    add_image(s, ASSETS / "first_eval_recap.png", 0.6, 1.9, w=8.6)
    add_text(s, 9.5, 2.1, 3.5, 0.4, "WHAT WAS WORKING",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 9.5, 2.55, 3.5, 4.6,
             "End-to-end classical pipeline:\n"
             "project -> simulate -> flow ->\nevent-gated fix.\n\n"
             "Interactive demo in\nmain.py with side-by-side\nbefore/after views.\n\n"
             "Validation harness reading\n4 KITTI sequences and\n"
             "comparing modes.\n\n"
             "Open questions: does it\ngeneralize, and does it\nmeasurably help?",
             size=11, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Recap")

    # ====================================================================
    # SLIDE 7 - APPROACH / PIPELINE
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Four stages, all in 2D image space.",
               kicker="Approach")
    add_image(s, PAPER_FIGS / "fig7_pipeline_diagram.png", 0.6, 1.95, w=8.4)
    add_text(s, 9.4, 2.1, 3.5, 0.4, "ONE-LINE SUMMARY",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 9.4, 2.55, 3.5, 4.6,
             "Take the dense flow\nbetween two RGB frames.\n\n"
             "Use simulated events as\na confidence gate so the\n"
             "fix only fires where\nsomething actually moved.\n\n"
             "Smooth across time to\nkeep the correction stable.\n\n"
             "Apply only to the 2D\nLiDAR projection - no 3D\npoints touched.",
             size=11, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Method")

    # ====================================================================
    # SLIDE 8 - KEY IDEA: EVENT GATING
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "The trick: trust flow only where events fire.",
               kicker="Key idea")
    add_image(s, ASSETS / "event_map_demo.png", 0.6, 1.95, w=8.6)
    add_text(s, 9.5, 2.1, 3.5, 0.4, "WHY GATING HELPS",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 9.5, 2.55, 3.5, 4.6,
             "Optical flow estimates\nsomething everywhere -\neven over a flat road.\n\n"
             "log(I_t+1) - log(I_t)\napproximates a DVS sensor\nat near-zero compute cost.\n\n"
             "Edge-proximity filtering\nlocalizes events to true\nscene boundaries.\n\n"
             "Net effect: correction\nfires on objects,\nstays quiet on flat ground.",
             size=11, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Method")

    # ====================================================================
    # SLIDE 9 - IMPLEMENTATION: FOUR STAGES
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Each stage is a few lines of NumPy / OpenCV.",
               kicker="Implementation")
    add_image(s, ASSETS / "illust_pipeline_icons.png", 0.6, 2.05, w=12.1)
    # four mini captions
    captions = [
        ("Project", "KITTI calibration chain:\nP_rect @ R_rect @ T_velo->cam",  GREEN),
        ("Simulate", "log-intensity diff,\nthreshold = 0.3,\nedge-gated by Canny",  BLUE),
        ("Flow", "Farneback dense flow,\nEMA-smoothed with\nalpha = 0.7",          YELLOW),
        ("Correct", "Shift point by 0.5 x flow\nonly inside the event mask",       RED),
    ]
    box_w = 2.9
    for i, (title, body, color) in enumerate(captions):
        x = 0.7 + i * (box_w + 0.18)
        add_accent_bar(s, x, 5.25, w=0.4, h=0.07, color=color)
        add_text(s, x, 5.4, box_w, 0.4, title,
                 size=15, bold=True, color=DARK, font=TITLE_FONT)
        add_text(s, x, 5.85, box_w, 1.2, body,
                 size=10, color=GREY, font=BODY_FONT, line_spacing=1.3)
    add_footer(s, pg, TOTAL, "Implementation")

    # ====================================================================
    # SLIDE 10 - QUALITATIVE RESULTS
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Sharper alignment on real KITTI frames.",
               kicker="Qualitative result")
    add_image(s, ASSETS / "before_after_zoom.png", 1.0, 1.95, w=11.3)
    add_text(s, 0.6, 6.5, 12, 0.4,
             "Zoom on drive_0009 frame 40. Same LiDAR scan, projected onto frame t+1, "
             "before and after the event-gated correction.",
             size=11, color=GREY, font=BODY_FONT, align=PP_ALIGN.CENTER)
    add_footer(s, pg, TOTAL, "Results")

    # ====================================================================
    # SLIDE 11 - QUANTITATIVE: SPEAS + SRC
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Statistically significant across 24 sequences.",
               kicker="Quantitative result")
    add_image(s, PAPER_FIGS / "fig_speas_src_improvement.png",
              0.6, 1.95, w=8.6)
    # right-side stat cards
    card_x = 9.5
    card_w = 3.4
    # SPEAS
    add_rect(s, card_x, 2.05, card_w, 1.5, fill=LIGHT)
    add_text(s, card_x + 0.25, 2.15, card_w - 0.5, 0.3,
             "SPEAS  -  edge alignment",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, card_x + 0.25, 2.5, card_w - 0.5, 0.7, "+0.64 %",
             size=30, bold=True, color=GREEN, font=TITLE_FONT)
    add_text(s, card_x + 0.25, 3.15, card_w - 0.5, 0.3,
             "p < 0.001  -  Cohen's d = 1.05",
             size=10, color=GREY, font=BODY_FONT)
    # SRC
    add_rect(s, card_x, 3.75, card_w, 1.5, fill=LIGHT)
    add_text(s, card_x + 0.25, 3.85, card_w - 0.5, 0.3,
             "SRC  -  stereo reprojection",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, card_x + 0.25, 4.2, card_w - 0.5, 0.7, "+0.37 %",
             size=30, bold=True, color=GREEN, font=TITLE_FONT)
    add_text(s, card_x + 0.25, 4.85, card_w - 0.5, 0.3,
             "p < 0.001  -  d = 0.90",
             size=10, color=GREY, font=BODY_FONT)
    # vs IMU
    add_rect(s, card_x, 5.45, card_w, 1.15, fill=DARK)
    add_text(s, card_x + 0.25, 5.55, card_w - 0.5, 0.3,
             "23 / 24 SEQUENCES",
             size=10, bold=True, color=YELLOW, font=BODY_FONT)
    add_text(s, card_x + 0.25, 5.85, card_w - 0.5, 0.6,
             "we beat the IMU-only\nbaseline on both metrics.",
             size=12, color=WHITE, font=BODY_FONT, line_spacing=1.2)
    add_footer(s, pg, TOTAL, "Results")

    # ====================================================================
    # SLIDE 12 - DOWNSTREAM: BDPS
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Sharper depth edges at object boundaries.",
               kicker="Downstream impact")
    add_image(s, PAPER_FIGS / "fig_bdps_improvement.png", 0.7, 1.95,
              w=8.0, h=4.5)
    # big number card
    add_rect(s, 9.2, 2.4, 3.7, 2.7, fill=GREEN)
    add_text(s, 9.4, 2.55, 3.3, 0.35, "BDPS IMPROVEMENT",
             size=11, bold=True, color=WHITE, font=BODY_FONT)
    add_text(s, 9.4, 2.95, 3.3, 1.4, "+48.5 %",
             size=58, bold=True, color=WHITE, font=TITLE_FONT)
    add_text(s, 9.4, 4.45, 3.3, 0.6,
             "mean improvement\nacross 1,691 frames",
             size=12, color=WHITE, font=BODY_FONT, line_spacing=1.2)
    add_text(s, 9.2, 5.3, 3.7, 0.4,
             "99.2 % of frames improve",
             size=13, bold=True, color=DARK, font=BODY_FONT,
             align=PP_ALIGN.CENTER)
    add_text(s, 0.7, 6.65, 8.0, 0.35,
             "BDPS = Sobel-gradient sharpness of sparse depth near image edges.",
             size=10, color=GREY, font=BODY_FONT, align=PP_ALIGN.CENTER)
    add_footer(s, pg, TOTAL, "Results")

    # ====================================================================
    # SLIDE 13 - ABLATION + RUNTIME
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "Every component carries its weight.",
               kicker="Ablation & runtime")
    # ablation chart (left) - height-capped so the caption fits below
    add_image(s, PAPER_FIGS / "fig_ablation.png", 0.5, 1.95, w=6.0, h=4.3)
    add_text(s, 0.5, 6.35, 6.0, 0.35,
             "Drop event gating -> SRC goes negative.  alpha = 0.75 -> both worsen.",
             size=10, color=GREY, font=BODY_FONT, align=PP_ALIGN.CENTER)
    # runtime (right) - height-capped to match the ablation chart
    add_image(s, PAPER_FIGS / "fig_runtime_breakdown.png", 6.9, 1.95, w=6.0, h=4.3)
    add_text(s, 6.9, 6.35, 6.0, 0.35,
             "Total 114.5 ms / pair  (8.7 Hz, CPU). Farneback is 87% of the bill.",
             size=10, color=GREY, font=BODY_FONT, align=PP_ALIGN.CENTER)
    add_footer(s, pg, TOTAL, "Results")

    # ====================================================================
    # SLIDE 14 - CONCLUSION + FUTURE WORK
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_header(s, "A small classical idea, a measurable win.",
               kicker="Conclusion")
    # left: takeaways
    add_text(s, 0.6, 2.05, 6.5, 0.4, "TAKEAWAYS",
             size=11, bold=True, color=BLUE, font=BODY_FONT)
    bullet_y = 2.5
    takeaways = [
        ("Image-space is enough.",
         "Correcting the projection, not the point cloud, works."),
        ("Events as a gate.",
         "Simulated DVS-style events cheaply localize where to fire."),
        ("Beats IMU baseline.",
         "On 23/24 KITTI sequences - dynamic objects matter."),
        ("Real-time-adjacent on CPU.",
         "8.7 Hz today; GPU flow takes us past 10 Hz."),
    ]
    for i, (head, sub) in enumerate(takeaways):
        y = bullet_y + i * 0.95
        add_rect(s, 0.6, y + 0.07, 0.12, 0.32, fill=BLUE)
        add_text(s, 0.85, y, 6.0, 0.4, head,
                 size=15, bold=True, color=DARK, font=TITLE_FONT)
        add_text(s, 0.85, y + 0.4, 6.0, 0.4, sub,
                 size=11, color=GREY, font=BODY_FONT)
    # right: future work
    add_rect(s, 7.7, 2.0, 5.2, 4.7, fill=LIGHT)
    add_text(s, 7.95, 2.15, 4.7, 0.4, "WHAT'S NEXT",
             size=11, bold=True, color=RED, font=BODY_FONT)
    add_text(s, 7.95, 2.55, 4.7, 0.5, "Future directions",
             size=20, bold=True, color=DARK, font=TITLE_FONT)
    future = [
        "Per-scan-angle temporal offset estimation.",
        "3D extension via depth-preserved unprojection.",
        "Validation on a real LiDAR-event dataset (DSEC).",
        "GPU Farneback / RAFT for full 10 Hz throughput.",
    ]
    for i, item in enumerate(future):
        y = 3.4 + i * 0.7
        add_text(s, 7.95, y, 0.3, 0.4, "->",
                 size=14, bold=True, color=RED, font=BODY_FONT)
        add_text(s, 8.25, y, 4.4, 0.6, item,
                 size=12, color=DARK, font=BODY_FONT, line_spacing=1.25)
    add_footer(s, pg, TOTAL, "Conclusion")

    # ====================================================================
    # SLIDE 15 - THANK YOU + REFERENCES
    # ====================================================================
    pg += 1
    s = add_blank(prs)
    add_accent_bar(s, 0.6, 1.5, w=0.5, h=0.08, color=BLUE)
    add_text(s, 0.6, 1.7, 8, 0.4, "EVALUATION 2  -  HONORS PROJECT",
             size=12, bold=True, color=BLUE, font=BODY_FONT)
    add_text(s, 0.6, 2.15, 8, 1.4, "Thank you.",
             size=54, bold=True, color=DARK, font=TITLE_FONT)
    add_text(s, 0.6, 3.45, 8, 0.5,
             "Questions and comments welcome.",
             size=15, color=GREY, font=BODY_FONT)
    add_text(s, 0.6, 4.15, 8, 0.4, "Sahal Ansar Theparambil",
             size=13, bold=True, color=DARK, font=BODY_FONT)
    add_text(s, 0.6, 4.5, 8, 0.4, "sahalansar.t23@iiits.in",
             size=11, color=GREY, font=BODY_FONT)
    # references mini block
    add_text(s, 0.6, 5.3, 8, 0.3, "KEY REFERENCES",
             size=10, bold=True, color=BLUE, font=BODY_FONT)
    refs = (
        "Geiger et al., KITTI dataset, IJRR 2013.\n"
        "Farneback, Two-frame motion estimation, SCIA 2003.\n"
        "Lichtsteiner et al., 128x128 DVS, JSSC 2008.\n"
        "Gallego et al., Event-based optical flow, TPAMI 2024."
    )
    add_text(s, 0.6, 5.65, 8, 1.6, refs,
             size=10, color=GREY, font=BODY_FONT, line_spacing=1.4)
    # right illustration
    add_image(s, ASSETS / "illust_thanks.png", 8.0, 2.4, w=5.0)
    add_footer(s, pg, TOTAL, "Thanks")

    # ====================================================================
    prs.save(str(OUT))
    print(f"[make_ppt] wrote {OUT}  ({TOTAL} slides)")


if __name__ == "__main__":
    build()
