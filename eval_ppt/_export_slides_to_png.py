"""Quick helper: open the generated .pptx with COM and export each slide as PNG
into ./assets/_slide_preview/ so we can visually inspect the deck.
"""
import os
import sys
from pathlib import Path
import win32com.client as win32

HERE = Path(__file__).resolve().parent
PPTX = HERE / "eval_presentation.pptx"
OUT_DIR = HERE / "assets" / "_slide_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean stale exports
for p in OUT_DIR.glob("*.png"):
    p.unlink()

ppt = win32.Dispatch("PowerPoint.Application")
ppt.Visible = True  # PowerPoint requires this when opening for some ops
deck = ppt.Presentations.Open(str(PPTX), WithWindow=False)
try:
    for i, slide in enumerate(deck.Slides, start=1):
        out = OUT_DIR / f"slide_{i:02d}.png"
        slide.Export(str(out), "PNG", 1600, 900)
        print(f"  wrote {out.name}")
finally:
    deck.Close()
    ppt.Quit()
print("[export] done.")
