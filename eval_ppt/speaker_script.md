# Speaker Script — Evaluation 2 Video
**Target runtime:** ≤ 8:00 · **Drafted for:** ~150 words / minute, natural pacing
**Conventions:**
- *Italic in parens* = visual cue (don't read aloud).
- **Bold** = land this word with stress.
- `(...)` inside spoken text = micro-pause / breath.
- Time per slide is a target, not a hard limit — feel free to breathe.

---

## Slide 1 — Title  ·  ~15 s  ·  ends @ 0:15
*Hold on the title slide while you introduce yourself.*

> Hi, I'm Sahal. For **Evaluation 2** of my honors project, I worked on a small but persistent problem in autonomous driving — getting LiDAR points and camera images to **actually line up** when the car is moving. Let me show you what that means.

---

## Slide 2 — A spinning LiDAR doesn't take a snapshot  ·  ~32 s  ·  ends @ 0:47
*Point at the sweep diagram — the rotating wedges, then the three cars on the right.*

> Here's the core problem. A spinning LiDAR — like the HDL-64E on a KITTI car — **doesn't take a snapshot** of the world. It fires sixty-four beams that sweep all the way around in **100 milliseconds**.
>
> So the point you measure at the **start** of the scan and the point you measure at the **end** were recorded a tenth of a second apart. At highway speed, the car has moved **1.67 meters** in that window. The world hasn't stopped either.
>
> And yet, when we project the scan onto an image, every point is treated as if it came from the **same instant**.

---

## Slide 3 — Misalignment compounds  ·  ~28 s  ·  ends @ 1:15
*Point at the three ghosted cars, then sweep to the three big stats on the right.*

> Why does this matter? 100 milliseconds sounds short. But it's enough for the ego car to drift, for another vehicle to roll forward, for a pedestrian to take a step.
>
> If we ignore it, **every** fusion task downstream — depth completion, object detection, mapping — starts from **misaligned** inputs. And the errors don't shrink as you go deeper into the stack. They **compound**.

---

## Slide 4 — Visual proof  ·  ~25 s  ·  ends @ 1:40
*Let the eye land on the image — point to a red dot, then the yellow arrow.*

> You can see this on a single real frame.
>
> The **red dots** are LiDAR points projected with KITTI's static calibration. The **yellow arrows** show where dense optical flow says those same points should sit in the very next frame.
>
> The gap between each dot and the tip of its arrow is the misalignment. And static calibration alone has **no way** to close it.

---

## Slide 5 — Related work  ·  ~42 s  ·  ends @ 2:22
*Walk through the three columns left-to-right.*

> There are basically three families of fixes.
>
> First — **IMU de-skew**. It integrates ego-motion between per-point timestamps. KITTI-sync doesn't expose those timestamps, so our baseline approximates them from the **sweep azimuth** using OXTS velocities and yaw rate — a faithful reimplementation, not a strawman. And it only models the **ego car** — not moving vehicles in the scene.
>
> Second — **deep fusion**. Transformers and CNNs. Strong benchmark numbers, but they need GPUs and large training sets.
>
> Our approach sits in a third lane. **Classical. Image-space. Event-guided.** No deep learning, no IMU, no hardware. Runs on a laptop CPU — and unlike the IMU, it captures **both** ego-motion **and** dynamic objects.

---

## Slide 6 — Recap  ·  ~28 s  ·  ends @ 2:50
*Point at the four panels in turn — projection, events, flow, corrected.*

> A quick recap from Evaluation 1.
>
> The pipeline was already running end-to-end: projection, simulated events, optical flow, event-gated correction. There was an interactive demo and a small validation harness on four sequences.
>
> The open questions coming into **this** evaluation were two. Does it **generalize** across many sequences? And does it **measurably** help downstream?

---

## Slide 7 — Approach: four stages  ·  ~38 s  ·  ends @ 3:28
*Trace the pipeline diagram top-to-bottom as you speak.*

> Here's the full method.
>
> We take two consecutive RGB frames. We compute **dense optical flow** between them. We simulate a DVS-style **event mask** from the log-intensity difference. We **smooth** the flow across time with an EMA.
>
> Then — and this is the important part — we apply the correction only at the **2D LiDAR projection**. We do **not** touch the 3D point cloud. Each projected point gets nudged by half the local flow vector — but **only** inside the event mask. The half-scale isn't a half-sweep derivation — KITTI-sync already interpolates returns to the trigger time — it's the empirical sweet spot from the ablation.

---

## Slide 8 — The trick: events as a gate  ·  ~34 s  ·  ends @ 4:02
*Hold on the event-overlay image; point at the white/blue events on object edges.*

> The most important design choice is the **gate**.
>
> Optical flow estimates something **everywhere** — even over flat asphalt. If we apply it blindly, we over-correct on static background.
>
> So instead, log of frame t-plus-one minus log of frame t, thresholded — the standard DVS response model — gives us a **cheap surrogate** for a real event camera. It lights up exactly where the scene changed. We add an edge-proximity filter on top, dilated by a seven-by-seven kernel.
>
> Now the correction **fires on objects** and stays **quiet on flat ground**.

---

## Slide 9 — Implementation  ·  ~26 s  ·  ends @ 4:28
*Quick pass across the four stage cards.*

> Each stage is short and standard. Projection uses KITTI's calibration chain. Events are a thresholded log-intensity difference, theta 0.3. Flow is OpenCV Farneback, smoothed with EMA alpha 0.7. The correction itself is just half-flow, gated on the mask.
>
> No deep learning. No GPU. Just NumPy and OpenCV.

---

## Slide 10 — Qualitative result  ·  ~24 s  ·  ends @ 4:52
*Eyes on the two crops — point at smeared edges on the left, snapped edges on the right.*

> Here's what that looks like.
>
> Same LiDAR scan, projected onto frame t-plus-one. On the **left**, uncorrected — points smeared along object edges. On the **right**, after the event-gated fix — points **snap** onto the actual scene boundaries.
>
> The difference is most pronounced exactly where you'd want it. **At the edges.**

---

## Slide 11 — Quantitative result  ·  ~44 s  ·  ends @ 5:36
*Glance at the chart, then point at each stat card in turn.*

> Quantitatively — 24 KITTI sequences, 6,584 frames.
>
> **SPEAS** — edge alignment — improves by 0.64 percent, with p below 0.001 and Cohen's d of 1.05. SPEAS uses the flow signal indirectly, so we treat it as a consistency check.
>
> **SRC** — stereo reprojection consistency — is the independent witness: the optical flow never **sees** the right camera. It improves by 0.37 percent, same significance, Cohen's d 0.90.
>
> And here's the headline. Against the **IMU-only baseline**, our method wins on **both** metrics in **23 of the 24 sequences**. The only one we don't win on is a parked sequence where neither method does anything.

---

## Slide 12 — Downstream: sharper depth  ·  ~32 s  ·  ends @ 6:08
*Point at the big green card.*

> The downstream signal is even stronger.
>
> We measured **BDPS** — Boundary Depth Projection Score — basically, how sharp depth edges are at object boundaries, via the Sobel gradient of the sparse depth raster — across 1,691 frames.
>
> Mean improvement: **plus 48.5 percent**. And **99.2 percent** of frames improve.
>
> The 48 percent looks huge next to the 0.64 on SPEAS, but the scales aren't comparable — Sobel on a sparse raster is dominated by zero pixels, so any geometric shift onto an edge swings it hard. The takeaway is the **sign and consistency**, frame after frame.

---

## Slide 13 — Ablation and runtime  ·  ~34 s  ·  ends @ 6:42
*Left chart first, then right.*

> The ablation on the left — on drive_0009, post-EMA-warmup — confirms every component pulls its weight. Drop the event gating, SRC goes **negative**. Push alpha to 0.75, both metrics regress below baseline. Half is the right scale.
>
> On the right — runtime. Total of **114.5 milliseconds** per frame pair on CPU. About **8.7 hertz** — close to the LiDAR rate. Farneback alone is 87 percent of the budget — so a GPU flow estimator would push us past 10 hertz immediately.

---

## Slide 14 — Conclusion and future work  ·  ~40 s  ·  ends @ 7:22
*Run down the four takeaways on the left, then gesture to the future-work panel.*

> Four takeaways.
>
> **One** — image-space correction is **enough**. You don't need to touch the 3D cloud.
>
> **Two** — even simulated events make a great **gate**.
>
> **Three** — the method beats an IMU baseline on **23 of 24** sequences, because dynamic objects matter.
>
> **Four** — it runs **near-real-time on a laptop**.
>
> Next steps: per-scan-angle offset estimation — replacing the scalar alpha with a function of azimuth — a 3D extension via depth-preserved unprojection, validation on a real event-camera dataset like DSEC, and GPU flow for full 10-hertz throughput.

---

## Slide 15 — Thank you  ·  ~10 s  ·  ends @ 7:32
*Hold on the thank-you slide; smile, brief pause.*

> That's all from me. Thanks for watching — and happy to take questions.

---

## Total target: **7:32 spoken**
- Leaves ~**28 s** of breathing room under the 8:00 cap.
- Distribute the buffer across transitions, screen pauses, and any one slide where you want to dwell a beat longer (Slide 4 or Slide 12 are good candidates).

## Recording tips
- **Lock pace on Slide 2** — if you rush the setup, the rest of the talk has nowhere to land. Take the full 30+ seconds.
- **Slide 11** has the densest numbers. Slow down for the *23 of 24* line specifically — it's the most quotable.
- **Slide 12** — when you hit the 48 vs 0.64 caveat, slow down. That's the line a prof is most likely to interrupt on.
- **Slide 13** — read "8.7 hertz" and "87 percent" with a small pause between, otherwise they blend.
- **Avoid filler words** when transitioning to a new slide. A clean half-second of silence is better than "so, um, …".
