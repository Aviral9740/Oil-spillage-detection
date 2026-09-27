# Oil Spill Detection and AIS Vessel Attribution

**Smart India Hackathon 2026 — Problem Statement 143**

This project started as a straightforward computer vision problem — find oil slicks in
satellite imagery — and turned into something much more interesting: a real, end-to-end
pipeline that goes from raw Synthetic Aperture Radar data all the way to a georeferenced
alert with a suspect vessel attached to it. Along the way it forced me to rethink the model,
rebuild the entire inference chain twice, and chase down bugs that had nothing to do with
machine learning and everything to do with getting the fundamentals right. This README is my
honest account of what's actually in this repository, how it got here, and where it still
falls short.

## What this project actually does

Given a Sentinel-1 SAR satellite scene, the system:

1. Detects oil spill regions in the scene using a trained segmentation model.
2. Converts the detected pixel regions into real, georeferenced polygons (actual latitude
   and longitude, not just pixel coordinates).
3. Estimates a confidence score and a rough age for each detected slick.
4. Correlates the spill's location and estimated timing against AIS (Automatic
   Identification System) vessel tracking data, to flag ships that were in the area shortly
   before the satellite pass — the actual "attribution" half of the problem statement.
5. Renders an evidence image for each detection — the actual SAR scene with the model's
   detected region outlined — and hosts it so it can be referenced from the frontend.
6. Feeds all of this into a Next.js frontend with a deck.gl and MapLibre-powered 3D globe,
   where detected spills, their forecasted drift, and correlated vessels are visualized
   together.

None of these pieces were obvious from the start, and most of them went through at least
one complete rebuild.

## The road here: why the model changed twice

The original approach used YOLO for bounding-box detection, trained on a dataset limited to
the Eastern Mediterranean Sea. It worked reasonably well on that region. It also came with a
full supporting pipeline already built — an ONNX inference engine, a bounding-box-to-GeoJSON
georeferencing module using bilinear corner interpolation, a Cloudinary image upload path,
and a backend JSON payload builder. That infrastructure turned out to be worth keeping; the
model behind it did not.

The problem became obvious once I looked for a dataset that wasn't limited to one sea. The
Zenodo Sentinel-1 SAR oil spill dataset is genuinely global — Gulf of Mexico, Eastern and
Western Mediterranean, Persian Gulf, West Africa, North Sea, Red Sea, Iberian Atlantic,
Southeast Asia, the Caribbean — real geographic diversity, which is exactly what a detector
meant to run anywhere in the world actually needs. But this dataset ships as raw two-band
GeoTIFF rasters (Sigma0 dB, VV and VH polarizations), not JPEGs, and YOLO's whole pipeline was
built around plain 8-bit images. Rather than force a mismatch, I moved to a segmentation
architecture that could work with the data as it actually exists: a U-Net++ with a ResNet-34
encoder, trained directly on the raw radar bands.

## The model

**Architecture:** U-Net++ (ResNet-34 encoder, ImageNet-initialized, no pretrained decoder),
single-channel sigmoid output, trained at 512x512 input resolution.

**Input: 3 channels, each one earning its place**

1. Normalized VV band (co-polarization — vertical transmit, vertical receive)
2. Normalized VH band (cross-polarization — vertical transmit, horizontal receive)
3. `clip(VV_normalized - VH_normalized, 0, 1)` — a polarization-difference channel

That third channel isn't padding to satisfy a pretrained encoder's expected channel count
(though it does that too). Oil suppresses fine capillary ocean waves differently across
polarizations than calm water or biogenic slicks do, and the explicit difference between
co- and cross-polarization gives the model a direct physical contrast signal to separate real
petroleum from look-alikes — low-wind calm zones, algae blooms, and similar false triggers
that plague SAR-based oil spill detection generally.

**Loss:** a compound loss — 40% binary cross-entropy, 60% Focal Tversky loss
(alpha 0.3, beta 0.7 in the original run) — chosen specifically because Tversky loss lets you
directly control the false-positive/false-negative tradeoff, which matters a great deal for a
disaster-detection tool where a missed spill is a materially worse outcome than a false alarm.

**Normalization bounds**, per band, empirically fit and independently verified rather than
assumed:

```
VV:  DB_MIN = -48.28, DB_MAX = -26.77
VH:  DB_MIN = -30.29, DB_MAX = -14.44
```

Raw value `0.0` is treated as a no-data / outside-swath sentinel and mapped directly to a
normalized value of `0`, rather than run through the clip-scale formula (where it would
otherwise be misread as near-maximum backscatter — the opposite of what it actually means).

## Data

Training and validation used Parts I and II of the Zenodo dataset combined:

- **Part I** — 1,200 oil-positive scenes with masks, 2048x2048, Sigma0 dB.
- **Part II** — 685 no-oil and 685 look-alike (hard negative) scenes, masks confirmed
  all-zero.

**Part III** — 450 scenes (150 each of oil, no-oil, and look-alike) — was held out entirely
as a true test set, never touched during training or validation. Every accuracy number in
this README that matters is measured against Part III specifically, because it's the only
honest measure of how this generalizes to scenes the model has never seen.

Zenodo ships none of these scenes with acquisition metadata. Acquisition timestamps are
recovered by scanning the raw TIFF bytes for an embedded Sentinel-1 product ID string (a
leftover from the SNAP processing chain) — a genuinely useful trick a teammate found, since
without it there'd be no way to time-correlate a detection against AIS vessel tracks at all.

## Training — the bugs that mattered as much as the architecture

Two bugs in the original training loop were quietly undermining everything until they were
found and fixed:

- **A crop-anchoring crash.** The original logic required a random training crop to fully
  contain a labeled oil bounding box, which threw an empty-range error on any spill larger
  than the crop size. Fixed to require only overlap between the crop and the labeled region.
- **A silent checkpointing bug.** Validation IoU and F1 were being averaged per-batch. Any
  batch with zero foreground pixels produces a `0/0` division — `nan` — and since
  `nan > best_iou` is always `False` in Python, the checkpointing logic silently stopped
  saving improved models without ever raising an error. The fix was to accumulate true
  positives, false positives, false negatives, and true negatives across the *entire*
  validation set first, and compute the ratio once at the end.

The current checkpoint (epoch 23) reached a validation IoU of 0.6576. On the held-out Part
III test set, at a decision threshold of 0.85: overall IoU 0.6858, oil-region IoU 0.5108,
recall 0.627, precision 0.744, and a 22.67 percent false-alarm rate on the negative
(no-oil/look-alike) scenes.

## Making the exported model trustworthy

Getting a trained PyTorch checkpoint into a fast, production-servable ONNX model that
*actually behaves the same way* turned into its own investigation, and I want to document it
honestly because it's the part of this project I'm most proud of getting right rather than
just getting working.

**The band order was wrong, and the evidence was clean enough to prove it.** An earlier
assumption held that raw Zenodo TIFFs ship their two bands as `[VH, VV]` and needed flipping
to `[VV, VH]`. Empirically fitting the real per-band normalization bounds against a known-good
reference output showed the opposite: reading these files with `rasterio` already returns
`[VV, VH]` natively on this reader, and the "fix" was actually introducing a swap, evidenced
by a clean R-squared jump from about 0.003 to about 0.84 once the pairing was corrected.

**The ONNX export itself had a stale-checkpoint bug.** An early export cell had no explicit
checkpoint-loading step of its own — it silently exported whatever model happened to be sitting
in memory from earlier cells, which is fine until a kernel restart or an out-of-order re-run
means that's the wrong model. The fix was to make every export cell fully self-contained:
reload the checkpoint explicitly, print a parameter fingerprint to compare across sessions,
run a PyTorch-versus-ONNX-Runtime numerical parity check, and check the logit range on a real
reference scene before trusting the output.

**End-to-end verification, not just unit checks.** Rather than trust each fix in isolation,
the full chain — raw GeoTIFF, through the corrected preprocessing, through the exported ONNX
model, through thresholding and connected-component filtering — was checked against the
original notebook's own output on the same scene. Final result: a mask intersection-over-union
of 0.9877 between the two pipelines. That's the number that actually let me trust the
production path.

## The production pipeline

`export_payload_unet.py` is the real inference entry point. For each scene, it:

1. Reads the raw two-band GeoTIFF and builds the 3-channel model input.
2. Runs inference through the ONNX model.
3. Thresholds the probability map and removes small speckle-noise blobs via connected-
   component filtering (anything under 100 pixels is discarded — this alone eliminates a
   meaningful share of false positives that are just SAR speckle, not real detections).
4. Rescales the affine geotransform to account for the resize-to-512 inference resolution,
   then reprojects the resulting pixel-space contours into real WGS84 latitude/longitude
   polygons.
5. Extracts the acquisition timestamp from the embedded Sentinel-1 product ID.
6. Computes a mean confidence score per detected polygon.
7. Renders and uploads a real evidence image (see below), and assembles everything into the
   same backend JSON schema the original YOLO pipeline already used — meaning nothing
   downstream had to change to accommodate the new model.

## Evidence images

Every detection includes a real, uploaded image showing the SAR scene with the model's
actual predicted region outlined in red — not a placeholder, and not a ground-truth mask
(which wouldn't exist for a genuinely new scene anyway). Getting this to look like something
a human being could actually read took real iteration: raw Sigma0 SAR is inherently speckled
at the pixel level, and stretching contrast without denoising first only makes that speckle
more visible. The current approach denoises the raw radar values first — a Gaussian blur
approximating real multi-look averaging — and only then computes the display contrast
stretch, so the stretch itself isn't calibrated against noise-inflated extremes. The
overlay is drawn from the exact same filtered detection mask that produces the JSON output,
so the image can never show a detection that isn't actually reported.

## AIS vessel attribution

`ais_correlation.py` takes a detected spill's polygon and acquisition time, and searches AIS
vessel tracking data for any vessel whose position or trajectory intersects the spill area in
the hours beforehand — a 12-hour lookback window by default, checking both single-ping
positions and full trajectory lines against a buffered spill polygon. This is the piece that
turns "we found oil" into "here are the vessels that were in the area," which is the actual
attribution half of the problem statement.

## Frontend

Built on Next.js, TypeScript, and Node.js. The map is powered by deck.gl and MapLibre, with
3D ship models rendered from GLB/glTF assets. Wind and ocean current visualization runs as a
custom particle system (built from scratch after two off-the-shelf particle libraries
conflicted with the installed deck.gl version), fed by a wind/current data grid generated
offline from Open-Meteo and bundled as a static asset rather than fetched live, after running
into daily API rate limits during testing. Both dark and light themes are supported, and the
particle layer toggles on with a single interaction rather than running always-on. Spill drift
forecasting visualizes the backend's own predicted trajectory data directly, rather than
running a separate physics simulation on the frontend.

## Where this honestly stands right now

The model is not done, and I'd rather say that plainly than let a headline IoU number imply
otherwise. Direct inspection of the held-out test set found real scenes — genuine, labeled
oil spills — where the model's raw output probability inside the true oil region never rises
above roughly 0.10. That's not a threshold that's set too conservatively; a full sweep down to
0.35 recovered none of these scenes, which rules out a tuning fix. It's a real gap in what
the model has learned to recognize.

I also seriously considered falling back to the original YOLO model on scenes like these,
reasoning that even an imprecise bounding box would be better than nothing. I tested that
directly rather than assume it — including on a scene the U-Net model detected confidently and
correctly — and YOLO produced zero detections across the board. That's consistent with its own
training history (Eastern Mediterranean only) rather than any genuine complementary skill on
this data, and a wrong, confidently-placed detection would arguably be worse for this tool than
an honest gap, so I decided against wiring it in as a fallback.

The real fix in progress is a recall-focused continuation of training from the same epoch-23
checkpoint, with the Tversky loss's false-negative penalty raised (beta 0.7 to 0.8), while
keeping the verified preprocessing and export pipeline completely untouched, so anything that
comes out of it can be validated against the exact same evidence-based process that got the
current model to a trustworthy state.

## Repository structure

```
ml_layer/
  dataset.py                  Original YOLO-era dataset loader
  train_yolo.py                YOLO training entry point
  export_yolo.py               Bounding-box dataset export for YOLO format
  inference_engine.py          ONNX Runtime engines: OnnxYoloEngine and OnnxUnetEngine
  export_payload_unet.py       Production U-Net segmentation-to-JSON pipeline
  evidence_image.py            Renders and uploads real detection evidence images
  georeference.py              Bbox-to-GeoJSON via bilinear corner interpolation (YOLO path)
  export_payload.py            Shared backend JSON payload schema builder
  ais_correlation.py           Suspect vessel identification from AIS trajectory data
  generate_synthetic_ais.py    Synthetic AIS data generation for testing
  main_pipeline.py             Batch historical-image processing (YOLO path)
  api.py                       FastAPI serving endpoint (YOLO path)
  visualize_ais.py             AIS trajectory and spill intersection plotting
  visualize_val.py             Model prediction visualization on validation images
```

## A closing thought

The part of this project I'd actually want a judge to look closely at isn't the headline
metric — it's the amount of the work that had nothing to do with the model architecture at
all. Getting a segmentation mask into a real, correctly-placed latitude and longitude on a
map is its own small research problem. So is proving that an exported model actually behaves
like the one you trained, instead of assuming it does because the export command didn't
error. Most of the debugging in this repository's history is exactly that kind of work, and I
think that's the part that actually determines whether a tool like this is safe to put in
front of a real disaster-response decision, not just a demo.