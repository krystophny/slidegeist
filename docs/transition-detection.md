# Transition detection

Slidegeist detects slide changes from the video itself. It never chooses a
target slide count, assumes a fixed cadence, or stops after an expected number
of slides. The legacy `--scene-threshold` option is retained as a bounded
sensitivity bias only.

## Evidence review

The design is a dependency-light synthesis of primary work and maintained
official implementations:

- Wang et al.'s SSIM paper motivates structural rather than raw-pixel
  comparison: <https://doi.org/10.1109/TIP.2003.819861>.
- SliTraNet is lecture-specific and explicitly treats slide transition
  detection as distinct from generic shot detection:
  <https://arxiv.org/abs/2202.03540> and
  <https://github.com/asindel/SliTraNet>.
- TransNet V2 demonstrates the value of temporal evidence for generic shot
  boundaries, but its trained model and benchmark domain are deliberately not
  made runtime requirements here:
  <https://arxiv.org/abs/2008.04838> and
  <https://github.com/soCzech/TransNetV2>.
- PySceneDetect's maintained `ContentDetector` combines HSV and optional edge
  differences and its adaptive detector is preferred over a single global
  threshold:
  <https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/detectors/content_detector.py>.
- PySceneDetect 0.7's source-time overhaul confirms that presentation
  timestamps, not average-frame-rate arithmetic, are required for correct VFR
  boundaries:
  <https://github.com/Breakthrough/PySceneDetect/releases/tag/v0.7>.
- Adams and MacKay provide the probabilistic foundation for cadence-free
  change-point inference: <https://arxiv.org/abs/0710.3742>.

The lightweight ensemble uses local SSIM dissimilarity, perceptual-hash
distance, HSV-histogram distance, edge change, changed-tile coverage, and mean
luminance change. Robust median/MAD baselines adapt each channel to the video
and deliberately avoid an upper-percentile
term: in a transition-rich clip, real changes must not raise their own
threshold. A transition requires global colour or spatial support plus either
two structural channels or one structural channel backed by a strong ensemble
score. This rejects localized presenter motion and brightness-only flashes
better than a pixel threshold while retaining colour-dominant slide changes.
Local peak selection merges multiple samples from one transition; the
minimum-segment setting is only a duplicate-suppression window.
A final peak is retained only when at least one later sample supports the
minimum stable-scene window (allowing for sampling resolution). This rejects
recording-end fades and flashes that cannot represent a usable slide; it does
not impose a cadence or cap transitions elsewhere in the video.

FFmpeg selects the source-presentation-time evidence grid and downsizes those
frames before piping them to Python. Source presentation timestamps, rather
than average-FPS arithmetic, preserve correct boundaries for variable-frame-
rate material without moving every decoded full-resolution frame through
OpenCV.
On the retained 2,543-second real regression lecture, the sampler reproduced
all 70 prior transition bursts in 117.1 seconds (real-time factor 0.046),
versus about 51 minutes for the full-resolution Python decode path. Only two
selected peaks moved, each by one 0.48-second sample within the same transition
burst.

Speech pauses and audio are not used as transition evidence. They correlate
too weakly with slide changes and can fail completely in silent or edited
material. Transcript timestamps remain useful after visual segmentation.

## Instructional-state filtering

Visual change detection intentionally reports every supported state change; a
desktop, file browser, recording control, or operating-system dialog can
therefore be a genuine detected state without being a lecture slide. During
the existing multimodal description pass, Slidegeist classifies each extracted
state as `SLIDE` or `NON-SLIDE`. Rejected states are removed from the deck,
their time span is merged into the neighboring instructional interval, and
their classification plus image hash is retained in
`transition_detection.json`. The raw detector boundaries remain under
`raw_timestamps`, so this semantic filtering is reversible and auditable.

This second stage uses visual content rather than cadence or an expected slide
count. Handwritten pages and whiteboards explicitly count as instructional;
desktops, file choosers, menus, loading screens, recording controls, and pages
substantially obscured by operating-system UI do not.

## Reproducible acceptance oracle

`tests/test_transition_detector.py` generates videos with known, irregular
transition times, a moving presenter-like occluder, and a full-frame
luminance flash. It also includes a strong terminal visual change without a
stable hold, which must not create an extra slide. A separate sparse VFR
fixture has known presentation-time changes. Tests require every oracle
transition within the stated tolerance and no false positive at either flash.
This is independent behavioral ground truth, not a check that repository state
matches the patch.

Run the benchmark:

```bash
python scripts/benchmark_transitions.py --json benchmark.json
```

It reports precision, recall, F1, elapsed time, and real-time factor. Real
lecture distributions still differ from the synthetic oracle, so production
runs retain `transition_detection.json` with raw feature evidence and chosen
thresholds for audit and retuning.
