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
- Adams and MacKay provide the probabilistic foundation for cadence-free
  change-point inference: <https://arxiv.org/abs/0710.3742>. A 2026
  lecture-video method combines visual embeddings with Bayesian online change
  points, reinforcing the direction while requiring a much heavier runtime:
  <https://doi.org/10.1016/j.procs.2026.06.556>.

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

Speech pauses and audio are not used as transition evidence. They correlate
too weakly with slide changes and can fail completely in silent or edited
material. Transcript timestamps remain useful after visual segmentation.

## Reproducible acceptance oracle

`tests/test_transition_detector.py` generates videos with known, irregular
transition times, a moving presenter-like occluder, and a full-frame
luminance flash. Tests require every oracle transition within 0.65 seconds
and no false positive at the flash. This is independent behavioral ground
truth, not a check that repository state matches the patch.

Run the benchmark:

```bash
python scripts/benchmark_transitions.py --json benchmark.json
```

It reports precision, recall, F1, elapsed time, and real-time factor. Real
lecture distributions still differ from the synthetic oracle, so production
runs retain `transition_detection.json` with raw feature evidence and chosen
thresholds for audit and retuning.
