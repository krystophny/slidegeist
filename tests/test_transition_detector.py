"""Behavioral-oracle tests for robust slide-transition detection."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from slidegeist.transition_detector import analyze_slide_transitions
from tests.synthetic_video import make_slide_video, make_vfr_slide_video


def _match(detected: list[float], expected: list[float], tolerance: float = 0.65) -> None:
    assert len(detected) == len(expected), (detected, expected)
    for actual, oracle in zip(detected, expected, strict=True):
        assert abs(actual - oracle) <= tolerance, (actual, oracle)


def test_variable_cadence_with_presenter_and_flicker(tmp_path: Path) -> None:
    """Known transitions survive occlusion and luminance noise without a rate prior."""
    oracle = [2.7, 8.9, 11.0, 14.6]
    video = tmp_path / "oracle.avi"
    make_slide_video(video, oracle, duration=17.0)

    analysis = analyze_slide_transitions(
        video,
        threshold=0.025,
        start_offset=0.0,
        min_scene_len=0.8,
    )

    _match(analysis.timestamps, oracle)
    assert all(abs(timestamp - 6.2) > 0.65 for timestamp in analysis.timestamps)


def test_same_result_when_expected_rate_would_be_wrong(tmp_path: Path) -> None:
    """A long hold and a rapid pair are both detected; count is evidence-driven."""
    oracle = [1.5, 2.8, 12.4]
    video = tmp_path / "irregular.avi"
    make_slide_video(video, oracle, duration=15.0, luminance_flicker_at=None)

    analysis = analyze_slide_transitions(
        video,
        threshold=0.025,
        start_offset=0.0,
    )

    _match(analysis.timestamps, oracle)


def test_sampler_decodes_long_hold_in_single_pass(tmp_path: Path) -> None:
    """A longer fixture remains comfortably faster than real time."""
    video = tmp_path / "long-hold.avi"
    make_slide_video(video, [3.0, 17.0], duration=20.0)

    started = monotonic()
    analysis = analyze_slide_transitions(video, start_offset=0.0)
    elapsed = monotonic() - started

    _match(analysis.timestamps, [3.0, 17.0])
    assert elapsed < 10.0


def test_dense_irregular_transitions_do_not_raise_their_own_threshold(tmp_path: Path) -> None:
    """A transition-rich clip must not poison a count-independent baseline."""
    oracle = [1.0, 2.2, 3.5, 5.0]
    video = tmp_path / "dense.avi"
    make_slide_video(
        video,
        oracle,
        duration=6.5,
        moving_occluder=False,
        luminance_flicker_at=None,
    )

    analysis = analyze_slide_transitions(video, start_offset=0.0)

    _match(analysis.timestamps, oracle)


def test_recording_end_fade_does_not_create_a_slide(tmp_path: Path) -> None:
    """A visually strong terminal state without a stable hold is not a slide."""
    video = tmp_path / "terminal-flash.avi"
    make_slide_video(
        video,
        [3.0, 9.4],
        duration=10.0,
        moving_occluder=False,
        luminance_flicker_at=None,
    )

    analysis = analyze_slide_transitions(video, start_offset=0.0)

    _match(analysis.timestamps, [3.0])


def test_variable_frame_rate_uses_source_presentation_times(tmp_path: Path) -> None:
    """Sparse VFR frames retain their known timeline instead of average-FPS timing."""
    oracle = [1.52, 5.52]
    video = tmp_path / "vfr.mkv"
    make_vfr_slide_video(video, oracle)

    analysis = analyze_slide_transitions(video, start_offset=0.0, sample_interval=0.5)

    _match(analysis.timestamps, oracle, tolerance=0.08)
