import numpy as np

from safety_risk.perception_degradation import PerceptionDegradationInjector


def test_flag_one_changes_actual_rgb_and_records_hash_evidence():
    image = np.full((10, 20, 3), 128, dtype=np.uint8)
    observations = {"cameras": {"split_aloha_head": {"color_image": image.copy()}}}
    injector = PerceptionDegradationInjector({
        "perception_degradation_injection_flag": True,
        "start_frame": 2,
        "end_frame": 4,
        "affected_cameras": ["split_aloha_head"],
        "corruption_type": "black_occlusion_and_noise",
        "occlusion_fraction": 0.6,
        "noise_std": 90.0,
        "seed": 7,
    })
    injector.apply(observations, 2)
    changed = observations["cameras"]["split_aloha_head"]["color_image"]
    assert not np.array_equal(image, changed)
    audit = injector.audit_log()
    assert audit["actual_corruption_applied"] is True
    assert audit["actual_start_frame"] == 2
    assert audit["actual_end_frame"] == 2
    assert audit["frames"][0]["before_sha256"] != audit["frames"][0]["after_sha256"]
    assert audit["frames"][0]["changed_pixels"] > 0
    assert audit["frames"][0]["stored_rgb_key_suffix"].endswith("/0002")


def test_flag_zero_preserves_rgb_and_has_no_affected_frames():
    image = np.full((4, 4, 3), 64, dtype=np.uint8)
    observations = {"cameras": {"split_aloha_head": {"color_image": image.copy()}}}
    injector = PerceptionDegradationInjector({
        "perception_degradation_injection_flag": False,
    })
    injector.apply(observations, 0)
    assert np.array_equal(image, observations["cameras"]["split_aloha_head"]["color_image"])
    assert injector.audit_log()["actual_corruption_applied"] is False


def test_episode_reset_discards_rejected_attempt_frames():
    injector = PerceptionDegradationInjector({
        "perception_degradation_injection_flag": True,
        "start_frame": 0,
        "end_frame": 1,
        "affected_cameras": ["split_aloha_head"],
        "corruption_type": "black_occlusion_and_noise",
    }, seed=7)
    first = {"cameras": {"split_aloha_head": {
        "color_image": np.full((4, 4, 3), 100, dtype=np.uint8),
    }}}
    injector.apply(first, 0)
    assert injector.audit_log()["affected_frame_count"] == 1

    injector.reset_episode()
    assert injector.audit_log()["affected_frame_count"] == 0
    second = {"cameras": {"split_aloha_head": {
        "color_image": np.full((4, 4, 3), 100, dtype=np.uint8),
    }}}
    injector.apply(second, 0)
    assert injector.audit_log()["affected_frame_count"] == 1
