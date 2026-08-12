import numpy as np

from safety_risk.random_diffusion_policy import RandomDiffusionConfig, RandomDiffusionPolicy


def test_random_diffusion_policy_is_reproducible_and_bounded():
    cfg = RandomDiffusionConfig(seed=7, max_joint_delta=0.02)
    q = np.zeros(12)
    first = RandomDiffusionPolicy(cfg)
    second = RandomDiffusionPolicy(cfg)

    target_a, delta_a = first.predict_joint_target(q)
    target_b, delta_b = second.predict_joint_target(q)

    np.testing.assert_allclose(target_a, target_b)
    np.testing.assert_allclose(delta_a, delta_b)
    assert np.max(np.abs(delta_a)) <= 0.02
    assert np.isfinite(target_a).all()


def test_random_diffusion_policy_clips_joint_limits():
    policy = RandomDiffusionPolicy(RandomDiffusionConfig(seed=9))
    q = np.full(12, 0.1)
    target, _ = policy.predict_joint_target(q, np.full(12, -0.1), np.full(12, 0.1))
    assert np.all(target >= -0.1)
    assert np.all(target <= 0.1)
