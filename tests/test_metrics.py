import pytest

from uav_relay_env.metrics import EpisodeMetrics


def test_episode_metrics_summary_contains_required_output_fields():
    metrics = EpisodeMetrics()
    info = {
        "rate_HR": 4.0e6,
        "rate_RL": 2.0e6,
        "rate_e2e": 1.0e6,
        "snr_HR": 10.0,
        "snr_RL": 20.0,
        "constraint_violation": True,
        "reward_terms": {"raw_outage_indicator": 1.0},
        "q_R": [1.0, 2.0, 3.0],
    }

    metrics.record(info, reward=3.5)
    summary = metrics.summary()

    assert summary["episode_length"] == 1
    assert summary["total_reward"] == pytest.approx(3.5)
    assert summary["avg_rate_e2e_bps"] == pytest.approx(1.0e6)
    assert summary["avg_rate_HR_bps"] == pytest.approx(4.0e6)
    assert summary["avg_rate_RL_bps"] == pytest.approx(2.0e6)
    assert summary["avg_snr_HR"] == pytest.approx(10.0)
    assert summary["avg_snr_RL"] == pytest.approx(20.0)
    assert summary["outage_count"] == 1
    assert summary["constraint_violation_count"] == 1
    assert summary["trajectory_length"] == 1
    assert summary["final_q_R"] == pytest.approx([1.0, 2.0, 3.0])


def test_empty_episode_metrics_summary_is_stable():
    summary = EpisodeMetrics().summary()

    assert summary["episode_length"] == 0
    assert summary["total_reward"] == pytest.approx(0.0)
    assert summary["avg_rate_e2e_bps"] == pytest.approx(0.0)
    assert summary["avg_rate_HR_bps"] == pytest.approx(0.0)
    assert summary["avg_rate_RL_bps"] == pytest.approx(0.0)
    assert summary["avg_snr_HR"] == pytest.approx(0.0)
    assert summary["avg_snr_RL"] == pytest.approx(0.0)
    assert summary["outage_count"] == 0
    assert summary["constraint_violation_count"] == 0
    assert summary["trajectory_length"] == 0
    assert summary["final_q_R"] == []
