import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402

SHORT = {"simulation_days": 1, "day_end": "12:30", "llm": {"backend": "mock", "max_workers": 4},
         "latent_events": {"event_rate": 0.35, "holdout_from_day": None}}


def run_sim(tmp: Path, overrides=None, name="run"):
    from backend.config import deep_merge
    from backend.simulation.engine import Simulation
    cfg = load_config("configs/baseline.yaml", deep_merge(SHORT, overrides or {}))
    sim = Simulation(cfg, tmp / name, progress=False)
    sim.run()
    return sim


@pytest.fixture(scope="session")
def mock_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    sim = run_sim(tmp)
    return sim
