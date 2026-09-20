"""The selected studies use an unseeded campus and the requested inexpensive model."""
import json
from fastapi.testclient import TestClient
from backend.config import load_config
from backend.research import experiments as E
from backend.experiment import design as D
from backend.simulation.engine import Simulation


def test_selected_studies_share_unseeded_setup():
    counts = {'emergence': 1, 'communities': 4, 'influence': 6, 'social_meaning': 4}
    assert {e['id'] for e in E.catalog() if e.get('featured')} == set(counts)
    for study, count in counts.items():
        cells = D.expand(E.resolve(study))
        assert len(cells) == count
        for cell in cells:
            cfg = cell.config()
            assert cfg['population_size'] == 100 and cfg['simulation_days'] == 3
            assert cfg['initial_memories_file'] is None
            assert cfg['shared_background']['markdown'] is None
            assert cfg['controls']['planted_phrase'] is None
            assert cfg['llm']['model'] == cfg['analysis']['observer']['model'] == 'gpt-5-nano'


def test_homewood500_baseline_does_not_seed_cafe_names(tmp_path):
    cfg = load_config('configs/campus100.yaml', {'population_size': 2, 'simulation_days': 1,
        'day_start': '11:00', 'day_end': '11:15', 'llm': {'backend': 'mock'}})
    sim = Simulation(cfg, tmp_path / 'unseeded', progress=False)
    sim.run()
    events = [json.loads(line) for line in (sim.run_dir / 'trace.jsonl').read_text().splitlines()]
    assert not any(e.get('initial_memory') or e['type'] == 'shared_background' for e in events)
    assert all('FFC' not in a.iss() and 'Hopkins Cafe' not in a.iss() for a in sim.agents.values())
    assert load_config('configs/homewood500_naming.yaml')['initial_memories_file'] is not None


def test_frontend_preview_exposes_actual_starting_material(tmp_path, monkeypatch):
    from backend.api import server
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    result = TestClient(server.app).post('/api/research/experiment-preview', json={
        'action': 'experiment', 'experiment': 'emergence', 'backend': 'mock'})
    assert result.status_code == 200
    data = result.json()
    assert data['population_size'] == 100
    assert data['initial_memories_file'] is None and not data['shared_background']
    assert data['active_hours'] == '11:00–20:00'
