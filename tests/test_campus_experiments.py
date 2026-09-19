"""Campus pipeline contracts, including supplied-origin evidence and recoverability."""
import json
from pathlib import Path

import pytest
import yaml

from backend.config import ConfigKeyWarning, load_config
from backend.experiment import design as D
from backend.research import experiments as E
from backend.research.common import read_json
from backend.research.evidence import Evidence
from backend.simulation.engine import Simulation, trace_digest


def test_campus_reference_and_presets():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConfigKeyWarning)
        for entry in E.catalog():
            cells = D.expand(E.resolve(entry['id']))
            assert entry['environment'] == 'campus'
            for cell in cells:
                cfg = cell.config()
                assert not any(cfg[k]['enabled'] for k in ('workshop', 'records', 'roster', 'turnover'))
    cfg = load_config('configs/memetics.yaml')
    assert cfg['population'].endswith('homewood8.yaml')
    assert cfg['latent_events']['event_rate'] == 0


def test_shared_markdown_in_real_prompts_and_replay_without_source(tmp_path):
    source = tmp_path / 'intro.md'
    markdown = '# Shared world\nA café tradition: leave a little room. 世界\n'
    source.write_text(markdown, encoding='utf-8')
    cfg = load_config('configs/memetics.yaml', {'population_size': 4, 'simulation_days': 1,
        'day_start': '11:00', 'day_end': '13:00', 'llm': {'backend': 'mock'},
        'analysis': {'observer': {'backend': 'mock'}}, 'shared_background': {'file': str(source)}})
    sim = Simulation(cfg, tmp_path / 'original', progress=False)
    sim.run()
    assert all(markdown in a.iss() for a in sim.agents.values())
    assert markdown in (tmp_path / 'original/shared_background.md').read_text(encoding='utf-8')
    calls = [json.loads(line) for line in (sim.run_dir / 'llm_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(markdown in c['prompt'] for c in calls)
    index = Evidence(sim.run_dir)
    index.build()
    supplied = [e for e in index.events() if e['kind'] == 'shared_background']
    assert len(supplied) == 4 and all(e['channel'] == 'initial' for e in supplied)
    assert not any(e['via'] == 'heard' and e['source'] in {s['id'] for s in supplied} for e in index.exposures())
    source.unlink()
    resolved = yaml.safe_load((sim.run_dir / 'config.resolved.yaml').read_text(encoding='utf-8'))
    replay = Simulation(resolved, tmp_path / 'replay', replay_from=sim.run_dir / 'llm_calls.jsonl', progress=False)
    replay.run()
    assert trace_digest(replay.run_dir) == trace_digest(sim.run_dir)


def test_custom_design_runs_subprocesses_and_reuses_results(tmp_path):
    source = tmp_path / 'background.md'
    source.write_text('# Before arrival\nA shared story about patient listening.', encoding='utf-8')
    design = E.configure(E.resolve('emergence', tmp_path), population_size=4, days=1, background=str(source))
    # Freeze source edits before workers start; they must use the reviewed design text.
    source.write_text('A completely different text.', encoding='utf-8')
    cells = D.select(D.expand(design, 'mock'), seeds=[11])
    assert 'patient listening' in cells[0].config()['shared_background']['markdown']
    result = E.run(design, cells, backend_override='mock')
    assert result['status'] == 'complete', result
    assert (Path(result['report_dir']) / 'report.html').exists()
    assert list((D.design_dir(design, 'mock') / 'inquiries').glob('*/report.html'))
    trace = cells[0].run_dir / 'trace.jsonl'
    before = trace.stat().st_mtime_ns
    again = E.run(design, cells, backend_override='mock')
    assert again['status'] == 'complete' and again['execution'] == []
    assert trace.stat().st_mtime_ns == before
    # A new observer setting reuses the society while making a compatible new analysis.
    design.common.setdefault('analysis', {})['memetics'] = {'window_chars': 8000}
    cell = D.expand(design, 'mock')[0]
    assert D.cell_status(cell) == 'stale'
    assert D.run_cell(cell) == 'analyzed'
    assert read_json(cell.run_dir / 'analysis.json')['observer']['window_chars'] == 8000
    assert trace.stat().st_mtime_ns == before


def test_background_changes_invalidate_existing_cell(tmp_path):
    source = tmp_path / 'intro.md'
    source.write_text('Original background.', encoding='utf-8')
    design = E.resolve('emergence', tmp_path)
    design.days = 1
    design.common.update(population_size=2, day_start='11:00', day_end='11:30',
                         shared_background={'file': str(source)})
    cell = D.expand(design, 'mock')[0]
    assert D.run_cell(cell) == 'analyzed'
    source.write_text('Changed background.', encoding='utf-8')
    assert D.cell_status(cell) == 'incompatible'


def test_paused_design_continues_and_remains_compatible(tmp_path):
    design = E.resolve('emergence', tmp_path)
    design.days = 1
    design.common.update(population_size=3, day_start='11:00', day_end='12:00')
    cell = D.expand(design, 'mock')[0]
    cell.run_dir.mkdir(parents=True)
    (cell.run_dir / 'pause.request').write_text('pause', encoding='utf-8')
    sim = Simulation(cell.config(), cell.run_dir, progress=False)
    sim.run()
    assert D.cell_status(cell) == 'paused'
    assert D.run_cell(cell) == 'analyzed'
    manifest = read_json(cell.run_dir / 'manifest.json')
    assert manifest['continuation']['prefix_verified']
    assert manifest['status'] == 'finished'
    assert cell.run_dir.with_name(cell.run_dir.name + '.failed1').exists()


@pytest.mark.parametrize('value', ['', '  ', 42, []])
def test_bad_background_fails_before_simulation(value):
    with pytest.raises(ValueError, match='nonempty Markdown'):
        load_config(overrides={'shared_background': {'markdown': value}})


def test_api_preview_and_launch_share_custom_design(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api import server
    from backend.api import research
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    launched = []
    monkeypatch.setattr(research.jobs, 'launch', lambda root, args, kind: launched.append(args) or {'id': 'test'})
    client = TestClient(server.app)
    body = {'action': 'experiment', 'experiment': 'shared_background', 'backend': 'mock',
            'population': 'configs/population/homewood500.yaml', 'population_size': 4, 'days': 1,
            'background_markdown': '# A shared world\nAn ordinary campus story.'}
    preview = client.post('/api/research/experiment-preview', json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()['population_size'] == 4 and preview.json()['days'] == [1]
    assert client.post('/api/research/jobs', json=body).status_code == 200
    launched_design = D.load_design(launched[0][2], tmp_path)
    assert launched_design.name == preview.json()['name']
    cells = D.expand(launched_design, 'mock')
    absent, supplied = cells[:2]
    assert absent.config()['shared_background']['markdown'] is None
    assert supplied.config()['shared_background']['markdown'] == body['background_markdown']
    response = client.post('/api/research/experiment-preview', json={**body, 'population': '../outside.yaml'})
    assert response.status_code == 400


def test_preflight_rejects_bad_population_and_missing_live_provider(tmp_path, monkeypatch):
    design = E.configure(E.resolve('emergence', tmp_path), population_size=1000)
    assert not E.check(design, 'mock')['ready']
    monkeypatch.setattr(E.shutil, 'which', lambda name: None)
    monkeypatch.setenv('OPENAI_API_KEY', '')
    assert any('.env' in error for error in E.check(E.resolve('emergence', tmp_path))['errors'])
    assert E.check(E.resolve('emergence', tmp_path), 'mock')['ready']


@pytest.mark.parametrize('committed', [False, True])
def test_resume_before_first_tick_or_after_last_tick(tmp_path, committed):
    design = E.resolve('emergence', tmp_path)
    design.days = 1
    design.common.update(population_size=2, day_start='11:00', day_end='11:15')
    cell = D.expand(design, 'mock')[0]
    sim = Simulation(cell.config(), cell.run_dir, progress=False)
    if committed:
        (cell.run_dir / 'pause.request').write_text('pause', encoding='utf-8')
        sim.run()
    else:
        # Authentication failure during seed initialization leaves no committed tick.
        from backend.llm.client import LLMUnavailable
        sim._seed_memories = lambda: (_ for _ in ()).throw(LLMUnavailable('missing key'))
        with pytest.raises(LLMUnavailable):
            sim.run()
    assert D.cell_status(cell) == 'paused'
    assert D.run_cell(cell) == 'analyzed'
    manifest = read_json(cell.run_dir / 'manifest.json')
    if committed:
        assert manifest['continuation']['prefix_verified']
    assert manifest['status'] == 'finished'

