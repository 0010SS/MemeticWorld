"""The four conditions share frequent carrier habits and the existing injection mechanisms."""
from copy import deepcopy

from backend.config import load_config
from backend.research import experiments as E
from backend.experiment import design as D
from backend.agents.profile import load_population, apply_planted
from backend.agents import topology as TOPO
from backend.simulation.rngs import seed_rng
from backend.simulation import incidents as INC
from backend.analysis.battery.registry import usage_pattern


PHRASES = ['seven six', 'rizz', 'lock in', 'aura farming']


def test_four_conditions_preserve_injection_settings(tmp_path):
    base = load_config('configs/homewood100_memes.yaml')
    design = E.resolve('communication_communities', tmp_path)
    cells = D.expand(design)
    assert len(cells) == 4
    assert {c.seed for c in cells} == {42}
    expected = load_config('configs/homewood100_memes_communication.yaml')['memes']['registry']
    assert [entry['phrase'] for entry in expected] == PHRASES
    for entry, original in zip(expected, base['memes']['registry']):
        assert entry['id'] == original['id']
        assert entry['seeds'] == original['seeds']
        assert entry['habit'].startswith('frequently ')
        assert 'sometimes' not in entry['habit']
        assert 'six seven' not in entry['habit']
        assert entry['incident'] is None
        assert entry['grounding'] == 'ungrounded'
        assert entry['breadth'] == 'broad'
        assert all(word not in entry['habit'] for word in ('greasy', 'fume hood', 'standing note'))
    for cell in cells:
        cfg = cell.config()
        assert cfg['memes']['registry'] == expected
        for key in ('population', 'population_size', 'initial_memories_file', 'simulation_days',
                    'day_start', 'day_end', 'memory', 'priming', 'retrieval', 'routine', 'world'):
            assert cfg[key] == base[key], key
        conversation = deepcopy(cfg['conversation'])
        for key in ('base_talk_prob', 'max_per_agent_per_day'):
            conversation[key] = base['conversation'][key]
        assert conversation == base['conversation']
        assert cfg['analysis']['pipeline'] == base['analysis']['pipeline']
        assert cfg['analysis']['probes'] == base['analysis']['probes']
        assert cfg['llm']['backend'] == cfg['analysis']['observer']['backend'] == 'codex_cli'
        assert cfg['llm']['model'] == cfg['analysis']['observer']['model'] == 'gpt-5.6-luna'
        assert cfg['topology']['generated']['n_circles'] == 4
        assert cfg['topology']['generated']['n_bridges'] == 0
        assert cfg['topology']['generated']['p_between'] == (0.0 if cell.levels['ties'] == 'separated' else 0.4)
        assert cfg['conversation']['base_talk_prob'] == (0.1 if cell.levels['talk'] == 'low' else 0.7)
        assert cfg['conversation']['max_per_agent_per_day'] == (2 if cell.levels['talk'] == 'low' else 8)


def test_replacement_labels_inject_into_five_carriers_and_pass_world_audit(tmp_path):
    circles = None
    for cell in D.expand(E.resolve('communication_communities', tmp_path)):
        cfg = cell.config()
        profiles, _ = load_population(cfg['population'], cfg['population_size'])
        topo = TOPO.generate(profiles, cfg, seed_rng(cfg['world_seed'], 'topology'))
        assert len(topo['circles']) == 4
        if circles is not None:
            assert topo['circles'] == circles
        circles = topo['circles']
        TOPO.apply(profiles, topo)
        registry = INC.registry(cfg)
        texts = [text for _, text in INC.world_vocabulary(cfg, profiles)]
        assert INC.audit(registry, texts) == []
        applied = apply_planted(profiles, cfg)['memes']
        assert [m['phrase'] for m in applied] == PHRASES
        assert len({a for m in applied for a in m['seeds']}) == 20
        for meme in applied:
            assert len(meme['seeds']) == 5
            actual = {a for a, profile in profiles.items()
                      if any(meme['phrase'] in habit for habit in profile.habits)}
            assert actual == set(meme['seeds'])
            assert all('frequently ' in meme['habits'][a] for a in meme['seeds'])
            pattern = usage_pattern(meme['phrase'])
            assert pattern.search(meme['phrase'])
            assert pattern.search(meme['phrase'].upper())
            assert not pattern.search('unrelated campus conversation')
