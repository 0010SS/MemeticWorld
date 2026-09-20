"""Codex transport isolation, credit authentication, and full-study configuration."""
import json
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.llm import codex_cli as C
from backend.llm.client import LLMClient, LLMUnavailable, NonRetryableProviderFailure, make_backend
from backend.research import experiments as E


@pytest.fixture
def cli(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(C, 'executable', lambda: 'codex.exe')
    monkeypatch.setattr(C.tempfile, 'tempdir', str(tmp_path))

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if cmd[1:3] == ['login', 'status']:
            return subprocess.CompletedProcess(cmd, 0, '', 'Logged in using ChatGPT')
        path = Path(cmd[cmd.index('--output-last-message') + 1])
        path.write_text('A café reply.', encoding='utf-8')
        assert Path(kw['cwd'], 'instructions.txt').read_text(encoding='utf-8').startswith('System role')
        events = [
            {'type': 'item.completed', 'item': {'type': 'error', 'message': C._STARTUP_ADVISORIES[1]}},
            {'type': 'turn.started'}, {'type': 'turn.completed'}]
        return subprocess.CompletedProcess(cmd, 0, '\n'.join(map(json.dumps, events)), '')

    monkeypatch.setattr(C.subprocess, 'run', run)
    return calls


def test_credit_auth_and_stateless_unicode_transport(cli, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-inherit')
    monkeypatch.setenv('CODEX_API_KEY', 'must-not-inherit-either')
    backend = make_backend({'backend': 'codex_cli'})
    assert backend.model == C.DEFAULT_MODEL
    for _ in range(2):
        assert backend.generate('long café prompt ' * 5000, 'System role', 400, 0.7) == 'A café reply.'
    assert len(cli) == 3  # one local login check, then two isolated completions
    first, second = cli[1:]
    assert first[1]['cwd'] != second[1]['cwd']
    for cmd, kw in cli:
        assert 'OPENAI_API_KEY' not in kw['env'] and 'CODEX_API_KEY' not in kw['env']
    cmd, kw = first
    assert len(kw['input']) > 32768 and kw['input'] not in cmd
    assert '--ephemeral' in cmd and '--ignore-user-config' in cmd and '--skip-git-repo-check' in cmd
    assert 'forced_login_method="chatgpt"' in cmd and 'project_doc_max_bytes=0' in cmd
    assert 'shell_tool' in cmd and 'apps' in cmd and 'memories' in cmd
    assert not Path(kw['cwd']).exists()


@pytest.mark.parametrize('status', ['Not logged in', 'Logged in using an API key'])
def test_rejects_missing_or_api_login(monkeypatch, status):
    monkeypatch.setattr(C, 'executable', lambda: 'codex.exe')
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, '', status))
    with pytest.raises(NonRetryableProviderFailure, match='ChatGPT sign-in'):
        C.check_login()


@pytest.mark.parametrize('output', [
    [{'type': 'turn.failed', 'error': {'message': 'usage limit reached secret-token'}}],
    [{'type': 'item.completed', 'item': {'type': 'command_execution'}}, {'type': 'turn.completed'}],
])
def test_failed_or_tool_turn_never_enters_recording(monkeypatch, tmp_path, output):
    monkeypatch.setattr(C, 'check_login', lambda: None)
    monkeypatch.setattr(C, 'executable', lambda: 'codex.exe')
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(
        a[0], 0, '\n'.join(map(json.dumps, output)), ''))
    client = LLMClient(C.CodexCLIBackend(), tmp_path / 'calls.jsonl', fail_fast={}, sleep=lambda _: None)
    try:
        with pytest.raises(LLMUnavailable) as caught:
            client.complete('prompt')
        assert 'secret-token' not in str(caught.value)
        assert client.stats['errors'] == 1 and client.stats['calls'] == 0
    finally:
        client.close()
    assert not (tmp_path / 'calls.jsonl').read_text()


def test_timeout_is_actionable_without_long_retry_cycle(monkeypatch, tmp_path):
    monkeypatch.setattr(C, 'check_login', lambda: None)
    monkeypatch.setattr(C, 'executable', lambda: 'codex.exe')
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 180)
    monkeypatch.setattr(C.subprocess, 'run', timeout)
    with pytest.raises(NonRetryableProviderFailure, match='timed out'):
        C.CodexCLIBackend().generate('prompt', None, 20, 0)


def test_final_answer_replays_without_cli(cli, tmp_path, monkeypatch):
    path = tmp_path / 'calls.jsonl'
    client = LLMClient(C.CodexCLIBackend(), path, fail_fast={})
    assert client.complete('prompt', system='System role') == 'A café reply.'
    client.close()
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: pytest.fail('Replay must not invoke Codex'))
    replay = LLMClient(C.CodexCLIBackend(), tmp_path / 'replay.jsonl', mode='replay', replay_path=path)
    assert replay.complete('prompt', system='System role') == 'A café reply.'
    replay.close()


def test_all_four_studies_keep_conditions_and_use_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'check_login', lambda: None)
    for study, count in [('emergence', 1), ('communities', 4), ('influence', 6), ('social_meaning', 4)]:
        original = E.resolve(study, tmp_path)
        design = E.configure(original, backend='codex_cli', days=2)
        assert E.configure(design, backend='codex_cli').path == design.path  # subprocess idempotence
        cells = E.D.expand(design, 'codex_cli')
        assert len(cells) == count
        assert E.check(design, 'codex_cli')['ready']
        assert {c.run_dir for c in cells}.isdisjoint(c.run_dir for c in E.D.expand(original))
        for cell in cells:
            cfg = cell.config()
            assert cfg['population_size'] == 100 and cfg['simulation_days'] == 2
            assert cfg['llm']['backend'] == cfg['analysis']['observer']['backend'] == 'codex_cli'
            assert cfg['llm']['model'] == cfg['analysis']['observer']['model'] == C.DEFAULT_MODEL
            assert cfg['priming']['enabled'] and cfg['memory']['verbatim']['enabled']
            assert cfg['initial_memories_file'] is None and cfg['shared_background']['markdown'] is None
            assert cfg['retrieval']['source_weights'] == {'seed': 0.5, 'ambient': 0.5}
    explicit = E.configure(E.resolve('emergence', tmp_path), backend='codex_cli', agent_model='gpt-5.5', observer_model='gpt-5.5')
    assert E.configure(explicit, backend='codex_cli').path == explicit.path
    assert E.D.expand(explicit)[0].config()['llm']['model'] == 'gpt-5.5'


def test_preview_and_observer_switch_select_codex_model(tmp_path, monkeypatch):
    from backend.api import server
    from backend.research.observer import settings
    from backend.config import load_config
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    result = TestClient(server.app).post('/api/research/experiment-preview', json={
        'action': 'experiment', 'experiment': 'emergence', 'backend': 'codex_cli', 'days': 2})
    assert result.status_code == 200
    assert result.json()['agent_model'] == C.DEFAULT_MODEL
    assert result.json()['observer'] == {'backend': 'codex_cli', 'model': C.DEFAULT_MODEL}
    assert result.json()['days'] == [2]
    cfg = load_config('configs/campus100.yaml')
    assert settings(cfg, backend='codex_cli')['model'] == C.DEFAULT_MODEL
    assert settings(cfg, backend='codex_cli', model='gpt-5.5')['model'] == 'gpt-5.5'
