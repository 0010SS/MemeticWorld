"""GPT requests, credential isolation, and offline recording/replay."""
import json

import httpx
import pytest

from backend.config import load_config
from backend.llm import environment as env
from backend.llm.client import LLMClient, LLMUnavailable, OpenAIBackend, ProviderFailure


def test_local_env_models_and_process_override(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('# Local settings\nOPENAI_API_KEY="test-secret"\nOPENAI_MODEL=gpt-5.4-mini\n'
                    'OPENAI_OBSERVER_MODEL=gpt-5.4\n', encoding='utf-8-sig')
    monkeypatch.setattr(env, 'ENV_FILE', path)
    for name in ('OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_OBSERVER_MODEL'):
        monkeypatch.delenv(name, raising=False)
    assert env.setting('OPENAI_API_KEY') == 'test-secret'
    cfg = load_config('configs/memetics.yaml')
    assert cfg['llm'] == {**cfg['llm'], 'backend': 'openai', 'model': 'gpt-5.4-mini'}
    assert cfg['analysis']['observer']['model'] == 'gpt-5.4'
    assert 'test-secret' not in json.dumps(cfg)
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-5.4')
    assert load_config('configs/memetics.yaml')['llm']['model'] == 'gpt-5.4'


def test_gpt_request_and_replay_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return httpx.Response(200, json={'status': 'completed', 'output': [
            {'type': 'reasoning', 'summary': []},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello '},
                                           {'type': 'output_text', 'text': 'campus.'}]}]})

    monkeypatch.setattr(httpx, 'post', post)
    original = tmp_path / 'record.jsonl'
    client = LLMClient(OpenAIBackend('gpt-5.4-mini'), original, fail_fast={})
    assert client.complete('A campus story', system='World context', max_tokens=8) == 'Hello campus.'
    client.close()
    url, request = sent[0]
    assert url == 'https://api.openai.com/v1/responses'
    assert request['headers']['Authorization'] == 'Bearer test-secret'
    assert request['json']['instructions'] == 'World context'
    assert request['json']['input'] == 'A campus story'
    assert request['json']['store'] is False
    assert request['json']['max_output_tokens'] == 16
    assert request['json']['reasoning'] == {'effort': 'none'}
    assert 'test-secret' not in original.read_text()
    monkeypatch.setenv('OPENAI_API_KEY', '')
    replay = LLMClient(OpenAIBackend('gpt-5.4-mini'), tmp_path / 'replay.jsonl', mode='replay', replay_path=original)
    assert replay.complete('A campus story', system='World context', max_tokens=8) == 'Hello campus.'
    replay.close()
    assert len(sent) == 1


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_auth_and_configuration_errors_stop_immediately(tmp_path, monkeypatch, status):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: httpx.Response(status, text='test-secret'))
    sleeps = []
    client = LLMClient(OpenAIBackend('gpt-5.4-mini'), tmp_path / 'calls.jsonl', fail_fast={}, sleep=sleeps.append)
    with pytest.raises(LLMUnavailable, match=f'HTTP {status}'):
        client.complete('Hello')
    client.close()
    assert not sleeps
    assert not (tmp_path / 'calls.jsonl').read_text()
    assert 'test-secret' not in (tmp_path / 'llm_errors.jsonl').read_text()


@pytest.mark.parametrize('body', [
    {'status': 'incomplete', 'output': []}, {'status': 'completed', 'output': []},
    {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]}]},
])
def test_incomplete_or_empty_answers_are_provider_failures(monkeypatch, body):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: httpx.Response(200, json=body))
    with pytest.raises(ProviderFailure):
        OpenAIBackend('gpt-5.4-mini').generate('hello', None, 100, .7)


def test_missing_key_fails_without_a_request(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: pytest.fail('must not make an unauthenticated request'))
    client = LLMClient(OpenAIBackend(), tmp_path / 'calls.jsonl', fail_fast={})
    with pytest.raises(LLMUnavailable, match='OPENAI_API_KEY'):
        client.complete('Hello')
    client.close()


def test_cheapest_gpt_uses_minimal_reasoning_and_reserves_output_room(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    requests = []
    def post(url, **kwargs):
        requests.append(kwargs['json'])
        return httpx.Response(200, json={'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': '7'}]}]})
    monkeypatch.setattr(httpx, 'post', post)
    assert OpenAIBackend('gpt-5-nano').generate('Rate this memory.', None, 8, .7) == '7'
    assert requests[0]['reasoning'] == {'effort': 'minimal'}
    assert requests[0]['text'] == {'verbosity': 'low'}
    assert requests[0]['max_output_tokens'] == 2064
    assert 'temperature' not in requests[0]


def test_exhausted_output_does_not_retry_identical_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: httpx.Response(200, json={
        'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'output': []}))
    sleeps = []
    client = LLMClient(OpenAIBackend('gpt-5-nano'), tmp_path / 'calls.jsonl', fail_fast={}, sleep=sleeps.append)
    with pytest.raises(LLMUnavailable, match='exhausted the output budget'):
        client.complete('Hello')
    client.close()
    assert not sleeps
