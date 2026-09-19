"""Read local API settings without placing credentials in resolved configs or run artifacts."""
from functools import lru_cache
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / '.env'


@lru_cache(maxsize=8)
def _read(path, modified, size):
    values = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:
            value = value.split(' #', 1)[0].rstrip()
        values[key.strip()] = value
    return values


def setting(name, default=None):
    if name in os.environ:
        return os.environ[name]
    if ENV_FILE.exists():
        stat = ENV_FILE.stat()
        return _read(str(ENV_FILE), stat.st_mtime_ns, stat.st_size).get(name, default)
    return default


def resolve_model(model, *, observer=False):
    defaults = {'OPENAI_MODEL': 'gpt-5.4-mini', 'OPENAI_OBSERVER_MODEL': 'gpt-5.4'}
    name = 'OPENAI_OBSERVER_MODEL' if observer else 'OPENAI_MODEL'
    if isinstance(model, str) and model.startswith('$'):
        name = model[1:]
        if name not in defaults:
            raise ValueError('Only OPENAI_MODEL and OPENAI_OBSERVER_MODEL can select models')
    elif model:
        return model
    return setting(name) or defaults[name]
