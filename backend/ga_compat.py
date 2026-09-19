"""Bridge to the vendored Generative Agents (Park et al., 2023) code base.

We import the upstream modules *unmodified* from
third_party/generative_agents/reverie/backend_server and:

  * provide a stub `utils` module (upstream expects a git-ignored utils.py
    holding OpenAI keys and file paths),
  * provide a stub `openai` module when the package is not installed,
  * re-route the upstream OpenAI wrappers (ChatGPT_request, GPT4_request,
    GPT_request, get_embedding) to MemeWorld's LLM/embedding layer,
  * make prompt-template paths absolute (upstream resolves them relative to cwd),
  * silence the upstream debug prints.

Everything upstream that we use is exposed as attributes of this module:
`AssociativeMemory`, `ConceptNode`, `Scratch`, `retrieve` (module), `reflect`
(module), `converse` (module), `rgp` (run_gpt_prompt module), `gs` (gpt_structure).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GA_ROOT = REPO_ROOT / "third_party" / "generative_agents" / "reverie" / "backend_server"
GA_ASSETS = REPO_ROOT / "third_party" / "generative_agents" / "environment" / "frontend_server" / "static_dirs" / "assets"

_LLM = None        # backend.llm.client.LLMClient
_EMBED = None      # callable(str) -> list[float]
_loaded = False


def _noop(*_a, **_k):
    return None


def _install_stubs():
    if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "openai_api_key"):
        u = types.ModuleType("utils")
        u.openai_api_key = ""
        u.key_owner = "memeworld"
        u.maze_assets_loc = str(GA_ASSETS)
        u.env_matrix = ""
        u.env_visuals = ""
        u.fs_storage = ""
        u.fs_temp_storage = ""
        u.collision_block_id = "32125"
        u.debug = False
        sys.modules["utils"] = u
    try:
        import openai  # noqa: F401
    except ImportError:
        o = types.ModuleType("openai")
        o.api_key = ""
        sys.modules["openai"] = o


# --- routed replacements for upstream OpenAI wrappers -----------------------

COMPLETION_SYSTEM = ("You are a text-completion engine. Continue the user's text exactly "
                     "where it stops, following any format it establishes. Output only the "
                     "continuation, with no preamble or commentary.")
CHAT_SYSTEM = ("You are a helpful assistant. Follow the requested output format exactly and output only "
               "what is asked for, with no explanation or reasoning afterwards.")


def _chat_request(prompt, *_a, **_k):
    out = _LLM.complete(prompt, system=CHAT_SYSTEM, max_tokens=400, temperature=0.7)
    # upstream parsers call json.loads on the raw text; modern chat models like
    # to wrap JSON in markdown fences, which upstream (gpt-3.5 era) never saw.
    if "```" in out:
        inner = out.split("```")[1]
        if inner.lstrip().lower().startswith("json"):
            inner = inner.lstrip()[4:]
        out = inner.strip()
    return out


def _gpt_request(prompt, gpt_parameter):
    return _LLM.complete(prompt, system=COMPLETION_SYSTEM,
                         max_tokens=max(64, int(gpt_parameter.get("max_tokens", 150)) * 2),
                         temperature=float(gpt_parameter.get("temperature", 0.5)),
                         stop=gpt_parameter.get("stop"))


def _get_embedding(text, model=None):
    text = (text or "").replace("\n", " ") or "this is blank"
    return _EMBED(text)


_orig_generate_prompt = None


def _generate_prompt(curr_input, prompt_lib_file):
    p = Path(prompt_lib_file)
    if not p.is_absolute():
        p = GA_ROOT / p
    return _orig_generate_prompt(curr_input, str(p))


def load(llm_client=None, embed_fn=None):
    """Import upstream modules and bind them to our LLM/embedding layer."""
    global _loaded, _LLM, _EMBED, _orig_generate_prompt
    if llm_client is not None:
        _LLM = llm_client
    if embed_fn is not None:
        _EMBED = embed_fn
    if _loaded:
        return sys.modules[__name__]
    _install_stubs()
    if str(GA_ROOT) not in sys.path:
        sys.path.insert(0, str(GA_ROOT))

    import persona.prompt_template.gpt_structure as gs
    import persona.prompt_template.run_gpt_prompt as rgp
    import persona.memory_structures.associative_memory as am
    import persona.memory_structures.scratch as sc
    import persona.cognitive_modules.retrieve as rt
    import persona.cognitive_modules.reflect as rf
    import persona.cognitive_modules.converse as cv

    _orig_generate_prompt = gs.generate_prompt
    mods = [gs, rgp, am, sc, rt, rf, cv, sys.modules.get("persona.prompt_template.print_prompt")]
    for m in mods:
        if m is None:
            continue
        m.print = _noop  # silence upstream debug prints (module-level shadowing)
        for name, fn in (("ChatGPT_request", _chat_request), ("GPT4_request", _chat_request),
                         ("ChatGPT_single_request", _chat_request), ("GPT_request", _gpt_request),
                         ("get_embedding", _get_embedding), ("generate_prompt", _generate_prompt),
                         ("temp_sleep", _noop)):
            if hasattr(m, name):
                setattr(m, name, fn)
        if hasattr(m, "debug"):
            m.debug = False

    this = sys.modules[__name__]
    this.gs, this.rgp, this.retrieve, this.reflect, this.converse = gs, rgp, rt, rf, cv
    this.AssociativeMemory, this.ConceptNode, this.Scratch = am.AssociativeMemory, am.ConceptNode, sc.Scratch
    _loaded = True
    return this


def template_path(rel: str) -> str:
    """Absolute path of an upstream prompt template, e.g. 'v2/decide_to_talk_v2.txt'."""
    return str(GA_ROOT / "persona" / "prompt_template" / rel)
