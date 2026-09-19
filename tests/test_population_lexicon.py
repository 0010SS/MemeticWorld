"""Supplied character descriptions must not be mistaken for novel wording."""
from types import SimpleNamespace

from backend.simulation.lexicon import population_lexicon


def test_personal_and_friendship_prose_is_part_of_supplied_vocabulary():
    profile = SimpleNamespace(
        background="A student", habits=[], routine=[], demographics={}, interests={},
        personality={"traits": ["meticulous"], "communication_style": "Gentle understatement"},
        personal={"small_joys": ["violet marzipan"], "trust_style": "Reliable candor"},
        friend_groups=[{"id": "unspokenidentifier", "name": "Sunset companions", "members": ["Zelda Finch"]}],
        circles={"hiddenpartition": ["secretmember"]},
    )
    lexicon = population_lexicon({"a": profile})
    assert {"meticulous", "marzipan", "candor", "zelda", "finch"} <= set(lexicon["tokens"])
    assert {"gentle understatement", "violet marzipan", "sunset companions", "zelda finch"} <= set(lexicon["bigrams"])
    assert not {"unspokenidentifier", "hiddenpartition", "secretmember", "trust_style"} & set(lexicon["tokens"])


def test_older_profiles_without_new_properties_still_build_a_lexicon():
    profile = SimpleNamespace(background="Morning porridge", habits=[], routine=[], demographics={}, interests={})
    assert "morning porridge" in population_lexicon({"a": profile})["bigrams"]
