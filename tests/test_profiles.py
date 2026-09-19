"""Profile additions reach agent identity without leaking unrelated or hidden groups."""
from copy import deepcopy

import pytest
import yaml

from backend.agents.profile import load_population
from backend.agents.topology import apply


def _population():
    agents = []
    for aid, name in [("ada", "Ada Lane"), ("ben", "Ben Park"), ("cara", "Cara Green"), ("dan", "Dan Reed")]:
        agents.append({
            "id": aid, "name": name,
            "demographics": {"age": 20, "year": "junior", "major": "History", "role": "undergraduate"},
            "background": "Enjoys campus life.",
            "personality": {"traits": ["patient"], "communication_style": "Asks questions"},
            "interests": {"topics": [], "hobbies": [], "clubs": []},
            "habits": ["reads after lunch"], "home": {"location": "Dorm", "arena": "Room 214"},
            "routine": [{"time": "12:00", "location": "Dining Hall", "activity": "eating lunch"}],
        })
    agents[0]["personal"] = {
        "goal": "Finish a short story", "values": ["candor", "kindness"],
        "strengths": ["patient editing"], "blind_spots": ["overcommits to plans"],
        "stress_response": "Becomes quiet", "coping_strategy": "Takes a walk",
        "social_energy": "Prefers small groups", "trust_style": "Values follow-through",
        "conflict_style": "Asks to speak privately", "humor_style": "Dry observations",
        "pet_peeves": ["people interrupting"], "small_joys": ["warm bread"],
    }
    for agent, groups in zip(agents, [["lunch"], ["lunch"], ["lunch", "walking"], ["walking"]]):
        agent["friend_groups"] = groups
    return {
        "agents": agents,
        "friend_groups": {
            "lunch": {"name": "Lunch friends", "members": ["ada", "ben", "cara"]},
            "walking": {"name": "Walking friends", "members": ["cara", "dan"]},
        },
        "groups": {"seminar": ["ada", "dan"]},
        "circles": {"researcher_only": ["ada", "ben"]},
    }


def _load(tmp_path, data, n=None):
    path = tmp_path / "population.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_population(path, n)


def test_personal_identity_and_own_friends_reach_prompt_and_public_profile(tmp_path):
    data = _population()
    profiles, groups = _load(tmp_path, data)
    ada = profiles["ada"]
    learned = ada.ga_learned()
    for value in data["agents"][0]["personal"].values():
        for text in value if isinstance(value, list) else [value]:
            assert text in learned
    assert "Ada's Lunch friends: Ben Park, Cara Green." in learned
    assert "Walking friends" not in learned and "Dan Reed" not in learned
    assert "researcher_only" not in learned
    assert ada.friend_groups == [{"id": "lunch", "name": "Lunch friends", "members": ["Ben Park", "Cara Green"]}]
    public = ada.to_public_dict()
    assert public["personal"] == data["agents"][0]["personal"]
    assert public["friend_groups"] == ada.friend_groups
    assert "circles" not in public
    assert groups == {"seminar": ["ada", "dan"], "lunch": ["ada", "ben", "cara"], "walking": ["cara", "dan"]}


def test_old_population_files_remain_valid(tmp_path):
    data = _population()
    del data["friend_groups"]
    for agent in data["agents"]:
        agent.pop("personal", None)
        del agent["friend_groups"]
    profiles, groups = _load(tmp_path, data)
    assert groups == {"seminar": ["ada", "dan"]}
    for profile in profiles.values():
        assert profile.personal == {} and profile.friend_groups == []
        assert "Personal goal:" not in profile.ga_learned()


def test_truncated_population_filters_friends_and_drops_singletons(tmp_path):
    profiles, groups = _load(tmp_path, _population(), n=2)
    assert groups == {"lunch": ["ada", "ben"]}
    assert profiles["ada"].friend_groups[0]["members"] == ["Ben Park"]
    assert "Cara Green" not in profiles["ada"].ga_learned()
    profiles, groups = _load(tmp_path, _population(), n=1)
    assert profiles["ada"].friend_groups == [] and groups == {}


@pytest.mark.parametrize("case,match", [
    ("unknown_member", "unknown members"),
    ("duplicate_member", "duplicate members"),
    ("unknown_group", "unknown friend groups"),
    ("missing_declaration", "do not match"),
    ("extra_declaration", "do not match"),
    ("duplicate_declaration", "duplicate friend group IDs"),
    ("conflicting_group", "conflicts with"),
])
def test_malformed_friendships_fail_with_clear_error(tmp_path, case, match):
    data = _population()
    if case == "unknown_member":
        data["friend_groups"]["lunch"]["members"].append("missing")
    elif case == "duplicate_member":
        data["friend_groups"]["lunch"]["members"].append("ada")
    elif case == "unknown_group":
        data["agents"][-1]["friend_groups"].append("missing")
    elif case == "missing_declaration":
        data["agents"][-1]["friend_groups"] = []
    elif case == "extra_declaration":
        data["agents"][0]["friend_groups"].append("walking")
    elif case == "duplicate_declaration":
        data["agents"][0]["friend_groups"].append("lunch")
    elif case == "conflicting_group":
        data["groups"]["lunch"] = ["ada", "dan"]
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, data)


def test_unloaded_agents_still_validate_friendship_declarations(tmp_path):
    data = _population()
    data["agents"][-1]["friend_groups"] = []
    with pytest.raises(ValueError, match="do not match"):
        _load(tmp_path, data, n=2)


@pytest.mark.parametrize("personal", [{"goal": 3}, {"values": "kindness"}, {"values": [3]}, {"latent_type": "hidden"}])
def test_personal_properties_reject_invalid_or_unknown_values(tmp_path, personal):
    data = _population()
    data["agents"][0]["personal"] = personal
    with pytest.raises(ValueError, match="personal"):
        _load(tmp_path, data)


def test_generated_topology_removes_stale_friend_groups_without_exposing_circles(tmp_path):
    profiles, _ = _load(tmp_path, _population())
    original_personal = deepcopy(profiles["ada"].personal)
    apply(profiles, {"relationships": {}, "circles": {"c1": list(profiles)}})
    assert profiles["ada"].personal == original_personal
    for profile in profiles.values():
        assert profile.friend_groups == []
        assert "Lunch friends" not in profile.ga_learned()
        assert "c1" not in profile.ga_learned()
