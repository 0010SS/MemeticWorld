"""Focused lexical-accounting tests; no LLM calls or simulation runs."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_dining_names import analyze, lexical_counts, main


ALIAS_CASES = [
    ("Lunch at FFC?", {"ffc": 1, "hopkins_cafe": 0, "label": "ffc_only"}),
    ("ffc's doors are open. FFC!", {"ffc": 2, "hopkins_cafe": 0, "label": "ffc_only"}),
    ("AFFC FFC2 _FFC Hopkins Cafeine", {"ffc": 0, "hopkins_cafe": 0, "label": "no_target_alias"}),
    ("HOPKINS CAFÉ / Hopkins Cafe\u0301", {"ffc": 0, "hopkins_cafe": 2, "label": "hopkins_only"}),
    ('They say "FFC"; I heard "Hopkins Cafe".', {"ffc": 1, "hopkins_cafe": 1, "label": "both"}),
    ("Fresh Food Café", {"ffc": 1, "hopkins_cafe": 0, "label": "ffc_only"}),
    ("Let's meet at the dining hall.", {"ffc": 0, "hopkins_cafe": 0, "label": "no_target_alias"}),
]


def write_run(tmp_path, agents, rows):
    (tmp_path / "manifest.json").write_text(json.dumps({"agents": agents}), encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def profile(category, year):
    return {"demographics": {"category": category, "year": year}}


def utterance(speaker, text):
    return {"type": "utterance", "speaker": speaker, "text": text}


def check_cohort_denominators_and_unique_speakers(tmp_path):
    run = write_run(tmp_path, {
        "new": profile("student", "first-year"),
        "quiet": profile("student", "first-year"),
        "old": profile("student", "senior"),
        "worker": profile("staff", "not_applicable"),
    }, [
        utterance("new", "Hopkins Cafe"),
        utterance("new", 'They call Hopkins Café "FFC".'),
        utterance("new", "Lunch at the dining hall?"),
        utterance("old", "FFC, FFC."),
        utterance("worker", "Hello."),
        {"type": "exposure", "speaker": "old", "text": "FFC"},
        {"type": "conversation", "transcript": [["new", "Hopkins Cafe"]]},
    ])
    result = analyze(run)
    total = result["overall"]
    assert total["total_utterances"] == 5
    assert total["alias_bearing_utterance_denominator"] == 3
    assert total["lexical_mentions"] == {"ffc": 3, "hopkins_cafe": 2}
    assert total["unique_speakers"] == {
        "any_utterance": 3, "any_target_alias": 2, "ffc": 2,
        "hopkins_cafe": 1, "both_aliases_across_run": 1,
    }
    assert total["population_agents_with_no_logged_utterances"] == 1
    assert total["population_agents_with_no_logged_target_alias"] == 2
    first = result["by_year"]["first-year"]
    assert first["population_agents"] == 2
    assert first["alias_bearing_utterance_denominator"] == 2
    assert first["fractions_of_alias_bearing_utterances"] == {
        "ffc_only": 0.0, "hopkins_only": 0.5, "both": 0.5,
    }
    assert result["by_category"]["student"]["alias_bearing_utterance_denominator"] == 3
    assert result["by_category"]["staff"]["fractions_of_alias_bearing_utterances"]["ffc_only"] is None
    assert len(result["by_category_and_year"]) == 3
    assert total["observed_naming_opportunities"] is None


def check_no_mentions_is_not_no_references_and_empty_cohorts_remain(tmp_path):
    run = write_run(tmp_path, {
        "a": profile("student", "first-year"), "b": profile("faculty", "not_applicable"),
    }, [utterance("a", "The dining hall has lunch.")])
    result = analyze(run)
    assert result["overall"]["utterances_by_alias"]["no_target_alias"] == 1
    assert result["overall"]["alias_bearing_utterance_denominator"] == 0
    assert all(v is None for v in result["overall"]["fractions_of_alias_bearing_utterances"].values())
    assert result["by_category"]["faculty"]["total_utterances"] == 0
    assert result["by_category"]["faculty"]["population_agents"] == 1
    assert "does not mean no reference" in result["definitions"]["no_target_alias"]


def check_missing_speaker_metadata_is_explicit(tmp_path):
    result = analyze(write_run(tmp_path, {"a": {}}, [utterance("missing", "FFC")]))
    assert result["speakers_missing_from_manifest"] == ["missing"]
    assert result["by_category"]["unknown"]["unique_speakers"]["ffc"] == 1


def check_cli_output(tmp_path):
    write_run(tmp_path, {"a": profile("student", "junior")}, [utterance("a", "FFC")])
    out = tmp_path / "result.json"
    assert main([str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["overall"]["lexical_mentions"]["ffc"] == 1


class DiningNameTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name)

    def test_aliases(self):
        for text, expected in ALIAS_CASES:
            with self.subTest(text=text):
                self.assertEqual(lexical_counts(text), expected)

    def test_cohort_denominators_and_unique_speakers(self):
        check_cohort_denominators_and_unique_speakers(self.path)

    def test_no_mentions_is_not_no_references_and_empty_cohorts_remain(self):
        check_no_mentions_is_not_no_references_and_empty_cohorts_remain(self.path)

    def test_missing_speaker_metadata_is_explicit(self):
        check_missing_speaker_metadata_is_explicit(self.path)

    def test_cli_output_and_input_protection(self):
        from contextlib import redirect_stderr
        from io import StringIO

        check_cli_output(self.path)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            main([str(self.path), "--out", str(self.path / "manifest.json")])
        self.assertEqual(caught.exception.code, 2)

    def test_malformed_utterance_fails_clearly(self):
        run = write_run(self.path, {}, [{"type": "utterance", "speaker": "a"}])
        with self.assertRaisesRegex(ValueError, "line 1: utterance needs speaker and text"):
            analyze(run)


if __name__ == "__main__":
    unittest.main()
