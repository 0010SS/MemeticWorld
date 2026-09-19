"""Synthetic logs with known ground truth for the meme detector."""
from app.analysis import meme_detector, propagation
from app.db import database as db
from app.simulation.world import SimConfig, run_config_snapshot


class Log:
    def __init__(self, conn, condition="full"):
        self.conn = conn
        self.run_id = db.create_run(conn, name="t", condition=condition, seed=1, days=1, total_ticks=56,
                                    model="mock", config=run_config_snapshot(SimConfig(days=1, condition=condition)))
        db.update_run(conn, self.run_id, current_tick=56, status="finished")

    def say(self, tick, speaker, listener, text, conv, overheard=(), memory_ids=()):
        event_id = db.insert_event(self.conn, self.run_id, tick, f"t{tick}", "utterance", agent_id=speaker,
                                   location="Quad", text=text,
                                   data={"conversation_id": conv, "audience": [listener],
                                         "overheard_by": list(overheard), "memory_ids": list(memory_ids)})
        return event_id

    def remember(self, tick, agent, text, source_type, source_event_id):
        cur = self.conn.execute(
            "INSERT INTO memories (run_id, agent_id, tick, sim_time, text, importance, source_type, source_event_id)"
            " VALUES (?, ?, ?, ?, ?, 5, ?, ?)", (self.run_id, agent, tick, f"t{tick}", text, source_type, source_event_id))
        return cur.lastrowid

    def memes(self, control=None):
        corpus = meme_detector.load_corpus(self.conn, self.run_id)
        control_corpus = meme_detector.load_corpus(self.conn, control.run_id) if control else None
        return corpus, {m["phrase"]: m for m in meme_detector.detect_memes(corpus, control_corpus)}


def test_transmission_is_detected_and_confirmed(conn):
    log = Log(conn)
    e1 = log.say(1, "maya", "ethan", "That was total zorblax.", "c1")
    heard = log.remember(1, "ethan", 'Maya said to me at the Quad: "That was total zorblax."', "heard", e1)
    log.say(2, "ethan", "maya", "Yeah, zorblax for sure.", "c1")                        # echo, same conversation
    e3 = log.say(5, "ethan", "leo", "Pure zorblax out there.", "c2", memory_ids=[heard])  # adoption via memory
    log.say(9, "leo", "sam", "It's zorblax again.", "c3")                               # adoption via exposure only
    corpus, memes = log.memes()
    meme = memes["zorblax"]
    assert meme["originator"] == "maya"
    assert set(meme["adopters"]) == {"ethan", "leo"}
    assert meme["n_confirmed"] == 1 and meme["tier"] == "strong"
    edges = {e["target"]: e for e in meme["edges"]}
    assert edges["ethan"]["source"] == "maya" and edges["ethan"]["evidence"] == "memory"
    assert edges["leo"]["source"] == "ethan" and edges["leo"]["evidence"] == "exposure"
    assert meme["depth"] == 2
    roles = {n["id"]: n["role"] for n in propagation.cascade(corpus, meme)["nodes"]}
    assert roles["maya"] == "originator" and roles["leo"] == "adopter" and roles["sam"] == "exposed"
    assert e3


def test_ordinary_language_is_not_a_meme(conn):
    log = Log(conn)
    for i, speaker in enumerate(["maya", "ethan", "leo", "sam"]):  # everyone says it unprompted
        log.say(i, speaker, "priya", "I want banana bread.", f"c{i}")
    log.say(10, "priya", "maya", "Banana bread sounds amazing.", "c10")
    _, memes = log.memes()
    assert "banana bread" not in memes


def test_echo_alone_is_not_adoption(conn):
    log = Log(conn)
    log.say(1, "maya", "ethan", "Pure florp.", "c1")
    log.say(1, "ethan", "maya", "Ha, florp.", "c1")
    log.say(3, "maya", "leo", "Florp again.", "c2")
    _, memes = log.memes()
    assert "florp" not in memes  # ethan only echoed; no adopter in a new conversation


def test_world_vocabulary_is_ignored(conn):
    log = Log(conn)
    log.say(1, "maya", "ethan", "The squirrel at the fountain!", "c1")
    log.say(3, "ethan", "leo", "That squirrel at the fountain though.", "c2")
    log.say(5, "leo", "sam", "Squirrel fountain.", "c3")
    _, memes = log.memes()
    assert not any("squirrel" in phrase and "fountain" in phrase for phrase in memes)


def test_control_run_demotes_baseline_phrases(conn):
    full, control = Log(conn), Log(conn, condition="no_speech_memory")
    for log in (full, control):
        log.say(1, "maya", "ethan", "Such wobble energy.", "c1")
        log.say(4, "ethan", "leo", "Wobble energy everywhere.", "c2")
        log.say(8, "leo", "sam", "Pure wobble energy.", "c3")
    _, memes = full.memes(control=control)
    assert memes["wobble energy"]["tier"] == "baseline"
    assert memes["wobble energy"]["control"]["n_adopters"] == 2
