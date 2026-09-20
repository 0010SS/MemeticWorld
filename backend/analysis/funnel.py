"""Meme-pipeline funnel and manipulation checks (OBSERVER ONLY; ontology v2 §4).

funnel: where does event -> talk -> recurrence -> linking -> naming -> verbatim memory -> reuse ->
spread break? (the old scripts/diagnose_funnel.py metrics, now on the provenance
`referent_event_ids` definition of "an utterance is about an event").

manipulation: one check per mechanism, counted from the trace types each mechanism writes, so a
condition can be shown to have actually switched its mechanism on. `memes` (meme_checks) is the same
check for the injected cohort, PER MEME: did its committed minority actually say it, how often, to how
many distinct hearers -- plus crowding, because four memes share one campus's conversational airtime.

validity: per mechanism "off" (config disables it), "active" (it fired at least once) or "inactive"
(enabled but never fired -- the condition did not manipulate what it claims to). Each declared meme is
its own mechanism (`meme:<id>`), so a cell whose minority never spoke reads "inactive" immediately.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import mean, median

from backend.analysis.candidates import STOP, tokens, zipf
from backend.analysis.provenance import EventText

NICK = re.compile(r"\bthe ([a-z\-']+ ){0,2}(thing|incident|saga|situation|mix-?up|disaster|fiasco|debacle|chaos|curse|"
                  r"story|effect|energy|move|streak|luck)\b", re.I)
LOGISTICS = re.compile(r"\b(heading|class|lecture|gym|library|lunch|dinner|coffee|later|tomorrow|shift|rehearsal|"
                       r"problem set|see you|catch you)\b", re.I)


def _pct(a, b):
    return round(100 * a / b, 1) if b else None


def _get(cfg: dict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _families(ev: dict) -> list[str]:
    return sorted({e["latent_type"] for e in ev.values() if e.get("latent_type")})


def funnel(rd) -> dict:
    ev, man, prov = rd.events, rd.manifest, rd.provenance
    tpd, tm = rd.ticks_per_day, man["tick_minutes"]
    names = rd.name_tokens
    utts = rd.utterances
    convs = list(rd.conversations.values())
    out = {"events": len(ev), "utterances": len(utts), "conversations": len(convs)}

    # --- talk about events (referent links)
    mentions = defaultdict(list)
    for u in utts:
        for e in prov[u["id"]]["referent_event_ids"]:
            mentions[e].append(u)
    linked = [u for u in utts if prov[u["id"]]["referent_event_ids"]]
    out["utt_about_events_pct"] = _pct(len(linked), len(utts))
    talked = [e for e in ev if mentions[e]]
    out["events_discussed_pct"] = _pct(len(talked), len(ev))
    if talked:
        out["mentions_per_discussed_event_med"] = median(len(mentions[e]) for e in talked)
        out["speakers_per_discussed_event_med"] = median(len({u["speaker"] for u in mentions[e]}) for e in talked)
        out["afterlife_hours_med"] = median((max(u["tick"] for u in mentions[e]) - ev[e]["start_tick"]) * tm / 60 for e in talked)
        out["discussed_on_later_day_pct"] = _pct(sum(1 for e in talked if any(u["tick"] // tpd > ev[e]["start_tick"] // tpd
                                                                               for u in mentions[e])), len(talked))
    refs = [prov[u["id"]]["referent_event_ids"] for u in linked]
    out["utts_linking_2plus_events"] = sum(1 for r in refs if len(r) >= 2)
    out["utts_linking_2plus_families"] = sum(1 for r in refs if len({ev[e].get("latent_type") for e in r}) >= 2)
    carried_only = [u for u in utts if prov[u["id"]]["carried_event_ids"] and not prov[u["id"]]["direct_event_ids"]]
    out["utts_carried_only_pct"] = _pct(len(carried_only), len(utts))

    # --- witnesses and recurrence (perception only)
    wit = defaultdict(set)
    beats_seen = defaultdict(lambda: defaultdict(set))
    for o in rd.of("observation"):
        if o.get("source_type") == "perception":
            for e in o.get("originating_event_ids") or []:
                wit[e].add(o["agent"])
                for f in o.get("facts", []):
                    beats_seen[e][o["agent"]].add(f["id"].split(".")[1] if "." in f["id"] else f["id"])
    if ev:
        out["witnesses_per_event_mean"] = round(mean(len(wit[e]) for e in ev), 2)
    pw = defaultdict(set)
    for e, ags in wit.items():
        for a in ags:
            pw[a].add(e)
    fams = _families(ev)
    pairs = [(a, f) for a in rd.agents for f in fams]
    if pairs:
        out["agent_family_pairs_2plus_witnessed_pct"] = _pct(
            sum(1 for a, f in pairs if sum(1 for e in pw[a] if e in ev and ev[e]["latent_type"] == f) >= 2), len(pairs))
    e3 = [e for e in ev.values() if e.get("latent_type") == "E3"]
    if e3:   # a coincidence needs both halves (P's and Q's beat) seen by the same person
        out["e3_both_halves_seen_pct"] = _pct(sum(any(len(b) >= 2 for b in beats_seen[e["id"]].values()) for e in e3), len(e3))

    # --- reflections that refer to 2+ different incidents
    et = EventText(rd)
    refl = rd.of("reflection")
    r_any = r_same = 0
    for r in refl:
        hit = et.link(r.get("text", ""), int(r["tick"]))
        if len(hit) >= 2:
            r_any += 1
            r_same += any(c >= 2 for c in Counter(ev[e]["latent_type"] for e in hit).values())
    out["reflections"] = len(refl)
    out["reflections_linking_2plus_events"] = r_any
    out["reflections_linking_2plus_same_family"] = r_same

    # --- reminding links: same hidden family vs chance
    links = reminding_links(rd)
    cross = [(a, b) for a, b, kind in links if kind == "cross_event"]
    out["remindings_asked"] = len(rd.of("reminding"))
    out["remindings_linked"] = sum(1 for r in rd.of("reminding") if r.get("reminded_of"))
    out["reminding_cross_event_links"] = len(cross)
    if cross:
        fc = Counter(e["latent_type"] for e in ev.values())
        tot = sum(fc.values())
        out["reminding_same_family_pct"] = _pct(sum(ev[a]["latent_type"] == ev[b]["latent_type"] for a, b in cross), len(cross))
        out["reminding_same_family_chance_pct"] = round(100 * sum((c / tot) ** 2 for c in fc.values()), 1)

    kinds = conversation_kinds(rd)
    out["catchup_conversations"] = kinds["catchup"]
    out["group_conversations"] = kinds["group"]

    # --- naming attempts
    nick = defaultdict(set)
    for u in utts:
        for m in NICK.finditer(u["text"]):
            nick[m.group(0).lower()].add(u["speaker"])
    out["nickname_phrases"] = len(nick)
    out["nickname_phrases_2plus_speakers"] = sum(1 for s in nick.values() if len(s) >= 2)

    out.update(_verbatim_funnel(rd, utts, names))

    pc = Counter(tuple(sorted(c["participants"])) for c in convs)
    out["distinct_pairs"] = len(pc)
    out["convs_per_pair_mean"] = round(mean(pc.values()), 2) if pc else 0
    out["utts_per_conv"] = round(len([u for u in utts if u.get("conversation_id")]) / max(1, len(convs)), 2)
    out["logistics_utts_pct"] = _pct(sum(1 for u in utts if LOGISTICS.search(u["text"])), len(utts))
    return out


def _verbatim_funnel(rd, utts, names) -> dict:
    """Distinctive bigrams heard -> kept verbatim in the listener's memory -> reused later by them."""
    def bigrams(text):
        t = tokens(text)
        return {f"{a} {b}" for a, b in zip(t, t[1:]) if a not in STOP and b not in STOP and a not in names
                and b not in names and min(zipf(a), zipf(b)) < 4.2}
    mem_of = defaultdict(str)
    for m in rd.of("memory_encoded"):
        for uid in m.get("utterance_ids") or []:
            mem_of[(m["agent"], uid)] += " " + " ".join(tokens(m["text"]))
    later = defaultdict(list)
    for u in utts:
        later[u["speaker"]].append((u["tick"], u.get("conversation_id") or u["id"], " " + " ".join(tokens(u["text"])) + " "))
    heard = ret = reused = reused_ret = reused_not = heard_not = 0
    for u in utts:
        bg = bigrams(u["text"])
        for L in u["listeners"]:
            if L == u["speaker"]:
                continue
            m = mem_of.get((L, u["id"]))
            fut = [t for tk, cid, t in later[L] if tk > u["tick"] and cid != (u.get("conversation_id") or u["id"])]
            for b in bg:
                heard += 1
                in_mem = m is not None and f" {b} " in m + " "
                hit = any(f" {b} " in t for t in fut)
                ret += in_mem
                reused += hit
                if in_mem:
                    reused_ret += hit
                else:
                    heard_not += 1
                    reused_not += hit
    echo_n = echo_hit = 0
    by_conv = defaultdict(list)
    for u in utts:
        if u.get("conversation_id"):
            by_conv[u["conversation_id"]].append(u)
    for us in by_conv.values():
        us.sort(key=lambda x: x.get("idx", 0))
        for a, b in zip(us, us[1:]):
            bg = bigrams(a["text"])
            if bg and b["speaker"] != a["speaker"]:
                echo_n += 1
                echo_hit += bool(bg & bigrams(b["text"]))
    return {"distinctive_bigrams_heard": heard,
            "bigrams_retained_verbatim_pct": _pct(ret, heard),
            "bigrams_reused_later_pct": round(100 * reused / heard, 2) if heard else None,
            "reuse_given_retained_pct": round(100 * reused_ret / ret, 2) if ret else None,
            "reuse_given_not_retained_pct": round(100 * reused_not / heard_not, 2) if heard_not else None,
            "next_turn_echo_pct": _pct(echo_hit, echo_n)}


# ------------------------------------------------------------------------------------------------
# manipulation checks

def reminding_links(rd) -> list[tuple]:
    """(new event, old event, kind) for each linked reminding; kind: same_event | cross_event | no_event."""
    nodes = rd.nodes
    out = []
    for r in rd.of("reminding"):
        if not r.get("reminded_of"):
            continue
        a = [e for e in r.get("originating_event_ids") or [] if e in rd.events]
        b = [e for e in nodes.get(r["reminded_of"], {}).get("event_ids", []) if e in rd.events]
        if not a or not b:
            out.append((None, None, "no_event"))
        elif set(a) & set(b):
            out.append((a[0], a[0], "same_event"))
        else:
            out.append((a[0], b[0], "cross_event"))
    return out


def conversation_kinds(rd) -> dict:
    c = Counter()
    for conv in rd.conversations.values():
        topic = (conv.get("trigger") or {}).get("topic")
        if topic == "catchup":
            c["catchup"] += 1
        elif topic == "group" or len(conv.get("participants", [])) >= 3:
            c["group"] += 1
    return {"catchup": c["catchup"], "group": c["group"], "total": len(rd.conversations)}


def _stuck(rd) -> list[dict]:
    """Verbatim-stuck wordings: v2 `wordings` on memory records, else v1 lens["verbatim"] phrases
    (self-produced if the agent itself said it in the encoded exchange)."""
    out = []
    for m in rd.of("memory_encoded"):
        ws = m.get("wordings")
        if ws:
            for w in ws:
                out.append({"agent": m["agent"], "phrase": w.get("phrase"),
                            "self": bool(w.get("self_produced")) or w.get("heard_from") == m["agent"]})
            continue
        for ph in (m.get("lens") or {}).get("verbatim") or []:
            own = " ".join(rd.utt_by_id[u]["text"].lower() for u in m.get("utterance_ids") or []
                           if u in rd.utt_by_id and rd.utt_by_id[u]["speaker"] == m["agent"])
            out.append({"agent": m["agent"], "phrase": ph, "self": ph.lower() in own})
    return out


def _module_fired(check) -> bool:
    """A module's manipulation_check(): {"active", "counters", ...}; any other shape counts numbers."""
    if isinstance(check, dict) and "active" in check:
        return bool(check["active"])
    if isinstance(check, dict) and isinstance(check.get("counters"), dict):
        return any(float(v) > 0 for v in check["counters"].values())
    return _numbers(check) > 0


def _numbers(d) -> float:
    if isinstance(d, (bool, int, float)):
        return float(d)
    if isinstance(d, dict):
        return sum(_numbers(v) for v in d.values())
    if isinstance(d, list):
        return sum(_numbers(v) for v in d)
    return 0.0


def planted_phrase(cfg: dict) -> dict | None:
    pp = _get(cfg, "controls.planted_phrase")
    if not pp:
        return None
    habit = pp.get("habit", "")
    q = re.findall(r"[\"“]([^\"”]+)[\"”]", habit)
    return {"agent": pp.get("agent"), "habit": habit, "phrase": (q[0] if q else habit).strip().lower()}


# ------------------------------------------------------------------------------------------------
# injected-cohort manipulation checks (the 2x2 registry)

def norm_phrase(text: str) -> str:
    """An injected phrase or an utterance reduced to a comparable form: tokens, lower case, hyphens read
    as spaces -- "blue-tray" and "blue tray" are one coinage said two ways (memo §2.1, formal variation),
    and a meme that is only ever written the other way has still been said."""
    return " ".join(" ".join(tokens(text)).replace("-", " ").split())


def says(text: str, norm: str) -> bool:
    return bool(norm) and f" {norm} " in f" {norm_phrase(text)} "


def meme_checks(rd) -> dict:
    """Did each injected meme actually get INJECTED, and is one minority crowding out the others?

    A meme its own seeds never utter has not been injected at all, and no amount of analysis three hours
    later will rescue the cell -- so this is a manipulation check, reported per meme and immediately:
    who said it, how often, in how many conversations, in front of how many distinct hearers, and how
    many of its seeds never said it once.

    With four memes sharing one campus's conversational airtime, crowding is the likeliest way the design
    fails: `crowding` gives each meme's share of all injected-phrase uses, the evenness of that split
    (normalised entropy: 1.0 = perfectly even, low = one minority dominating), and the airtime the whole
    cohort takes of the run's utterances.

    `contamination` is the R1/R2 check from the observer's side: the phrase, or one of its distinctive
    words, occurring in text the WORLD produced means any agent could have coined it independently and
    transmission is no longer distinguishable from rediscovery.
    """
    memes = list(getattr(rd, "memes", []) or [])
    if not memes:
        return {}
    utts = rd.utterances
    tpd = rd.ticks_per_day
    # the wording agents were actually given: event fact texts, viewpoint renderings and referent names
    # (emergence.world_segments), plus profile, routine and background text minus the injected habits
    # themselves (rd.world_text). This is the corpus R1 and R2 are about.
    from backend.analysis.emergence import world_segments
    given = [" ".join(seg) for seg in world_segments(rd)] + [rd.world_text]
    world = {t for seg in given for t in tokens(seg)}
    world_norm = [f" {n} " for n in (norm_phrase(seg) for seg in given) if n]
    rows, totals = {}, {}
    for m in memes:
        norm = norm_phrase(m["phrase"])
        seeds = set(m.get("seeds") or [])
        uses = [u for u in utts if says(u["text"], norm)]
        seed_uses = [u for u in uses if u["speaker"] in seeds]
        other = [u for u in uses if u["speaker"] not in seeds]
        hearers = {l for u in seed_uses for l in (u.get("listeners") or []) if l != u["speaker"]}
        said_by = Counter(u["speaker"] for u in seed_uses)
        content = [t for t in tokens(m["phrase"]) if t not in STOP]
        rows[m["id"]] = {
            "phrase": m["phrase"], "cell": f"{m['grounding']}x{m['breadth']}",
            "grounding": m["grounding"], "breadth": m["breadth"], "site": m.get("site"),
            "n_seeds": len(seeds), "seeds": sorted(seeds), "seeds_source": m.get("seeds_source"),
            "injected": bool(seed_uses),
            "seed_uses": len(seed_uses), "seed_speakers": len(said_by), "silent_seeds": len(seeds) - len(said_by),
            "uses_per_speaking_seed_med": round(median(said_by.values()), 1) if said_by else None,
            "seed_conversations": len({u.get("conversation_id") for u in seed_uses if u.get("conversation_id")}),
            "distinct_hearers": len(hearers), "distinct_non_seed_hearers": len(hearers - seeds),
            "uses": len(uses), "other_uses": len(other), "other_speakers": len({u["speaker"] for u in other}),
            "first_use_tick": uses[0]["tick"] if uses else None,
            "first_use_day": (uses[0]["tick"] // tpd + 1) if uses else None,
            "first_other_use_day": (other[0]["tick"] // tpd + 1) if other else None,
            "last_use_day": (uses[-1]["tick"] // tpd + 1) if uses else None,
            "uses_by_day": {d: sum(1 for u in uses if u["tick"] // tpd + 1 == d)
                            for d in sorted({u["tick"] // tpd + 1 for u in uses})},
            "contamination": {"phrase_in_world_text": any(f" {norm} " in seg for seg in world_norm),
                              "distinctive_words_in_world_text": sorted(t for t in content if t in world)},
        }
        totals[m["id"]] = len(uses)
    tot = sum(totals.values())
    ent = -sum((c / tot) * math.log(c / tot) for c in totals.values() if c) if tot else 0.0
    return {"memes": rows,
            "crowding": {"total_uses": tot, "utterances": len(utts),
                         "uses_per_100_utterances": round(100 * tot / len(utts), 2) if utts else None,
                         "share": {k: round(v / tot, 3) for k, v in totals.items()} if tot else
                                  {k: None for k in totals},
                         "max_share": round(max(totals.values()) / tot, 3) if tot else None,
                         "evenness": round(ent / math.log(len(totals)), 3) if tot and len(totals) > 1 else None},
            "injected": sorted(k for k, r in rows.items() if r["injected"]),
            "not_injected": sorted(k for k, r in rows.items() if not r["injected"]),
            "contaminated": sorted(k for k, r in rows.items()
                                   if r["contamination"]["phrase_in_world_text"]
                                   or r["contamination"]["distinctive_words_in_world_text"])}


def meme_table(checks: dict) -> str:
    """The cohort's manipulation check as plain text: one column per meme."""
    rows = checks.get("memes") or {}
    ids = list(rows)
    if not ids:
        return "(no memes)"
    w = max(18, *(len(i) + 2 for i in ids))
    out = [f"{'':26}" + "".join(f"{i:>{w}}" for i in ids)]
    for label, key in (("cell", "cell"), ("seeds", "n_seeds"), ("injected", "injected"),
                       ("seed uses", "seed_uses"), ("seeds who said it", "seed_speakers"),
                       ("silent seeds", "silent_seeds"), ("seed conversations", "seed_conversations"),
                       ("distinct hearers", "distinct_hearers"), ("non-seed uses", "other_uses"),
                       ("non-seed speakers", "other_speakers"), ("first non-seed day", "first_other_use_day"),
                       ("uses total", "uses")):
        out.append(f"{label:26}" + "".join(f"{str(rows[i][key]):>{w}}" for i in ids))
    c = checks.get("crowding") or {}
    out.append(f"{'share of cohort airtime':26}" + "".join(f"{str((c.get('share') or {}).get(i)):>{w}}" for i in ids))
    out.append("")
    out.append(f"cohort uses {c.get('total_uses')} in {c.get('utterances')} utterances "
               f"({c.get('uses_per_100_utterances')} per 100); evenness {c.get('evenness')}, "
               f"max share {c.get('max_share')}")
    if checks.get("not_injected"):
        out.append(f"NOT INJECTED (no seed ever said it): {', '.join(checks['not_injected'])}")
    if checks.get("contaminated"):
        out.append(f"CONTAMINATED (world text carries the phrase or its distinctive words): "
                   f"{', '.join(checks['contaminated'])}")
    return "\n".join(out)


def manipulation(rd) -> dict:
    ev = rd.events
    enc = rd.of("memory_encoded")
    vp = rd.of("viewpoint")
    lens = [m["lens"] for m in enc if m.get("lens")]
    rem = rd.of("reminding")
    rl = reminding_links(rd)
    stuck = _stuck(rd)
    ml = Counter(r.get("mechanism") for r in rd.of("memory_link"))
    prim = rd.of("priming")
    if any("cast_from_home" in e for e in ev.values()):
        cast, cast_src = [bool(e.get("cast_from_home")) for e in ev.values() if e.get("circle") is not None], "events"
    else:
        cast, cast_src = _v1_home(rd) or [], "v1_home_groups"
    refs = [r for e in ev.values() for r in e.get("referents") or []]
    inner = [f for e in ev.values() for b in e.get("beats", []) for f in b.get("facts", []) if f.get("kind") == "inner"]
    out = {
        "viewpoint": {"records": len(vp), "renderings": sum(len(r.get("facts", [])) for r in vp),
                      "dropped": sum(len(r.get("dropped") or []) for r in vp)},
        "lens": {"encodings": len(lens), "distortions": sum(1 for l in lens if l.get("distort")),
                 "association_drawn": sum(1 for l in lens if l.get("associate")),
                 "associations": sum(1 for l in lens if l.get("association"))},
        "verbatim": {"stuck": len(stuck), "stuck_self": sum(1 for s in stuck if s["self"]),
                     "stuck_other": sum(1 for s in stuck if not s["self"]),
                     "distinct_phrases": len({(s["phrase"] or "").lower() for s in stuck})},
        "reminding": {"asked": len(rem), "linked": sum(1 for r in rem if r.get("reminded_of")),
                      "same_event": sum(1 for *_, k in rl if k == "same_event"),
                      "cross_event": sum(1 for *_, k in rl if k == "cross_event"),
                      "no_event": sum(1 for *_, k in rl if k == "no_event")},
        "memory_links": dict(sorted(ml.items(), key=lambda kv: str(kv[0]))),
        "conversations": conversation_kinds(rd),
        "priming": {"lines": sum(1 for r in prim if r.get("phrases")), "phrases": sum(len(r.get("phrases") or []) for r in prim),
                    "agents": len({r.get("agent") for r in prim if r.get("phrases")})},
        "need": {"open_matters": len(rd.of("open_matter")), "focal": len(rd.of("open_matter_focal"))},
        "casting": {"events": len(ev), "assigned": len(cast), "cast_from_home": sum(cast),
                    "cast_from_home_rate": round(sum(cast) / len(cast), 3) if cast else None,
                    "source": cast_src},
        "referents": {"n": len(refs), "reused": sum(1 for r in refs if r.get("reused")),
                      "reuse_rate": round(sum(1 for r in refs if r.get("reused")) / len(refs), 3) if refs else None,
                      "distinct": len({r.get("id") for r in refs})},
        "link_visibility": {"inner_facts": len(inner), "inner_public": sum(1 for f in inner if f.get("visibility") == "all")},
        "retrieval": {"utterances_with_retrieval": sum(1 for u in rd.utterances if u.get("retrieved"))},
        "modules": rd.manifest.get("manipulation") or {},
    }
    vr = out["viewpoint"]
    vr["drop_rate"] = round(vr["dropped"] / (vr["renderings"] + vr["dropped"]), 3) if vr["renderings"] + vr["dropped"] else None
    pp = planted_phrase(rd.cfg)
    if pp:
        uses = [u for u in rd.utterances if pp["phrase"] and pp["phrase"] in u["text"].lower()]
        out["planted"] = {"agent": pp["agent"], "phrase": pp["phrase"],
                          "uses_by_agent": sum(1 for u in uses if u["speaker"] == pp["agent"]),
                          "uses_by_others": sum(1 for u in uses if u["speaker"] != pp["agent"]),
                          "other_speakers": len({u["speaker"] for u in uses} - {pp["agent"]})}
    mc = meme_checks(rd)
    if mc:
        out["memes"] = mc
    return out


def _v1_home(rd) -> list[bool] | None:
    """v1 runs (D46 clustering) record the family -> home group map in the manifest instead of a
    per-event cast_from_home: recompute it from the protagonist."""
    fh = rd.manifest.get("family_home_groups")
    if not fh:
        return None
    out = []
    for e in rd.events.values():
        g = fh.get(e.get("latent_type"))
        p = ((e.get("roles") or {}).get("P") or {}).get("agent")
        if g:
            out.append(p in rd.groups.get(g, []))
    return out


def mechanisms_enabled(cfg: dict) -> dict:
    """mechanism -> is it switched on in this run's config (v2 keys; v1 equivalents where they existed)."""
    le = cfg.get("latent_events") or {}
    mode = _get(cfg, "latent_events.assignment.mode")
    sw = _get(cfg, "retrieval.source_weights") or {}
    mods = cfg.get("modules") or {}
    return {
        "referents": bool(_get(cfg, "latent_events.referents.enabled", False)),
        "assignment": (mode not in (None, "none")) if mode is not None else float(le.get("clustering") or 0) > 0,
        "catchup": bool(_get(cfg, "conversation.catchup.enabled", False)),
        "group": bool(_get(cfg, "conversation.group.enabled", False)),
        "need": bool(_get(cfg, "need.enabled", False)),
        "reminding": bool(_get(cfg, "reminding.enabled", False)),
        "link_visibility": float(le.get("link_visibility") or 0) > 0,
        "verbatim": bool(_get(cfg, "memory.verbatim.enabled", False)),
        "priming": bool(_get(cfg, "priming.enabled", False)),
        "source_weights": any(float(v) != 1.0 for v in sw.values()),
        "viewpoints": bool(_get(cfg, "perception.viewpoints", False)),
        "lens": float(_get(cfg, "memory.encoding_variability", 0) or 0) > 0 and float(_get(cfg, "memory.encoding_noise", 0) or 0) > 0,
        "planted_phrase": bool(_get(cfg, "controls.planted_phrase")),
        # one entry per declared meme, so a cell whose minority never spoke is visible in `validity`
        # rather than inferred from a null result hours later
        **{f"meme:{m.get('id')}": True for m in (_get(cfg, "memes.registry") or [])
           if _get(cfg, "memes.enabled") and isinstance(m, dict) and m.get("id")},
        **{f"module:{m}": bool(v) for m, v in mods.items()},
    }


def validity(cfg: dict, manip: dict) -> dict:
    """mechanism -> off | active | inactive. Modules without recorded counters (v1 manifests) cannot
    be verified and are left out."""
    fired = {
        "referents": manip["referents"]["reused"] > 0,
        "assignment": manip["casting"]["cast_from_home"] > 0,
        "catchup": manip["conversations"]["catchup"] > 0,
        "group": manip["conversations"]["group"] > 0,
        "need": manip["need"]["focal"] > 0,
        "reminding": manip["reminding"]["linked"] > 0,
        "link_visibility": manip["link_visibility"]["inner_public"] > 0,
        "verbatim": manip["verbatim"]["stuck"] > 0,
        "priming": manip["priming"]["lines"] > 0,
        "source_weights": manip["retrieval"]["utterances_with_retrieval"] > 0,
        "viewpoints": manip["viewpoint"]["renderings"] > 0,
        "lens": manip["lens"]["encodings"] > 0,
        "planted_phrase": (manip.get("planted") or {}).get("uses_by_agent", 0) > 0,
    }
    memes = ((manip.get("memes") or {}).get("memes") or {})
    mods = manip.get("modules") or {}
    out = {}
    for mech, on in mechanisms_enabled(cfg).items():
        if not on:
            out[mech] = "off"
            continue
        if mech.startswith("meme:"):
            # "inactive" here means the committed minority never uttered its phrase: the cell was not
            # injected, and anything measured about it afterwards is measuring nothing
            mid = mech.split(":", 1)[1]
            if mid not in memes:
                continue
            f = memes[mid]["injected"]
        elif mech.startswith("module:"):
            name = mech.split(":", 1)[1]
            if name not in mods:
                continue
            f = _module_fired(mods[name])
        else:
            f = fired[mech]
        out[mech] = "active" if f else "inactive"
    return out


def diagnose(rd) -> dict:
    m = manipulation(rd)
    return {"metrics": funnel(rd), "manipulation": m, "validity": validity(rd.cfg, m)}


def table(rows: list[dict]) -> str:
    """Plain-text side-by-side table of funnel metrics for several runs (rows: {"run", "metrics"})."""
    keys = list(dict.fromkeys(k for r in rows for k in r["metrics"]))
    lines = [f"{'':46}" + " ".join(f"{r['run'][-12:]:>12}" for r in rows)]
    for k in keys:
        lines.append(f"{k:46}" + " ".join(f"{str(r['metrics'].get(k, '-')):>12}" for r in rows))
    return "\n".join(lines)

