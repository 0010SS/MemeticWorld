#!/usr/bin/env python3
"""Build a fictional, reproducible 500-person campus and separate naming seeds.

No network, LLM calls, or simulation runs. Public benchmarks and scenario
assumptions are documented in docs/HOMEWOOD_500_EXPERIMENT.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YEARS = ["first-year", "sophomore", "junior", "senior"]
# Counts PER YEAR; degree mix is a modeling choice, not enrollment microdata.
MAJORS = [
    ("Computer Science", "Engineering", 10, 4),
    ("Biomedical Engineering", "Engineering", 7, 3),
    ("Mechanical Engineering", "Engineering", 5, 2),
    ("Electrical and Computer Engineering", "Engineering", 3, 2),
    ("Chemical and Biomolecular Engineering", "Engineering", 3, 2),
    ("Environmental Engineering", "Engineering", 3, 1),
    ("Molecular and Cellular Biology", "Arts and Sciences", 14, 5),
    ("Neuroscience", "Arts and Sciences", 11, 3),
    ("Public Health Studies", "Arts and Sciences", 11, 4),
    ("Economics", "Arts and Sciences", 9, 3),
    ("International Studies", "Arts and Sciences", 6, 3),
    ("Psychology", "Arts and Sciences", 6, 3),
    ("Applied Mathematics and Statistics", "Engineering", 5, 2),
    ("Physics", "Arts and Sciences", 2, 2),
    ("Chemistry", "Arts and Sciences", 2, 2),
    ("History", "Arts and Sciences", 1, 1),
    ("English", "Arts and Sciences", 1, 1),
    ("Philosophy", "Arts and Sciences", 1, 1),
]
CLUBS = ["board-game club", "outdoor recreation group", "music ensemble", "arts group",
         "community volunteering group", "film discussion club", "recreational sports club",
         "student newspaper", "sustainability group", "debate club", "cooking group", "dance group"]
HOBBIES = ["reading fiction", "hiking", "sketching", "basketball", "cooking", "photography",
           "piano", "running", "gardening", "board games", "watching films", "cycling",
           "baking", "swimming", "podcasts", "knitting", "tennis", "birdwatching"]
FIRST = "Alex Avery Jordan Morgan Casey Taylor Quinn Riley Cameron Reese Skyler Robin Jamie Rowan Devon Sam Charlie Harper Emery Peyton Kai Sage Blair Remy Ellis Eden Noel Micah Adrian Sasha Dana Drew Blair Sidney Parker Jules Shiloh Bailey Finley Kendall Arden Alexis Cameron Dakota Elliot Frankie Hayden Jesse Lee Logan Marley Nico Phoenix River Rory Shawn Spencer Tatum Terry Toby Val Wren Zane Amara Anika Arjun Asha Ben Celia Diego Elena Ezra Farah Hana Hugo Iris Isla Javier Jia Jun Kiran Leila Lena Liam Lina Luca Luis Mae Mateo Mina Mira Nadia Naomi Nia Nina Omar Owen Priya Rafael Rina Ryan Salma Sara Theo Vera Vivian Yara Zoe".split()
LAST = "Adams Ahmed Allen Alvarez Anderson Bailey Baker Bennett Brooks Brown Campbell Carter Chen Choi Clark Collins Cooper Davis Diaz Edwards Evans Flores Foster Garcia Gomez Green Gupta Hall Harris Hassan Hayes Hernandez Hill Howard Hughes Jackson James Johnson Jones Khan Kim King Kumar Lee Lewis Li Lin Lopez Martin Martinez Miller Mitchell Moore Morgan Morris Murphy Nelson Nguyen Ortiz Park Patel Perez Perry Price Quinn Ramirez Reed Reyes Rivera Roberts Robinson Ross Ruiz Sanchez Sanders Scott Shah Singh Smith Stewart Sullivan Taylor Thomas Thompson Torres Tran Turner Walker Wang Ward Washington White Williams Wilson Wong Wright Wu Yang Young Zhang Zhao".split()
STAFF = [
    ("Hopkins dining service", "dining shift supervisor", "Dining Hall", "Main Floor", 1),
    ("Hopkins dining service", "cook", "Dining Hall", "Main Floor", 2),
    ("Hopkins dining service", "food service worker", "Dining Hall", "Main Floor", 2),
    ("Hopkins dining service", "cashier", "Dining Hall", "Main Floor", 1),
    ("Hopkins dining service", "dishroom worker", "Dining Hall", "Main Floor", 1),
    ("Hopkins dining service", "dining utility worker", "Dining Hall", "Main Floor", 1),
    ("Levering dining service", "cook", "Cafe", "Counter", 1),
    ("Levering dining service", "cashier", "Cafe", "Counter", 1),
    ("Levering dining service", "food service worker", "Cafe", "Counter", 1),
    ("Levering dining service", "dining utility worker", "Cafe", "Counter", 1),
    ("Library services", "reference librarian", "Library", "Study Tables", 4),
    ("Library services", "circulation assistant", "Library", "Quiet Floor", 4),
    ("Library services", "digital collections specialist", "Library", "Quiet Floor", 2),
    ("Facilities", "custodian", "Dorm", "Lounge", 6),
    ("Facilities", "maintenance technician", "Classroom", "Lecture Hall", 4),
    ("Student services", "academic adviser", "Classroom", "Seminar Room", 4),
    ("Student services", "residential life coordinator", "Dorm", "Lounge", 3),
    ("Student services", "student organization coordinator", "Quad", "Lawn", 3),
    ("Student services", "registration services assistant", "Classroom", "Seminar Room", 2),
    ("Laboratory support", "laboratory technician", "Research Lab", "Wet Lab", 3),
    ("Laboratory support", "laboratory technician", "Research Lab", "Dry Lab", 2),
    ("Laboratory support", "makerspace technician", "Research Lab", "Makerspace", 1),
    ("Campus safety", "campus safety liaison", "Quad", "Lawn", 4),
    ("IT services", "IT support specialist", "Library", "Study Tables", 2),
]
SOC = {"reserved": "reserved", "moderate": "moderately sociable", "outgoing": "outgoing"}
PLAN = {"structured": "organized", "flexible": "flexible about daily plans", "spontaneous": "spontaneous"}
STYLE = {"brief": "Usually speaks briefly and directly.",
         "balanced": "Uses ordinary conversational detail and takes turns speaking.",
         "detailed": "Often explains context and gives examples."}


def balanced(rng, n, labels):
    """20/60/20 in each student year; largest-remainder approximation elsewhere."""
    counts = [round(n * .2), n - 2 * round(n * .2), round(n * .2)]
    values = [label for label, count in zip(labels, counts) for _ in range(count)]
    rng.shuffle(values)
    return values


def entry(time, loc, arena, activity):
    return {"time": time, "location": loc, "arena": arena, "activity": activity}


def time_at(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def academic_place(major):
    if major in {"Molecular and Cellular Biology", "Neuroscience", "Chemistry", "Biomedical Engineering"}:
        return "Research Lab", "Wet Lab"
    if "Engineering" in major:
        return "Research Lab", "Makerspace"
    if major in {"Computer Science", "Physics", "Applied Mathematics and Statistics"}:
        return "Research Lab", "Dry Lab"
    return "Library", "Study Tables"


def build(seed=20260919):
    rng = random.Random(seed)
    people = []
    major_pool = [(m, division) for m, division, n, _ in MAJORS for _ in range(n)]
    # 36% engineering overall, including AMS; one primary field per student.
    for y, year in enumerate(YEARS):
        majors = list(major_pool)
        rng.shuffle(majors)
        ages = [18+y] * 45 + [19+y] * 50 + [20+y] * 5
        rng.shuffle(ages)
        club_flags = [True]*83 + [False]*17
        resident_count = [93,90,40,25][y]
        residence_flags = ["on campus"]*resident_count + ["off campus"]*(100-resident_count)
        dining_count,cafe_count = [(85,10),(75,15),(50,30),(45,30)][y]
        lunch_places = ["Dining Hall"]*dining_count + ["Cafe"]*cafe_count + ["Quad"]*(100-dining_count-cafe_count)
        rng.shuffle(club_flags)
        rng.shuffle(residence_flags)
        rng.shuffle(lunch_places)
        for i, ((major, division), age) in enumerate(zip(majors, ages)):
            people.append({"category": "student", "year": year, "age": age, "role": "undergraduate student",
                           "major": major, "division": division, "department": major, "local_index": i,
                           "club_member": club_flags[i], "residence": residence_flags[i], "lunch_location":lunch_places[i]})
    faculty_roles = ["assistant professor"]*12 + ["associate professor"]*14 + ["professor"]*14 + ["lecturer"]*4
    rng.shuffle(faculty_roles)
    age_ranges = {"assistant professor":(30,44),"associate professor":(36,62),"professor":(43,70),"lecturer":(28,64)}
    for major, division, _, nfaculty in MAJORS:
        for i in range(nfaculty):
            faculty_role = faculty_roles.pop()
            people.append({"category": "faculty", "year": "faculty", "age": rng.randint(*age_ranges[faculty_role]),
                           "role": faculty_role,
                           "department": major, "major": major, "division": division,
                           "local_index": i, "residence": "off campus"})
    for dept, role, loc, arena, count in STAFF:
        for i in range(count):
            people.append({"category": "staff", "year": "staff", "age": rng.randint(24,64), "role": role,
                           "department": dept, "major": None, "local_index": i,
                           "workplace": [loc, arena], "residence": "off campus"})
    assert len(people) == 500
    # Assign traits within each cohort, independently of identity, role, and major.
    strata = defaultdict(list)
    for p in people:
        strata[p["year"]].append(p)
    for members in strata.values():
        soc = balanced(rng, len(members), list(SOC))
        plan = balanced(rng, len(members), list(PLAN))
        style = balanced(rng, len(members), list(STYLE))
        for p,s,b,c in zip(members,soc,plan,style):
            p.update(sociability=s, planning=b, communication=c)
    names = [f"{a} {b}" for a in dict.fromkeys(FIRST) for b in LAST]
    rng.shuffle(names)
    rng.shuffle(people)  # neutral IDs, not lexical year/occupation ordering
    profiles, audit, memories, both_memories = [], {}, {}, {}
    available_sprites = sorted(p.stem for p in (ROOT / "third_party/generative_agents/environment/frontend_server/static_dirs/assets/characters").glob("*.png"))
    if not available_sprites:
        available_sprites = ["Abigail_Chen"]
    for index,p in enumerate(people):
        aid, name = f"p{index+1:04d}", names[index]
        p.update(id=aid, name=name)
        hobbies = rng.sample(HOBBIES, 2)
        loc, arena = academic_place(p["major"]) if p["category"] != "staff" else p["workplace"]
        slot = rng.randrange(6)
        meal_min = 11*60 + 30*slot
        lunch_loc = p["lunch_location"] if p["category"] == "student" else ["Dining Hall", "Cafe", "Quad"][index % 3]
        lunch_arena = {"Dining Hall": "Main Floor", "Cafe": "Counter", "Quad": "Lawn"}[lunch_loc]
        clubs = [CLUBS[(p["local_index"] + YEARS.index(p["year"]) * 3) % len(CLUBS)]] if p.get("club_member") else []
        if p["category"] == "student":
            bg = (f"A {p['year']} undergraduate with coursework in {p['major']}. "
                  f"Lives {p['residence']}; enjoys {hobbies[0]} and {hobbies[1]}. "
                  "Balances classes, individual work, and time with friends.")
            home_loc,home_arena = ("Dorm", ["Room 214","Room 310","Room 118","Room 105","Room 402"][index%5]) if p["residence"] == "on campus" else ("Library","Study Tables")
            classroom = "Lecture Hall" if index % 2 else "Seminar Room"
            # Varied morning/afternoon class blocks, identical distribution by naming cohort in expectation.
            morning = ("Classroom", classroom, f"attending a class in {p['major']}") if index%2 else (loc,arena,f"working on {p['major']} coursework")
            afternoon = (loc,arena,f"working on {p['major']} coursework") if index%2 else ("Classroom",classroom,f"attending a class in {p['major']}")
            break_time = 10*60 + 15*(index%4)
            break_loc,break_arena = [("Quad","Lawn"),("Cafe","Counter"),("Dorm","Lounge"),("Gym","Main Floor")][(index//4)%4]
            social_loc,social_arena = [("Quad","Lawn"),("Dorm","Lounge"),("Classroom","Seminar Room"),("Gym","Main Floor")][(index//3)%4]
            routine = [entry("08:30",home_loc,home_arena,"preparing for the campus day"),
                       entry("09:00",*morning), entry(time_at(break_time),break_loc,break_arena,"taking a short break between commitments"),
                       entry(time_at(break_time+15),loc,arena,"reading and preparing work"),
                       entry(time_at(meal_min),lunch_loc,lunch_arena,"having lunch and making plans for the afternoon"),
                       entry(time_at(meal_min+30),loc,arena,"returning to coursework"),
                       entry("14:00",*afternoon), entry("15:30",loc,arena,"reviewing notes and meeting a study partner"),
                       entry("16:15",social_loc,social_arena,f"meeting people from the {clubs[0]}" if clubs else f"taking time for {hobbies[0]}")]
        elif p["category"] == "faculty":
            home_loc,home_arena = loc,arena
            bg = (f"Works in {p['department']}, teaching undergraduates and contributing to research. "
                  f"Lives off campus and commutes to work. Outside work enjoys {hobbies[0]} and {hobbies[1]}. "
                  "Campus rooms stand in for workspaces during the modeled daytime window.")
            routine = [entry("08:30",loc,arena,"preparing teaching materials and responding to work messages"),
                       entry("09:30","Classroom","Lecture Hall" if index%2 else "Seminar Room",f"teaching {p['department']}"),
                       entry("11:00",loc,arena,"doing scholarly work"),
                       entry(time_at(meal_min),lunch_loc,lunch_arena,"taking a lunch break"),
                       entry(time_at(meal_min+30),loc,arena,"returning to scholarly work"),
                       entry("14:00","Library","Study Tables","holding student consultations"),
                       entry("15:00",loc,arena,"meeting colleagues and preparing the next class")]
        else:
            home_loc,home_arena = loc,arena
            bg = (f"Works as a {p['role']} in {p['department']}. Lives off campus and commutes to work. "
                  f"Outside work enjoys {hobbies[0]} and {hobbies[1]}. "
                  "The routine describes a daytime work block, including a break; workspaces are simplified.")
            # Dining staff take their meal break after peak service, not during the student lunch wave.
            if "dining service" in p["department"]:
                meal_min = 14*60 + 15*(index%3)
                lunch_loc,lunch_arena = "Quad","Lawn"
            routine = [entry("08:30",loc,arena,f"preparing for daytime duties as a {p['role']}"),
                       entry("09:00",loc,arena,f"working as a {p['role']} and assisting people as needed"),
                       entry(time_at(meal_min),lunch_loc,lunch_arena,"taking a meal break"),
                       entry(time_at(meal_min+30),loc,arena,f"returning to duties as a {p['role']}"),
                       entry("16:30",loc,arena,"finishing tasks and preparing a work handover")]
        # Last entry wins if the same minute occurs twice; no invalid/zero-length ordering.
        routine = sorted({r["time"]:r for r in routine}.values(),key=lambda r:r["time"])
        demographics = {k:p[k] for k in ("category","age","year","role","department","major","residence")}
        if "division" in p:
            demographics["division"] = p["division"]
        profiles.append({"id":aid,"name":name,"sprite":available_sprites[index%len(available_sprites)],
                         "demographics":demographics,"background":bg,
                         "personality":{"traits":[SOC[p["sociability"]],PLAN[p["planning"]],f"interested in {hobbies[0]}"],
                                        "communication_style":STYLE[p["communication"]]},
                         "interests":{"topics":[p["department"]],"hobbies":hobbies,"clubs":clubs},
                         "habits":["checks the day's commitments before starting","takes a meal break during the day"],
                         "home":{"location":home_loc,"arena":home_arena},"routine":routine})
        label = "Hopkins Cafe" if p["year"] == "first-year" else "FFC"
        seed_memory = f"In my recent everyday conversations, I have called the dining hall beside AMR III {label} when arranging meals. I have eaten or worked at that physical dining hall."
        memories[aid] = [seed_memory]
        both_memories[aid] = [seed_memory + " I have also heard both FFC and Hopkins Cafe used for that same dining hall."]
        audit[aid] = {"category":p["category"],"year":p["year"],"initial_preferred_name":label,
                      "initially_presented_names":[label],"referent":"Dining Hall",
                      "sociability":p["sociability"],"planning":p["planning"],"communication":p["communication"],
                      "residence":p["residence"],"lunch_start":time_at(meal_min),"lunch_location":lunch_loc,"name":name}
    groups, relations = {}, {}
    def tie(a,b,kind="friend",fam=.65,aff=.65):
        if a == b:
            return
        key=tuple(sorted((a,b)))
        rec={"a":key[0],"b":key[1],"type":kind,"familiarity":fam,"affinity":aff}
        if key not in relations or relations[key]["familiarity"] < fam:
            relations[key]=rec
    def ring(members, kind="friend",fam=.65):
        if len(members)>1:
            for i,a in enumerate(members):
                tie(a,members[(i+1)%len(members)],kind,fam)
    for year in YEARS:
        ids=[p["id"] for p in people if p["year"]==year]
        rng.shuffle(ids)
        for i in range(0,len(ids),10):
            groups[f"{year}_peer_group_{i//10+1:02d}"]=ids[i:i+10]
            ring(ids[i:i+10])
    for club in CLUBS:
        ids=[p["id"] for p in profiles if club in p["interests"]["clubs"]]
        rng.shuffle(ids)
        groups[club]=ids
        ring(ids,"clubmate",.45)
    for major,_,_,_ in MAJORS:
        students=[p["id"] for p in people if p["category"]=="student" and p["major"]==major]
        faculty=[p["id"] for p in people if p["category"]=="faculty" and p["major"]==major]
        groups[f"academic_{major}"]=students+faculty
        for i,student in enumerate(students):
            # Acquaintance wording is role-neutral and understood by the current relationship renderer.
            tie(student,faculty[i%len(faculty)],"acquaintance",.35,.6)
    first=[p["id"] for p in people if p["year"]=="first-year"]
    older=[p["id"] for p in people if p["year"] in YEARS[1:]]
    rng.shuffle(older)
    for a,b in zip(first,older):
        tie(a,b,"acquaintance",.4,.6)
    for department in {p["department"] for p in people if p["category"]=="staff"}:
        ids=[p["id"] for p in people if p["category"]=="staff" and p["department"]==department]
        groups[f"work_{department}"]=ids
        ring(ids,"acquaintance",.7)
    dining=[p["id"] for p in people if p["category"]=="staff" and p["workplace"][0]=="Dining Hall"]
    students=[p["id"] for p in people if p["category"]=="student"]
    rng.shuffle(students)
    for i,student in enumerate(students[:120]):
        tie(student,dining[i%len(dining)],"acquaintance",.3,.6)
    population={"agents":profiles,"relationships":[relations[k] for k in sorted(relations)],
                "groups":dict(sorted(groups.items())),"circles":{}}
    return population,memories,both_memories,audit


def validate(population, memories, audit):
    from backend.simulation.world import ARENAS
    agents=population["agents"]
    assert len(agents)==500
    assert len({a["id"] for a in agents})==len({a["name"] for a in agents})==500
    assert Counter(a["demographics"]["category"] for a in agents)=={"student":400,"faculty":44,"staff":56}
    assert set(memories)==set(audit)=={a["id"] for a in agents}
    assert Counter(a["initial_preferred_name"] for a in audit.values())=={"Hopkins Cafe":100,"FFC":400}
    for year in YEARS:
        rows=[a for a in audit.values() if a["year"]==year]
        assert len(rows)==100
        assert Counter(a["sociability"] for a in rows)=={"reserved":20,"moderate":60,"outgoing":20}
        assert Counter(a["planning"] for a in rows)=={"structured":20,"flexible":60,"spontaneous":20}
        assert Counter(a["communication"] for a in rows)=={"brief":20,"balanced":60,"detailed":20}
    for a in agents:
        assert "FFC" not in json.dumps(a) and "Hopkins Cafe" not in json.dumps(a)
        assert a["home"]["arena"] in ARENAS[a["home"]["location"]]
        times=[]
        for r in a["routine"]:
            assert r["arena"] in ARENAS[r["location"]], (a["id"],r)
            h,m=map(int,r["time"].split(":"))
            assert 0<=h<24 and 0<=m<60
            times.append(h*60+m)
        assert times==sorted(set(times))
        term=audit[a["id"]]["initial_preferred_name"]
        other="FFC" if term=="Hopkins Cafe" else "Hopkins Cafe"
        assert term in memories[a["id"]][0] and other not in memories[a["id"]][0]
    seen=set()
    for r in population["relationships"]:
        assert r["a"] in audit and r["b"] in audit and r["a"]!=r["b"]
        key=tuple(sorted((r["a"],r["b"])))
        assert key not in seen
        assert 0<=r["familiarity"]<=1 and 0<=r["affinity"]<=1
        seen.add(key)
    for members in population["groups"].values():
        assert len(set(members))==len(members) and set(members)<=set(audit)
    return {"agents":500,"roles":dict(Counter(a["demographics"]["category"] for a in agents)),
            "years":dict(Counter(a["year"] for a in audit.values())),
            "initial_names":dict(Counter(a["initial_preferred_name"] for a in audit.values())),
            "student_faculty_ratio":400/44,"relationships":len(seen),"groups":len(population["groups"]),
            "student_divisions":dict(Counter(a["demographics"]["division"] for a in agents if a["demographics"]["category"]=="student")),
            "club_members":sum(bool(a["interests"]["clubs"]) for a in agents if a["demographics"]["category"]=="student")}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",type=int,default=20260919)
    ap.add_argument("--out-dir",type=Path,default=ROOT/"configs/population")
    ap.add_argument("--roster",type=Path,default=ROOT/"docs/HOMEWOOD_500_ROSTER.md")
    args=ap.parse_args()
    pop,mem,both,audit=build(args.seed)
    summary=validate(pop,mem,audit)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,data in [("homewood500.yaml",pop),("homewood500_initial_memories.yaml",mem),
                      ("homewood500_both_names_memories.yaml",both)]:
        (args.out_dir/name).write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True,width=110),encoding="utf-8")
    digest=hashlib.sha256((args.out_dir/"homewood500.yaml").read_bytes()).hexdigest()
    payload={"schema_version":1,"seed":args.seed,"synthetic":True,"summary":summary,
             "profile_sha256":digest,"agents":audit,
             "source_document":"docs/HOMEWOOD_500_EXPERIMENT.md",
             "observer_aliases":{"FFC":["FFC","Fresh Food Cafe","Fresh Food Café"],
                                 "Hopkins Cafe":["Hopkins Cafe","Hopkins Café"]},
             "notes":["Preferred names are assigned initial conditions, not guaranteed generated speech.",
                      "Initially presented names do not establish all knowledge already held by the language model.",
                      "Personality, staff proportions, ages, and detailed major counts are synthetic assumptions.",
                      "home is a daytime campus starting anchor; demographics.residence records modeled housing."]}
    (args.out_dir/"homewood500_audit.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Homewood 500 — fictional profile roster","",
           "Generated with seed `"+str(args.seed)+"`. All names and biographies are synthetic. This table is observer-only; agents do not receive other people's initial naming assignments.","",
           "Full profiles and routines: [homewood500.yaml](../configs/population/homewood500.yaml). Method and source boundaries: [experiment guide](HOMEWOOD_500_EXPERIMENT.md).","",
           "| ID | Name | Role / year | Academic or work area | Personality / speaking style | Initial dining name |",
           "|---|---|---|---|---|---|"]
    for a in pop["agents"]:
        d=a["demographics"]; row=audit[a["id"]]
        role=d["year"]+" student" if d["category"]=="student" else d["role"]
        lines.append(f"| {a['id']} | {a['name']} | {role} | {d['department']} | {row['sociability']}; {row['planning']}; {row['communication']} | {row['initial_preferred_name']} |")
    args.roster.parent.mkdir(parents=True,exist_ok=True)
    args.roster.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
