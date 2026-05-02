"""Property-based I1 validation: 200 random but valid inputs × 2 runs each."""
import random

from sevim.pipeline import run_pipeline

_SUBJECTS = [
    "dog", "cat", "mouse", "lion", "tiger", "bird", "fish",
    "tree", "flower", "rock", "cloud", "river", "ocean",
    "gravity", "friction", "energy", "force", "mass",
    "atom", "molecule", "cell", "neuron", "muscle", "bone",
    "teacher", "student", "book", "page", "word", "idea",
]

_TEMPLATES = [
    "{s} causes {o}.",
    "{s} is part of {o}.",
    "{s} has a {o}.",
    "{s} is similar to {o}.",
    "{s} opposes {o}.",
    "{s} contains {o}.",
    "{s} is an example of {o}.",
    "{s} leads to {o}.",
    "{s} then {o}.",
]


def _gen_input(rng: random.Random) -> str:
    n_clauses = rng.randint(1, 5)
    clauses = []
    for _ in range(n_clauses):
        tmpl = rng.choice(_TEMPLATES)
        s = rng.choice(_SUBJECTS)
        o = rng.choice(_SUBJECTS)
        if s == o:
            continue
        clauses.append(tmpl.format(s=s, o=o))
    return " ".join(clauses) if clauses else "A causes B."


def test_i1_holds_over_200_random_inputs():
    rng = random.Random(0xDEADBEEF)
    violations = []
    for i in range(200):
        text = _gen_input(rng)
        a = run_pipeline(text)
        b = run_pipeline(text)
        if a.svg != b.svg:
            violations.append((i, text))
    assert not violations, f"I1 violated on {len(violations)} inputs: {violations[:3]}"


def test_trace_log_always_present():
    rng = random.Random(12345)
    for _ in range(50):
        text = _gen_input(rng)
        r = run_pipeline(text)
        stages = [t.stage for t in r.trace]
        # Five primary stages must always be present in order; S4.5 overlap
        # check is inserted between S4 and S5 and is optional for back-compat.
        must_have = ["S1", "S2", "S3", "S4", "S5"]
        filtered = [s for s in stages if s in must_have]
        assert filtered == must_have, f"stages: {stages}"
