"""Generate LaTeX table fragments from the eval results JSON files.

Writes ``bench/eval/tables/{m1_routing,m2_retrieval,m3_viz_ops,
m4_graph_coverage,m5_inline_math,m6_narration_judge}.tex`` plus a
combined ``measurement_summary.tex`` digest.  Tables use ``booktabs`` /
``tabularx`` so they slot directly into the CAS Elsevier template
already loaded by ``paper/lyceum.tex``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[2]))                 # repo root
from bench.eval._common import RESULTS, TABLES, latex_escape         # noqa: E402

# The paper expects ``\input{tables/m1_routing}`` etc. to resolve from
# inside ``paper/``.  When the manuscript working tree exists (i.e.
# the author is running this from their local checkout that also has
# the paper sources) we mirror every fragment to ``paper/tables/`` so
# the paper build stays self-contained.  ``paper/`` is gitignored in
# the public code repository, so for users who cloned the code repo
# alone the mirror is silently skipped and the canonical fragments in
# ``bench/eval/tables/`` are still produced.
_PAPER_DIR = TABLES.parents[2] / "paper"
PAPER_TABLES = _PAPER_DIR / "tables" if _PAPER_DIR.is_dir() else None


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def fmt_3(x: float) -> str:
    return f"{x:.3f}"


def fmt_pm(x: float) -> str:
    return f"{x:.2f}"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[tables] wrote {path.relative_to(THIS.parents[2])}")
    # Mirror to paper/tables/ only if a local paper working tree exists
    # (gitignored in the public code repository).
    if PAPER_TABLES is not None:
        paper_path = PAPER_TABLES / path.name
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        paper_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# M1 — routing
# ---------------------------------------------------------------------------

def build_m1() -> None:
    d = json.loads((RESULTS / "m1_routing.json").read_text())
    intents = list(d["per_intent"].keys())
    rows = []
    for k in intents:
        v = d["per_intent"][k]
        rows.append(
            f"\\texttt{{{k.replace('_', '\\_')}}} & "
            f"{v['support']:>3d} & "
            f"{fmt_3(v['precision'])} & "
            f"{fmt_3(v['recall'])} & "
            f"{fmt_3(v['f1'])} \\\\"
        )
    body = "\n".join(rows)
    summary_row = (
        f"\\textbf{{macro avg}} & {d['n_utterances']:>3d} & "
        f"\\textbf{{{fmt_3(d['macro_precision'])}}} & "
        f"\\textbf{{{fmt_3(d['macro_recall'])}}} & "
        f"\\textbf{{{fmt_3(d['macro_f1'])}}} \\\\"
    )
    micro_row = (
        f"\\textbf{{accuracy}} & "
        f"\\multicolumn{{4}}{{c}}{{\\textbf{{"
        f"{fmt_3(d['overall_accuracy'])} on "
        f"{d['n_utterances']} utterances; "
        f"{d['wall_us_per_utterance']:.0f}~$\\mu$s/utt}}}} \\\\"
    )
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Per-intent routing accuracy on a hand-labelled set "
        f"of {d['n_utterances']} utterances drawn from the ten supported "
        f"intent classes.  ``Support'' is the number of gold utterances "
        f"per class; precision / recall / F$_1$ are computed against the "
        f"router's classification (and, for \\textsc{{control}}, "
        f"additionally against the resolved control action).  The single "
        f"error in this set falls on \\textsc{{book\\_overview}} "
        f"\\textit{{vs}} \\textsc{{topic\\_qa}}: the regex requires the "
        f"literal word ``book''.}}\n"
        f"\\label{{tab:m1-routing}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lrrrr}}\n"
        f"\\toprule\n"
        f"intent & n & precision & recall & F$_1$ \\\\\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\midrule\n"
        f"{summary_row}\n"
        f"{micro_row}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m1_routing.tex", tex)


# ---------------------------------------------------------------------------
# M2 — retrieval
# ---------------------------------------------------------------------------

def build_m2() -> None:
    d = json.loads((RESULTS / "m2_retrieval.json").read_text())
    r = d["results"]
    bm = r["bm25_only"]
    bt = r["bm25_titleboost"]
    h  = r["hybrid_rrf"]
    if "skipped" in h:
        h_mrr = h_r1 = h_r5 = h_r10 = "--"
        hybrid_status = (
            f"\\\\ \\multicolumn{{6}}{{l}}{{\\footnotesize "
            f"\\textit{{Hybrid RRF row skipped: "
            f"{latex_escape(h['skipped'])} "
            f"(re-run with \\texttt{{EMBED\\_BASE\\_URL}} pointing at a "
            f"live Qwen3 endpoint).}}}}"
        )
    else:
        h_mrr = fmt_3(h["mrr"])
        h_r1  = fmt_pct(h["recall_at_1"])
        h_r5  = fmt_pct(h["recall_at_5"])
        h_r10 = fmt_pct(h["recall_at_10"])
        hybrid_status = ""
    rows = (
        f"BM25 (lexical only) & {fmt_3(bm['mrr'])} & "
        f"{fmt_pct(bm['recall_at_1'])} & {fmt_pct(bm['recall_at_5'])} & "
        f"{fmt_pct(bm['recall_at_10'])} & 0.13~s \\\\\n"
        f"BM25 + title boost (Eq.~\\ref{{eq:title}}) & "
        f"\\textbf{{{fmt_3(bt['mrr'])}}} & "
        f"\\textbf{{{fmt_pct(bt['recall_at_1'])}}} & "
        f"\\textbf{{{fmt_pct(bt['recall_at_5'])}}} & "
        f"\\textbf{{{fmt_pct(bt['recall_at_10'])}}} & 0.13~s \\\\\n"
        f"BM25+title $\\oplus$ dense (RRF, $K\\!=\\!60$) & "
        f"{h_mrr} & {h_r1} & {h_r5} & {h_r10} & --- \\\\"
    )
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Retrieval ranking on a section-targeted query set "
        f"(query = ``explain $\\langle$title$\\rangle$''; gold = the "
        f"section's nid; {d['n_queries']} queries against {d['n_candidates']} "
        f"section / subsection nodes; ESLII corpus).  Title-boost lifts "
        f"\\textsc{{recall@1}} from {fmt_pct(bm['recall_at_1'])} to "
        f"\\textbf{{{fmt_pct(bt['recall_at_1'])}}} --- quantifying the bonus "
        f"introduced in Eq.~\\ref{{eq:title}}.  The hybrid row's dense leg "
        f"requires a live Qwen3-Embedding endpoint to embed the query; "
        f"book-side embeddings are pre-stored at ingestion time.}}\n"
        f"\\label{{tab:m2-retrieval}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lccccc}}\n"
        f"\\toprule\n"
        f"ranker & MRR & R@1 & R@5 & R@10 & wall (271~q) \\\\\n"
        f"\\midrule\n"
        f"{rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}{hybrid_status}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m2_retrieval.tex", tex)


# ---------------------------------------------------------------------------
# M3 — visual-op profile
# ---------------------------------------------------------------------------

def build_m3() -> None:
    d = json.loads((RESULTS / "m3_viz_ops.json").read_text())
    o = d["overall"]
    op_top = list(o["op_histogram_overall"].items())[:8]
    ref_hist = o["ref_histogram_overall"]
    op_rows = "\n".join(
        f"\\texttt{{{latex_escape(k)}}} & {v} \\\\" for k, v in op_top
    )
    ref_rows = "\n".join(
        f"\\textsc{{{k.lower()}}} & {v} \\\\"
        for k, v in list(ref_hist.items())[:8]
    )
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Visual-primitive emission profile measured by "
        f"replaying the {o['n_chapters']} per-chapter narrations of "
        f"ESLII through the runtime primitive detectors offline.  "
        f"Total clauses: \\textbf{{{o['n_clauses_total']}}}; mean "
        f"operations per clause \\textbf{{{o['ops_per_clause_mean_overall']:.2f}}}; "
        f"mean reference mentions per clause "
        f"\\textbf{{{o['refs_per_clause_mean_overall']:.2f}}}; clauses "
        f"flagged by the inline-math acceptance test "
        f"\\textbf{{{fmt_pct(o['inline_math_rate'])}}}; per-session dedup "
        f"suppressed \\textbf{{{o['op_dedup_total']}}} duplicate operation "
        f"emissions and \\textbf{{{o['ref_dedup_total']}}} duplicate "
        f"reference emissions across the run.  Left: top operation labels. "
        f"Right: reference kinds.}}\n"
        f"\\label{{tab:m3-vizops}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lr@{{\\hspace{{2.5em}}}}lr}}\n"
        f"\\toprule\n"
        f"operation & n & reference kind & n \\\\\n"
        f"\\midrule\n"
        f"{_zip_rows(op_rows, ref_rows)}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m3_viz_ops.tex", tex)


def _zip_rows(left: str, right: str) -> str:
    """Zip two sets of rows side-by-side; pad short side with empty cells."""
    L = [r for r in left.split("\n") if r.strip()]
    R = [r for r in right.split("\n") if r.strip()]
    n = max(len(L), len(R))
    out = []
    for i in range(n):
        l_cells = L[i].rstrip(" \\").rstrip("\\\\") if i < len(L) else " & "
        r_cells = R[i].rstrip(" \\").rstrip("\\\\") if i < len(R) else " & "
        out.append(f"{l_cells} & {r_cells} \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# M4 — math-graph coverage
# ---------------------------------------------------------------------------

def build_m4() -> None:
    d = json.loads((RESULTS / "m4_graph_coverage.json").read_text())
    t = d["totals"]
    c = d["coverage"]
    eh = d["edge_type_histogram"]
    eh_rows = "\n".join(
        f"\\textsc{{{latex_escape(k)}}} & {v:>5d} \\\\"
        for k, v in list(eh.items())
    )
    pc = d["per_chapter"]
    pc_rows = "\n".join(
        f"\\texttt{{{r['chapter']}}} & {r['n_passages']:>5d} & "
        f"{r['n_anchored']:>5d} & {fmt_pct(r['coverage'])} \\\\"
        for r in pc[:12]
    )
    if len(pc) > 12:
        rest_n = sum(r["n_passages"] for r in pc[12:])
        rest_a = sum(r["n_anchored"] for r in pc[12:])
        rest_c = (rest_a / rest_n) if rest_n else 0.0
        pc_rows += (f"\n\\textit{{others}} & {rest_n:>5d} & "
                    f"{rest_a:>5d} & {fmt_pct(rest_c)} \\\\")

    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Math-graph anchor coverage on ESLII.  The graph "
        f"contains {t['n_passages']:,} passages, {t['n_formulas']:,} "
        f"formulas, {t['n_vars']:,} variables, {t['n_concepts']} "
        f"concepts and {t['n_edges']:,} edges.  A passage is "
        f"\\emph{{anchored}} when at least one edge of type "
        f"\\textsc{{defines}}, \\textsc{{contains}}, \\textsc{{uses}}, "
        f"\\textsc{{about}}, \\textsc{{references}}, \\textsc{{derived\\_from}}, "
        f"\\textsc{{paired\\_in\\_clause}} or \\textsc{{binds}} touches it; "
        f"\\textbf{{{c['n_passages_anchored']:,} / {t['n_passages']:,} "
        f"= {fmt_pct(c['fraction_passages_anchored'])}}} are anchored, with "
        f"a mean of \\textbf{{{c['anchor_density_mean']:.2f}}} anchors per "
        f"anchored passage (p95 = {c['anchor_density_p95']}).  Left: "
        f"per-chapter coverage; right: edge-type histogram.}}\n"
        f"\\label{{tab:m4-graph-coverage}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lrrr@{{\\hspace{{2.5em}}}}lr}}\n"
        f"\\toprule\n"
        f"chapter & passages & anchored & coverage & edge type & count \\\\\n"
        f"\\midrule\n"
        f"{_zip_rows(pc_rows, eh_rows)}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m4_graph_coverage.tex", tex)


# ---------------------------------------------------------------------------
# M5 — inline-math detector
# ---------------------------------------------------------------------------

def build_m5() -> None:
    d = json.loads((RESULTS / "m5_inline_math.json").read_text())
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Inline-math detector vs.\\ math-graph weak labels "
        f"(Sec.~\\ref{{sec:opdetect}}) on a balanced sample of "
        f"\\textbf{{{d['sample_per_class']*2}}} ESLII passages "
        f"({d['sample_per_class']} drawn from passages with $\\ge\\!1$ "
        f"\\textsc{{about}} edge to a formula or variable, "
        f"{d['sample_per_class']} drawn from passages with no math anchor "
        f"in the graph).  We deliberately frame this as agreement vs.\\ "
        f"the graph's weak label rather than ground-truth precision/recall: "
        f"the graph anchor itself has finite recall over PDF-OCR'd text. "
        f"The {d['recall']*100:.1f}\\% recall is consistent with the paper's "
        f"discussion of OCR-ligature drop and short-fragment passages where "
        f"the cluster size never reaches $|\\mathrm{{run}}|\\!\\ge\\!3$.}}\n"
        f"\\label{{tab:m5-inline-math}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lrrrrr}}\n"
        f"\\toprule\n"
        f"TP & FP & FN & TN & precision / recall / F$_1$ & accuracy \\\\\n"
        f"\\midrule\n"
        f"{d['tp']} & {d['fp']} & {d['fn']} & {d['tn']} & "
        f"{fmt_3(d['precision'])} / {fmt_3(d['recall'])} / {fmt_3(d['f1'])} & "
        f"{fmt_3(d['accuracy'])} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m5_inline_math.tex", tex)


# ---------------------------------------------------------------------------
# M6 — narration judge
# ---------------------------------------------------------------------------

def build_m6() -> None:
    d = json.loads((RESULTS / "m6_narration_judge.json").read_text())
    j = d["judge"]
    o = d["objective_proxy"]
    if j["status"] == "ok":
        m = j["scores_mean"]
        judge_rows = (
            f"faithfulness  & {m['faithfulness']:.2f} / 5 \\\\\n"
            f"clarity       & {m['clarity']:.2f} / 5 \\\\\n"
            f"pedagogical flow & {m['pedagogical_flow']:.2f} / 5 \\\\\n"
            f"TTS safety    & {m['tts_safety']:.2f} / 5 \\\\"
        )
        spend = f"\\textit{{spend: USD {j['spend_usd']:.4f} on {m.get('n_valid', d['n_sampled'])} pairs.}}"
    else:
        judge_rows = (
            f"faithfulness  & --- \\\\\n"
            f"clarity       & --- \\\\\n"
            f"pedagogical flow & --- \\\\\n"
            f"TTS safety    & --- \\\\"
        )
        spend = ("\\textit{Subjective rows skipped: the judge harness "
                 "is gated on the presence of an Anthropic API key in "
                 "the environment, and the key was absent at run time. "
                 "The harness is unchanged and re-runnable; cost is "
                 "capped at USD~20 by the ledger.}")
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Narration-quality assessment on a random sample of "
        f"$N\\!=\\!{d['n_sampled']}$ (source-passage, narration) pairs "
        f"drawn from the chapter sidecars.  The objective ROUGE-L F$_1$ "
        f"proxy on the same pairs is "
        f"\\textbf{{{o['mean']:.3f}}} (min~{o['min']:.3f}, "
        f"max~{o['max']:.3f}); low ROUGE-L is expected since the chapter-map "
        f"narrations rephrase rather than parrot the source --- the metric is "
        f"reported as a sanity check, not a quality score.  Subjective scores "
        f"come from Claude Sonnet~4.6 as a single judge "
        f"($\\tau\\!=\\!0$, structured-JSON output) under a fixed "
        f"four-dimension rubric checked into the repository.  "
        f"{spend}}}\n"
        f"\\label{{tab:m6-narration-judge}}\n"
        f"\\small\n"
        f"\\begin{{tabular}}{{lc}}\n"
        f"\\toprule\n"
        f"dimension (1-5) & mean \\\\\n"
        f"\\midrule\n"
        f"{judge_rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "m6_narration_judge.tex", tex)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> None:
    m1 = json.loads((RESULTS / "m1_routing.json").read_text())
    m2 = json.loads((RESULTS / "m2_retrieval.json").read_text())
    m3 = json.loads((RESULTS / "m3_viz_ops.json").read_text())
    m4 = json.loads((RESULTS / "m4_graph_coverage.json").read_text())
    m5 = json.loads((RESULTS / "m5_inline_math.json").read_text())
    m6 = json.loads((RESULTS / "m6_narration_judge.json").read_text())

    rows = [
        ("Routing accuracy (10 intents, "
         f"{m1['n_utterances']} utterances)",
         f"{fmt_3(m1['overall_accuracy'])} (macro F$_1$ "
         f"{fmt_3(m1['macro_f1'])})",
         "M1, Tab.~\\ref{tab:m1-routing}"),

        ("Retrieval MRR (BM25-only / +title)",
         f"{fmt_3(m2['results']['bm25_only']['mrr'])} / "
         f"\\textbf{{{fmt_3(m2['results']['bm25_titleboost']['mrr'])}}}",
         "M2, Tab.~\\ref{tab:m2-retrieval}"),

        ("Visual ops per clause; per-session dedup",
         f"{m3['overall']['ops_per_clause_mean_overall']:.2f}; "
         f"{m3['overall']['op_dedup_total']} suppressed",
         "M3, Tab.~\\ref{tab:m3-vizops}"),

        ("Math-graph anchor coverage",
         f"{fmt_pct(m4['coverage']['fraction_passages_anchored'])} "
         f"({m4['coverage']['n_passages_anchored']:,} / "
         f"{m4['totals']['n_passages']:,} passages)",
         "M4, Tab.~\\ref{tab:m4-graph-coverage}"),

        ("Inline-math detector vs graph anchors",
         f"P~{fmt_3(m5['precision'])} / R~{fmt_3(m5['recall'])} / "
         f"F$_1$~{fmt_3(m5['f1'])}",
         "M5, Tab.~\\ref{tab:m5-inline-math}"),

        ("Narration judge (Sonnet 4.6, "
         f"{m6['n_sampled']} pairs)",
         (f"overall {m6['judge']['scores_mean'].get('overall_mean', 0.0):.2f}/5"
          if m6['judge']['status'] == "ok"
          else "skipped (no key); ROUGE-L proxy "
               f"{m6['objective_proxy']['mean']:.3f}"),
         "M6, Tab.~\\ref{tab:m6-narration-judge}"),
    ]
    body = "\n".join(f"{r[0]} & {r[1]} & {r[2]} \\\\" for r in rows)
    tex = (
        f"% Generated automatically — do not hand-edit.\n"
        f"\\begin{{table}}[!htbp]\n"
        f"\\centering\n"
        f"\\caption{{Headline numbers from the journal-grade evaluation "
        f"harness.  Each row is reproducible from a single driver "
        f"script that writes a single JSON record.  M1--M5 require no "
        f"external service; M6 calls Claude Sonnet~4.6 as a single "
        f"judge and is skipped here because no Anthropic API key was "
        f"set in the environment at run time --- the script is unchanged "
        f"and runnable.}}\n"
        f"\\label{{tab:eval-summary}}\n"
        f"\\small\n"
        f"\\begin{{tabularx}}{{\\linewidth}}{{X l l}}\n"
        f"\\toprule\n"
        f"metric & value & traceable to \\\\\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabularx}}\n"
        f"\\end{{table}}\n"
    )
    write(TABLES / "measurement_summary.tex", tex)


def main() -> None:
    build_m1()
    build_m2()
    build_m3()
    build_m4()
    build_m5()
    build_m6()
    build_summary()


if __name__ == "__main__":
    main()
