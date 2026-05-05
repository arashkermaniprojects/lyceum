"""Journal-grade evaluation harness for Lyceum.

Re-runnable measurement scripts producing JSON in ``results/`` and LaTeX
fragments in ``tables/``.  Every metric is traceable to the script that
emitted it.  No invented numbers — when an upstream service is down or
an API key is absent the harness writes a structured ``skipped`` record
so downstream table generators render an explicit ``--`` cell rather
than a fabricated value.

Surface metrics
---------------
* M1 ``routing_accuracy``        — drivers/m1_routing.py
* M2 ``retrieval_ranking``       — drivers/m2_retrieval.py
* M3 ``visual_op_profile``       — drivers/m3_viz_ops.py
* M4 ``math_graph_coverage``     — drivers/m4_graph_coverage.py
* M5 ``inline_math_pr``          — drivers/m5_inline_math.py
* M6 ``narration_judge``         — drivers/m6_narration_judge.py  (Sonnet 4.6)
"""
