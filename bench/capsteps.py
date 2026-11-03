"""Decode step times (p50 / p90 ms per decode xN) of a capacity run, from the step trace."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report
for t in sys.argv[1:]:
    meta = json.load(open(os.path.join(report.HERE, "results", t, "capacity.meta.json")))
    st = report.steps_summary(report.load_steps(meta["t0"], meta.get("t1") or meta.get("saved")))
    print(t, " ".join(f"{k}: p50 {v['p50']:.1f} p90 {v['p90']:.1f} (n {v['n']})" for k, v in st.items() if k.startswith("decode")))
