"""Run every policy on SIMULATED synthetic traces across device counts and skew levels.
Writes results/sim_results.json and results/REPORT.md."""
import argparse, json
from pathlib import Path

from moesim import serve, traces

ap = argparse.ArgumentParser()
ap.add_argument("--n-seqs", type=int, default=384)
ap.add_argument("--seq-len", type=int, default=128)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--experts", type=int, default=64)
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--devices", default="4,8,16")
ap.add_argument("--domain-conc", default="0.1,0.3,1.0", help="lower = each domain prefers fewer experts")
ap.add_argument("--seed", type=int, default=20261110)
ap.add_argument("--out-dir", default="results")
a = ap.parse_args()

rows = []
for conc in [float(x) for x in a.domain_conc.split(",")]:
    t = traces.synthetic(n_seqs=a.n_seqs, seq_len=a.seq_len, n_layers=a.layers, n_experts=a.experts,
                         top_k=a.top_k, n_domains=8, domain_conc=conc, seed=a.seed)
    calib, ev = traces.split_by_seq(t, 0.5, seed=a.seed)
    for D in [int(x) for x in a.devices.split(",")]:
        base = None
        for p in serve.POLICIES:
            r = serve.simulate(calib, ev, D, p)
            r = {k: (float(v) if hasattr(v, "item") else v) for k, v in r.items()}
            base = base or r
            r.update(domain_conc=conc, speedup_vs_naive=r["sim_throughput_tok_s"] / base["sim_throughput_tok_s"],
                     traffic_vs_naive=r["cross_device_bytes_per_token"] / base["cross_device_bytes_per_token"])
            rows.append(r)
            print(f"conc={conc} D={D} {p.name[:50]:50s} remote={r['remote_expert_frac']:.3f} "
                  f"traffic={r['traffic_vs_naive']:.3f} speedup={r['speedup_vs_naive']:.2f}", flush=True)
meta = {"label": "SIMULATED: synthetic router traces + analytic serving cost model", "hardware": serve.HW,
        "trace_params": t["params"], "calib_eval_split": "50/50 by sequence; placement fit on calib only"}
out = Path(a.out_dir)
(out / "sim_results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

md = ["**SIMULATED.** Synthetic router traces (Zipf global popularity × per-domain Dirichlet preference) and an",
      f"analytic cost model with ASSUMED hardware: {serve.HW}.", ""]
for conc in sorted({r["domain_conc"] for r in rows}):
    md += [f"#### domain concentration {conc} ({'strong' if conc <= 0.1 else 'medium' if conc <= 0.3 else 'weak'} request→expert locality)", "",
           "| devices | policy | remote expert hits | remote devices / token·layer | cross-device KB / token | traffic vs naive | load imbalance (max/mean) | sim throughput vs naive |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["domain_conc"] != conc:
            continue
        md.append(f"| {r['n_devices']} | {r['policy']} | {100*r['remote_expert_frac']:.1f}% | {r['remote_devices_per_token_layer']:.2f} | "
                  f"{r['cross_device_bytes_per_token']/1024:.1f} | {r['traffic_vs_naive']:.2f}× | {r['expert_load_imbalance']:.2f} | {r['speedup_vs_naive']:.2f}× |")
    md.append("")
(out / "REPORT.md").write_text("\n".join(md))
