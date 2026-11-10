"""Router traces. A trace is {"experts": int array [n_tokens, n_layers, top_k], "seq": int array [n_tokens],
"n_experts": E, "source": str}. Real traces come from scripts/capture_traces.py; synthetic ones from here."""
from __future__ import annotations

import numpy as np


def synthetic(n_seqs=256, seq_len=128, n_layers=8, n_experts=64, top_k=4, n_domains=8,
              zipf_a=1.1, domain_conc=0.3, seed=0) -> dict:
    """SIMULATED traces with realistic skew: global Zipf expert popularity x per-domain preference.
    Each sequence has one domain; tokens sample top_k distinct experts from the mixed distribution."""
    rng = np.random.default_rng(seed)
    experts = np.zeros((n_seqs * seq_len, n_layers, top_k), dtype=np.int16)
    seq = np.repeat(np.arange(n_seqs), seq_len)
    domains = rng.integers(0, n_domains, n_seqs)
    for layer in range(n_layers):
        rank = rng.permutation(n_experts)
        glob = 1.0 / (np.arange(1, n_experts + 1) ** zipf_a)
        glob = glob[np.argsort(rank)]
        glob /= glob.sum()
        dom_pref = rng.dirichlet(np.full(n_experts, domain_conc), n_domains)
        for s in range(n_seqs):
            p = 0.5 * glob + 0.5 * dom_pref[domains[s]]
            p /= p.sum()
            # Gumbel top-k == sampling k distinct experts with prob proportional to p
            g = np.log(p)[None, :] - np.log(-np.log(rng.random((seq_len, n_experts))))
            experts[s * seq_len:(s + 1) * seq_len, layer] = np.argsort(-g, axis=1)[:, :top_k]
    return {"experts": experts, "seq": seq, "n_experts": n_experts, "source": "SIMULATED-synthetic",
            "params": dict(n_seqs=n_seqs, seq_len=seq_len, n_layers=n_layers, n_experts=n_experts, top_k=top_k,
                           n_domains=n_domains, zipf_a=zipf_a, domain_conc=domain_conc, seed=seed)}


def save(trace: dict, path: str) -> None:
    np.savez_compressed(path, experts=trace["experts"], seq=trace["seq"], n_experts=trace["n_experts"],
                        source=trace["source"])


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"experts": z["experts"], "seq": z["seq"], "n_experts": int(z["n_experts"]), "source": str(z["source"])}


def split_by_seq(trace: dict, frac: float = 0.5, seed: int = 0) -> tuple[dict, dict]:
    """Calibration / evaluation split on whole sequences (placement is fit on calib only)."""
    seqs = np.unique(trace["seq"])
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    cut = set(seqs[: int(len(seqs) * frac)].tolist())
    m = np.array([s in cut for s in trace["seq"]])
    pick = lambda mask: {**trace, "experts": trace["experts"][mask], "seq": trace["seq"][mask]}
    return pick(m), pick(~m)
