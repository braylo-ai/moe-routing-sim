"""Expert-parallel serving cost model (SIMULATED). Hardware numbers are ASSUMPTIONS in HW."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import placement as P

HW = {"d_model": 2048, "d_ff_expert": 1024, "bytes_per_act": 2, "tflops_per_device": 400.0,
      "link_GBps_per_device": 50.0, "tokens_per_step_per_device": 512, "label": "ASSUMED hardware"}


@dataclass
class Policy:
    name: str
    placement: str = "contiguous"   # contiguous | coactivation | affinity
    request_routing: str = "round_robin"  # round_robin | locality
    replicas: int = 0               # hot experts replicated on every device (per layer)
    prefix_tokens: int = 16         # tokens used to predict a request's experts
    load_slack: float = 0.10        # bounded load for locality request routing
    extra: dict = field(default_factory=dict)


def build_placements(calib: dict, n_devices: int, policy: Policy):
    E, L = calib["n_experts"], calib["experts"].shape[1]
    places, hots = [], []
    aff = P.affinity(calib, n_devices) if policy.placement == "affinity" else None
    for layer in range(L):
        ex = calib["experts"][:, layer]
        if policy.placement == "contiguous":
            places.append(P.contiguous(E, n_devices))
        elif policy.placement == "coactivation":
            places.append(P.clustered(ex, E, n_devices))
        else:
            places.append(aff[layer])
        hots.append(P.hot_experts(ex, E, policy.replicas) if policy.replicas else np.array([], dtype=int))
    return places, hots


def route_requests(trace: dict, places, n_devices: int, policy: Policy) -> dict[int, int]:
    """Assign each sequence (request) to a device."""
    seqs, starts = np.unique(trace["seq"], return_index=True)
    order = np.argsort(starts)
    seqs = seqs[order]
    if policy.request_routing == "round_robin":
        return {int(s): i % n_devices for i, s in enumerate(seqs)}
    lengths = {int(s): int((trace["seq"] == s).sum()) for s in seqs}
    cap = (1 + policy.load_slack) * sum(lengths.values()) / n_devices
    used = np.zeros(n_devices)
    out = {}
    for s in seqs:
        idx = np.flatnonzero(trace["seq"] == s)[: policy.prefix_tokens]
        aff = np.zeros(n_devices)
        for layer, pl in enumerate(places):
            np.add.at(aff, pl[trace["experts"][idx, layer].ravel()], 1)
        ok = used + lengths[int(s)] <= cap
        if not ok.any():
            ok = used == used.min()
        aff = np.where(ok, aff, -np.inf)
        best = np.flatnonzero(aff == aff.max())
        d = int(best[np.argmin(used[best])])
        out[int(s)] = d
        used[d] += lengths[int(s)]
    return out


def simulate(calib: dict, evalset: dict, n_devices: int, policy: Policy, hw: dict = HW) -> dict:
    places, hots = build_placements(calib, n_devices, policy)
    req = route_requests(evalset, places, n_devices, policy)
    origin = np.array([req[int(s)] for s in evalset["seq"]])
    ex = evalset["experts"]
    n_tok, L, k = ex.shape
    act_bytes = hw["d_model"] * hw["bytes_per_act"]
    flops_assign = 2 * 3 * hw["d_model"] * hw["d_ff_expert"]
    S = hw["tokens_per_step_per_device"]
    # step index of each token inside its device queue
    step = np.zeros(n_tok, dtype=int)
    for d in range(n_devices):
        m = np.flatnonzero(origin == d)
        step[m] = np.arange(len(m)) // S
    n_steps = step.max() + 1
    total_time = 0.0
    remote_pairs = 0
    remote_dev_pairs = 0
    total_bytes = 0.0
    assign = np.zeros(n_devices)
    for layer in range(L):
        dest = places[layer][ex[:, layer]]                 # [n_tok, k] device of each chosen expert
        if len(hots[layer]):
            is_hot = np.isin(ex[:, layer], hots[layer])
            dest = np.where(is_hot, origin[:, None], dest)  # replicated -> served locally
        remote = dest != origin[:, None]
        remote_pairs += remote.sum()
        # one dispatch+combine per (token, remote device), even if several experts live there
        uniq = np.zeros((n_tok, n_devices), dtype=bool)
        np.put_along_axis(uniq, dest, True, axis=1)
        uniq[np.arange(n_tok), origin] = False
        remote_dev_pairs += uniq.sum()
        comp = np.zeros((n_steps, n_devices))
        np.add.at(comp, (np.repeat(step, k), dest.ravel()), 1)
        out_b = np.zeros((n_steps, n_devices))
        in_b = np.zeros((n_steps, n_devices))
        tok_dev_steps = np.repeat(step, n_devices).reshape(n_tok, n_devices)
        np.add.at(out_b, (step, origin), uniq.sum(1) * act_bytes * 2)
        np.add.at(in_b, (tok_dev_steps[uniq], np.nonzero(uniq)[1]), act_bytes * 2)
        assign += comp.sum(0)
        t_comp = comp.max(1) * flops_assign / (hw["tflops_per_device"] * 1e12)
        t_comm = np.maximum(out_b, in_b).max(1) / (hw["link_GBps_per_device"] * 1e9)
        total_time += float((t_comp + t_comm).sum())
        total_bytes += float(out_b.sum())
    return {"policy": policy.name, "n_devices": n_devices, "tokens": int(n_tok), "layers": int(L), "top_k": int(k),
            "remote_expert_frac": remote_pairs / (n_tok * L * k),
            "remote_devices_per_token_layer": remote_dev_pairs / (n_tok * L),
            "cross_device_bytes_per_token": total_bytes / n_tok,
            "expert_load_imbalance": float(assign.max() / assign.mean()),
            "sim_time_s": total_time, "sim_throughput_tok_s": n_tok / total_time}


POLICIES = [
    Policy("naive: contiguous experts, round-robin requests"),
    Policy("co-activation placement, round-robin requests", placement="coactivation"),
    Policy("affinity placement, round-robin requests", placement="affinity"),
    Policy("locality-aware: affinity placement + locality request routing", placement="affinity", request_routing="locality"),
    Policy("locality-aware + 2 hot-expert replicas/layer", placement="affinity", request_routing="locality", replicas=2),
]
