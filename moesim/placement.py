"""Expert -> device placement per layer."""
from __future__ import annotations

import numpy as np


def contiguous(n_experts: int, n_devices: int) -> np.ndarray:
    """Naive: expert e lives on device e // (E / D)."""
    return np.arange(n_experts) * n_devices // n_experts


def coactivation(experts_layer: np.ndarray, n_experts: int) -> np.ndarray:
    """Counts of experts chosen together by the same token. experts_layer: [n_tokens, top_k]."""
    C = np.zeros((n_experts, n_experts))
    k = experts_layer.shape[1]
    for i in range(k):
        for j in range(k):
            if i != j:
                np.add.at(C, (experts_layer[:, i], experts_layer[:, j]), 1)
    return C


def clustered(experts_layer: np.ndarray, n_experts: int, n_devices: int, load_slack: float = 0.10) -> np.ndarray:
    """Greedy balanced partition: put strongly co-activated experts on the same device.
    Constraints: equal expert count per device; per-device token load <= (1+slack) x mean."""
    C = coactivation(experts_layer, n_experts)
    load = np.bincount(experts_layer.ravel(), minlength=n_experts).astype(float)
    cap_e = n_experts // n_devices
    cap_load = (1 + load_slack) * load.sum() / n_devices
    place = -np.ones(n_experts, dtype=int)
    dev_load = np.zeros(n_devices)
    dev_cnt = np.zeros(n_devices, dtype=int)
    for e in np.argsort(-load):  # heaviest first
        score = np.array([C[e, place == d].sum() for d in range(n_devices)])
        ok = (dev_cnt < cap_e) & (dev_load + load[e] <= cap_load)
        if not ok.any():
            ok = dev_cnt < cap_e
        score = np.where(ok, score, -np.inf)
        best = np.flatnonzero(score == score.max())
        d = best[np.argmin(dev_load[best])]
        place[e] = d
        dev_load[d] += load[e]
        dev_cnt[d] += 1
    return place


def hot_experts(experts_layer: np.ndarray, n_experts: int, n_replicas: int) -> np.ndarray:
    load = np.bincount(experts_layer.ravel(), minlength=n_experts)
    return np.argsort(-load)[:n_replicas]


def seq_profiles(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-sequence normalized expert-usage vectors, concatenated over layers: [n_seqs, L*E]."""
    seqs = np.unique(trace["seq"])
    E, L = trace["n_experts"], trace["experts"].shape[1]
    prof = np.zeros((len(seqs), L * E))
    pos = {int(s): i for i, s in enumerate(seqs)}
    rows = np.array([pos[int(s)] for s in trace["seq"]])
    for layer in range(L):
        for j in range(trace["experts"].shape[2]):
            np.add.at(prof, (rows, layer * E + trace["experts"][:, layer, j]), 1)
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
    return seqs, prof


def balanced_kmeans(X: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Cosine k-means with equal cluster sizes (greedy capacity-constrained assignment)."""
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)]
    cap = int(np.ceil(len(X) / k))
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        sim = X @ cent.T
        lab[:] = -1
        cnt = np.zeros(k, dtype=int)
        for flat in np.argsort(-sim, axis=None):
            i, c = divmod(int(flat), k)
            if lab[i] < 0 and cnt[c] < cap:
                lab[i] = c
                cnt[c] += 1
        new = np.stack([X[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
        if np.allclose(new, cent):
            break
        cent = new
    return lab


def affinity(calib: dict, n_devices: int, load_slack: float = 0.10, seed: int = 0) -> list[np.ndarray]:
    """Layer-consistent placement: cluster calibration requests into n_devices groups, then give each
    device the experts its group uses most (equal expert count, bounded token load). Same device
    serves the same request group in EVERY layer, so routing a request there keeps it local."""
    seqs, prof = seq_profiles(calib)
    lab = balanced_kmeans(prof, n_devices, seed=seed)
    group = {int(s): int(g) for s, g in zip(seqs, lab)}
    g_tok = np.array([group[int(s)] for s in calib["seq"]])
    E, L = calib["n_experts"], calib["experts"].shape[1]
    cap_e = E // n_devices
    out = []
    for layer in range(L):
        demand = np.zeros((n_devices, E))
        for j in range(calib["experts"].shape[2]):
            np.add.at(demand, (g_tok, calib["experts"][:, layer, j]), 1)
        load = demand.sum(0)
        cap_load = (1 + load_slack) * load.sum() / n_devices
        place = -np.ones(E, dtype=int)
        dev_load, dev_cnt = np.zeros(n_devices), np.zeros(n_devices, dtype=int)
        for flat in np.argsort(-demand, axis=None):
            d, e = divmod(int(flat), E)
            if place[e] >= 0 or dev_cnt[d] >= cap_e or dev_load[d] + load[e] > cap_load:
                continue
            place[e], dev_cnt[d] = d, dev_cnt[d] + 1
            dev_load[d] += load[e]
        for e in np.flatnonzero(place < 0):  # leftovers: least-loaded device with room
            room = np.flatnonzero(dev_cnt < cap_e)
            d = room[np.argmin(dev_load[room])]
            place[e], dev_cnt[d] = d, dev_cnt[d] + 1
            dev_load[d] += load[e]
        out.append(place)
    return out
