import numpy as np
import pytest

from moesim import placement as P
from moesim import serve, traces


@pytest.fixture(scope="module")
def tr():
    return traces.synthetic(n_seqs=48, seq_len=32, n_layers=3, n_experts=16, top_k=2, n_domains=4, seed=1)


def test_synthetic_shapes_and_distinct_topk(tr):
    ex = tr["experts"]
    assert ex.shape == (48 * 32, 3, 2)
    assert (ex[..., 0] != ex[..., 1]).all()
    assert ex.min() >= 0 and ex.max() < 16


def test_synthetic_is_seeded():
    a = traces.synthetic(n_seqs=4, seq_len=8, n_layers=2, n_experts=8, top_k=2, seed=5)
    b = traces.synthetic(n_seqs=4, seq_len=8, n_layers=2, n_experts=8, top_k=2, seed=5)
    assert np.array_equal(a["experts"], b["experts"])


def test_synthetic_is_skewed(tr):
    counts = np.bincount(tr["experts"][:, 0].ravel(), minlength=16)
    assert counts.max() > 2 * counts.mean() or counts.max() / max(counts.min(), 1) > 3


def test_split_by_seq_is_disjoint(tr):
    c, e = traces.split_by_seq(tr, 0.5)
    assert not set(c["seq"].tolist()) & set(e["seq"].tolist())
    assert len(c["seq"]) + len(e["seq"]) == len(tr["seq"])


def test_save_load_roundtrip(tr, tmp_path):
    p = tmp_path / "t.npz"
    traces.save(tr, str(p))
    t2 = traces.load(str(p))
    assert np.array_equal(t2["experts"], tr["experts"]) and t2["n_experts"] == 16


def test_contiguous_placement():
    assert P.contiguous(8, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("fn", ["clustered", "affinity"])
def test_placements_are_balanced(tr, fn):
    D = 4
    places = [P.clustered(tr["experts"][:, l], 16, D) for l in range(3)] if fn == "clustered" else P.affinity(tr, D)
    for pl in places:
        assert np.bincount(pl, minlength=D).tolist() == [4, 4, 4, 4]


def test_coactivation_symmetric():
    ex = np.array([[0, 1], [1, 2], [0, 1]])
    C = P.coactivation(ex, 3)
    assert C[0, 1] == 2 and C[1, 0] == 2 and C[0, 2] == 0


def test_balanced_kmeans_respects_capacity():
    X = np.random.default_rng(0).random((20, 5))
    lab = P.balanced_kmeans(X, 4)
    assert np.bincount(lab).max() <= 5


def test_single_device_has_no_traffic(tr):
    c, e = traces.split_by_seq(tr)
    r = serve.simulate(c, e, 1, serve.POLICIES[0])
    assert r["remote_expert_frac"] == 0 and r["cross_device_bytes_per_token"] == 0


def test_full_replication_removes_traffic(tr):
    c, e = traces.split_by_seq(tr)
    r = serve.simulate(c, e, 4, serve.Policy("all-replicated", replicas=16))
    assert r["cross_device_bytes_per_token"] == 0


def test_naive_remote_fraction_matches_uniform_expectation(tr):
    c, e = traces.split_by_seq(tr)
    r = serve.simulate(c, e, 4, serve.POLICIES[0])
    assert 0.6 < r["remote_expert_frac"] < 0.9  # ~ (D-1)/D under round-robin origin


def test_locality_routing_bounded_load(tr):
    c, e = traces.split_by_seq(tr)
    pol = serve.Policy("loc", placement="affinity", request_routing="locality", load_slack=0.1)
    places, _ = serve.build_placements(c, 4, pol)
    req = serve.route_requests(e, places, 4, pol)
    tok = np.bincount([req[int(s)] for s in e["seq"]], minlength=4)
    assert tok.max() <= 1.1 * tok.sum() / 4 + 32  # one sequence of slack


def test_locality_reduces_traffic_on_strongly_local_traces():
    t = traces.synthetic(n_seqs=128, seq_len=32, n_layers=4, n_experts=32, top_k=2, n_domains=4, domain_conc=0.05, seed=3)
    c, e = traces.split_by_seq(t)
    naive = serve.simulate(c, e, 4, serve.POLICIES[0])
    loc = serve.simulate(c, e, 4, serve.POLICIES[3])
    assert loc["cross_device_bytes_per_token"] < naive["cross_device_bytes_per_token"]
