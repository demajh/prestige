"""Tests for value metadata, the candidates call and outcome records (exact mode)."""

import pytest

import prestige


def test_put_with_metadata_and_get_metadata(store):
    store.put("k1", b"the answer", metadata={"family": "support-faq@v3", "tool": "resolver/1.4"})
    assert store.get_metadata("k1") == {"family": "support-faq@v3", "tool": "resolver/1.4"}
    store.put("k2", b"the answer", metadata={"tool": "resolver/1.5", "schema": "c@2"})  # same bytes
    merged = {"family": "support-faq@v3", "tool": "resolver/1.5", "schema": "c@2"}
    assert store.get_metadata("k1") == merged
    assert store.get_metadata("k2") == merged
    store.put("k3", b"plain")
    assert store.get_metadata("k3") == {}
    with pytest.raises(prestige.NotFoundError):
        store.get_metadata("missing")


def test_candidates_exact_mode(store):
    store.put("k1", b"the answer", metadata={"family": "a"})
    out = store.candidates(b"the answer")
    assert len(out) == 1
    c = out[0]
    assert c["rank"] == 0
    assert c["similarity"] == 1.0
    assert c["reranker_score"] is None
    assert c["metadata"] == {"family": "a"}
    assert c["digest"] == store.digest(b"the answer")
    assert len(c["object_id"]) == 16
    assert c["size_bytes"] == 10
    assert "value" not in c

    assert store.candidates(b"the answer", include_values=True)[0]["value"] == b"the answer"
    assert store.candidates(b"other") == []
    assert store.candidates(b"the answer", filter={"family": "a"})
    assert store.candidates(b"the answer", filter={"family": "b"}) == []


def test_metadata_validation(store):
    with pytest.raises(prestige.InvalidArgumentError):
        store.put("k", b"v", metadata={"": "x"})
    with pytest.raises(prestige.InvalidArgumentError):
        store.put("k", b"v", metadata={f"k{i}": "v" for i in range(65)})
    assert store.count_keys() == 0


def test_dict_style_assignment_still_works(store):
    store["k"] = b"v"
    assert store["k"] == b"v"


def test_outcomes_report_and_listing(store):
    store.put("k", b"the answer", metadata={"family": "support-faq@v3"})
    cand = store.candidates(b"the answer")[0]
    seq = store.record_outcome("support-faq@v3", "accepted", candidate=cand, threshold=0.92,
                               tool_version="resolver/1.4", schema_version="c@2")
    assert seq == 1
    assert store.record_outcome("support-faq@v3", "rejected", rank=1, similarity=0.93, reason="tenant mismatch") == 2
    assert store.record_outcome("support-faq@v3", "no_candidate") == 3
    assert store.record_outcome("billing@v1", "accepted", similarity=0.99) == 4

    with pytest.raises(ValueError):
        store.record_outcome("f", "maybe")

    outcomes = store.list_outcomes("support-faq@v3")
    assert [o["verdict"] for o in outcomes] == ["accepted", "rejected", "no_candidate"]
    assert outcomes[0]["candidate_object_id"] == cand["object_id"]
    assert outcomes[0]["candidate_digest"] == cand["digest"]
    assert outcomes[0]["similarity"] == 1.0 and outcomes[0]["threshold"] == pytest.approx(0.92)
    assert outcomes[1]["reason"] == "tenant mismatch" and outcomes[1]["rank"] == 1
    assert outcomes[2]["similarity"] is None
    assert store.list_outcomes("support-faq@v3", limit=1, after_sequence=1)[0]["sequence"] == 2

    report = store.family_report("support-faq@v3")
    assert report["accepted"] == 1 and report["rejected"] == 1 and report["no_candidate"] == 1
    assert report["false_accept_rate"] == 0.5
    assert len(report["accepted_similarity_hist"]) == 20 and report["accepted_similarity_hist"][19] == 1
    assert report["rejected_similarity_hist"][18] == 1
    assert report["rejected_rank_hist"][1] == 1
    assert report["suggested_threshold"] is None
    assert report["first_sequence"] == 1 and report["last_sequence"] == 3

    assert store.list_families() == ["billing@v1", "support-faq@v3"]
    with pytest.raises(prestige.NotFoundError):
        store.family_report("unknown")


def test_suggested_threshold_is_advisory(store):
    for _ in range(40):
        store.record_outcome("f", "accepted", similarity=0.96)
    for _ in range(2):
        store.record_outcome("f", "rejected", similarity=0.97, rank=0)
    for _ in range(10):
        store.record_outcome("f", "rejected", similarity=0.91, rank=2)
    report = store.family_report("f")
    assert report["suggested_threshold"] == pytest.approx(0.95)
    assert report["false_accept_rate"] == pytest.approx(12 / 52)
    assert report["rejected_rank_hist"][2] == 10
