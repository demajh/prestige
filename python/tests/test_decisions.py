"""Tests for decision records (provenance committed with the write)."""

import pytest

import prestige


def test_put_with_decision_round_trips(store):
    d = prestige.Decision("dec-1", "policy-v1", parent_decision_id="dec-0", note="first")
    store.put("k1", b"value one", decision=d)
    assert store.get("k1") == b"value one"

    rec = store.get_decision("dec-1")
    assert rec["decision_id"] == "dec-1"
    assert rec["policy_revision"] == "policy-v1"
    assert rec["parent_decision_id"] == "dec-0"
    assert rec["note"] == "first"
    assert rec["key"] == "k1"
    assert rec["sequence"] == 1
    assert rec["committed_at_us"] > 0
    assert rec["output_digest"] == store.digest(b"value one")
    assert len(rec["output_digest"]) == 32
    assert rec["resolved"] is True
    assert rec["missing_digests"] == []
    assert rec["queued_for_repair"] is False

    health = store.get_health()
    assert health["decisions_total"] == 1
    assert health["decision_queue_size"] == 0


def test_unknown_decision_raises(store):
    with pytest.raises(prestige.NotFoundError):
        store.get_decision("missing")


def test_validation(store):
    with pytest.raises(prestige.InvalidArgumentError):
        store.put("k", b"v", decision=prestige.Decision("", "policy"))
    with pytest.raises(prestige.InvalidArgumentError):
        store.put("k", b"v", decision=prestige.Decision("dec", ""))
    with pytest.raises(ValueError):
        prestige.Decision("dec", "policy", input_digests=[b"short"])
    assert store.count_keys() == 0


def test_input_digests_accept_bytes_or_hex(store):
    raw = store.digest(b"input body")
    d = prestige.Decision("dec", "policy", input_digests=[raw, raw.hex()])
    assert d.input_digests == [raw, raw]


def test_missing_input_is_detected_queued_and_repaired(store):
    missing = store.digest(b"input body")
    store.put("k1", b"derived", decision=prestige.Decision("dec-1", "policy", input_digests=[missing]))

    rec = store.get_decision("dec-1")
    assert rec["resolved"] is False
    assert rec["missing_digests"] == [missing]
    assert rec["queued_for_repair"] is True

    with pytest.raises(prestige.CorruptionError):
        store.get_decision("dec-1", strict=True)

    health = store.get_health()
    assert health["decision_queue_size"] == 1
    assert health["decision_dangling_reads"] == 2  # the strict read counted too

    sweep = store.sweep_decisions()
    assert sweep["dangling"] == 1 and sweep["repaired"] == 0 and sweep["queue_size"] == 1

    store.put("anywhere", b"input body")  # the body arrives
    sweep = store.sweep_decisions()
    assert sweep["repaired"] == 1 and sweep["queue_size"] == 0
    assert store.get_decision("dec-1", strict=True)["resolved"] is True
    assert store.get_health()["decision_repairs"] == 1


def test_replay_is_idempotent_and_reuse_is_refused(store):
    d = prestige.Decision("dec-1", "policy")
    store.put("k1", b"v", decision=d)
    store.put("k1", b"v", decision=d)  # replay after a crash: no-op
    assert len(store.list_decisions()) == 1
    with pytest.raises(prestige.InvalidArgumentError):
        store.put("k2", b"other", decision=d)
    assert "k2" not in store


def test_list_decisions_in_write_order(store):
    for i in range(1, 4):
        store.put(f"k{i}", f"v{i}", decision=prestige.Decision(f"dec-{i}", "policy"))
    ids = [r["decision_id"] for r in store.list_decisions()]
    assert ids == ["dec-1", "dec-2", "dec-3"]
    assert [r["decision_id"] for r in store.list_decisions(limit=1, after_sequence=1)] == ["dec-2"]


def test_decision_requires_exact_mode_only_when_semantic(store):
    # In the default exact-mode store, digest() and decisions are available.
    assert len(store.digest("text")) == 32
    repr_text = repr(prestige.Decision("dec", "policy"))
    assert "dec" in repr_text and "policy" in repr_text
