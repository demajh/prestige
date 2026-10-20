# Decision records (provenance)

A decision record ties one write to the reason it happened. It is committed in the **same RocksDB transaction**
as the value it describes, so a reader can never find a value without the record that explains it, nor a record
whose content commitment never happened.

The record carries commitments, not bodies:

| Field | Set by | Meaning |
|---|---|---|
| `decision_id` | caller | Unique id of the decision that caused the write. Replaying the same decision for the same write is a no-op; reusing the id for a different write is refused. |
| `policy_revision` | caller | Identifier or hash of the policy that was in force when the decision ran. |
| `parent_decision_id` | caller, optional | The decision this one follows. |
| `input_digests` | caller, optional | SHA-256 content keys of the values the decision consumed (`Store::Digest`). The bodies may already be in the store, arrive later, or be evicted. |
| `note` | caller, optional | Short free text. |
| `output_digest` | store | Content key of the value that was written. Always resolvable at commit time. |
| `user_key`, `sequence`, `committed_at_us` | store | The key written, the write-order sequence number, and the wall-clock commit time (see [Witnessed and declared](#witnessed-and-declared)). |

[Who stamps what, field by field](#who-stamps-what-field-by-field) gives the same split for every field, with what the
store checks before it accepts each one, and extends it to value metadata and outcome records.

## Why commitments and not bodies

A missing body is detectable: the digest is in the record, the store has no object for it, and a reader can say
exactly what is absent. A missing hash is not detectable from anything, because the write would contain no trace
that the decision depended on particular inputs. So the hashes must land in the same transaction as the mutation,
and the bodies can be deferred I/O with a receipt. prestige's existing content key (the SHA-256 that drives
deduplication) already serves as the input snapshot hash; what the record adds is the pointer from the decision
back to that key, the policy revision, and the parent decision.

## Witnessed and declared

A record has two kinds of field, and a reader should keep them apart.

**Witnessed** fields are set by the store inside the commit: `sequence`, `committed_at_us`, `user_key` and
`output_digest`. A caller cannot supply or alter them; the `Decision` struct does not carry them, so there is nothing
to pass in. Together they witness three facts and no more: that the write happened, where it falls among this
store's decision commits, and what the store host's clock read when it committed.

**Declared** fields are what the caller says: `decision_id`, `policy_revision`, `parent_decision_id`,
`input_digests`, `note`, and every pair of metadata. The store checks their shape and keeps them verbatim. It does
not know who the caller was, which runner or build issued the call, or whether the policy revision names a policy
that was really in force. Authorship is declared; existence, order and time are witnessed. A store that took the
caller's word for a witnessed field would only hand the caller a longer route to the same claim, which is why there
is no way to pass one in.

Two consequences for reading the witnessed fields:

- `sequence` orders the decision commits of this one store. It says nothing about reads, about other stores, about
  when the caller's own work finished, or about the order of two writes that went to different stores. Compare
  sequences only within a store.
- `committed_at_us` is the store host's wall clock at commit, in microseconds since the epoch. It is not monotonic
  (the host clock can step) and it is not the caller's clock. A caller that has its own completion stamp should keep
  it as a separate declared field rather than read `committed_at_us` as the same event.

If who or what minted a record matters, declare it with the write. The metadata overload of `PutWithDecision`
(see [candidates.md](candidates.md#value-metadata)) commits a small string map in the same transaction as the
value and the record:

```cpp
prestige::Decision d;
d.decision_id = "dec-2026-10-13-0007";
d.policy_revision = "policy@3f9c2a";
d.input_digests = {input};
store->PutWithDecision("memory/summary/42", summary_bytes, d,
                       prestige::Metadata{{"actor", "summarizer"},
                                          {"runner", "worker-17"},
                                          {"build", "2026.10.13+g9f1c2a"},
                                          {"clock", "worker-17 wall clock, NTP-synced"},
                                          {"completed_at_us", "1760400000123456"}});
```

```python
store.put("memory/summary/42", summary,
          decision=prestige.Decision("dec-2026-10-13-0007", "policy@3f9c2a", input_digests=inputs),
          metadata={"actor": "summarizer", "runner": "worker-17", "build": "2026.10.13+g9f1c2a",
                    "clock": "worker-17 wall clock, NTP-synced", "completed_at_us": "1760400000123456"})
```

Every pair in that map is declared: the record then shows which runner, build and clock the caller claims, beside
the one clock reading the store witnessed. Metadata belongs to the value, and writing the same bytes again merges
the maps with later pairs winning per key, so a field that must keep its first value (the `actor` that originally
wrote it) goes in the first write only; see the merge rule in [candidates.md](candidates.md#value-metadata).

`input_digests` cannot tell "this decision consumed nothing" from "the inputs were not recorded": both are an empty
list, in C++ and in Python, where omitting `input_digests` is the same as passing `[]`. When a decision consumed
nothing, say so in `note` (`inputs: none`); read an empty list without that note as not recorded, never as none.

## Who stamps what, field by field

An agent who wrote to us over A2A asked: "Which fields of a decision record does the store stamp itself?" Four. The
store stamps `sequence`, `committed_at_us`, `user_key` and `output_digest`, and nothing else. The record's identity
(`decision_id`) is the caller's, and so are the verdict and its cause on an outcome record (`verdict`, `reason`), where
the store stamps only `sequence` and `recorded_at_us`. Every caller field is kept **verbatim**: the bytes come back as
they went in, with no trimming, normalization, encoding check or truncation. A field over its limit is not cut; the
whole write is refused with `InvalidArgument` and nothing is stored.

So a reader can tell the two kinds of timestamp apart by name alone. `committed_at_us` on a decision record and
`recorded_at_us` on an outcome record are the only times the store writes. Any other time in a record, whether in
`note`, in a metadata pair or in a field the caller named `completed_at_us`, is a claim the caller made, and the store
did not read a clock to check it.

### Decision record

| Field | Set by | Checked at write | What the store keeps |
|---|---|---|---|
| `decision_id` | caller | Required, at most 4096 bytes. An id that already names a different write (another key or other content) is refused; the same key and content is a replay and a no-op. | Verbatim. Also the key the record is looked up by. |
| `policy_revision` | caller | Required, at most 4096 bytes. | Verbatim. Never resolved against a policy. |
| `parent_decision_id` | caller | At most 4096 bytes. Not checked to exist. | Verbatim. |
| `input_digests` | caller | Each exactly 32 bytes, at most 65536 of them. The bodies are not checked to exist at write time; that is the job of `GetDecision` and the sweep. | Verbatim, in the order given. The Python binding turns a 64-character hex string into its 32 bytes before the store sees it. |
| `note` | caller | At most 65536 bytes. | Verbatim. |
| `sequence` | store | | Taken from an in-memory counter once per `PutWithDecision` call, before the first attempt, so a retried transaction keeps its number and a failed call leaves a gap. Monotone within one store. |
| `committed_at_us` | store | | The store's clock, read inside the transaction just before commit, microseconds since the Unix epoch. The clock is the host wall clock unless the process that opened the store set `Options::custom_clock`, so the stamp is as good as that process. |
| `user_key` (`key` in Python) | store | | Copied from the key argument of the `PutWithDecision` call, not from the `Decision`: the key the store actually wrote. |
| `output_digest` | store | | SHA-256 the store computed over the written bytes, after the store's normalization when one is configured. `Digest` returns the same value for the same bytes. |

Three more names come back from `GetDecision` and are in neither column: `resolved`, `missing_digests` and
`queued_for_repair` (`DecisionCheck` in C++, keys of the same dict in Python). They are computed at read time against
the bodies present at that moment and are never stored, so two reads of one record can disagree as bodies arrive or are
evicted.

### Value metadata written with the record

| Field | Set by | Checked at write | What the store keeps |
|---|---|---|---|
| each pair | caller | At most 64 pairs. Keys non-empty and at most 256 bytes, values at most 4096 bytes. | Each pair verbatim, but the map belongs to the value, not to the record: a later write of the same bytes with metadata merges the maps, later pairs winning per key, and the merged map must still fit 64 pairs. |

### Outcome record

| Field | Set by | Checked at write | What the store keeps |
|---|---|---|---|
| `family_id` | caller | Required, at most 512 bytes. | Verbatim. Also the prefix the record is listed under. |
| `verdict` | caller | Must be `accepted`, `rejected` or `no_candidate`. | As given. The store never derives a verdict and never changes one. |
| `reason` | caller | At most 4096 bytes. | Verbatim. |
| `tool_version`, `schema_version` | caller | At most 4096 bytes each. | Verbatim. |
| `decision_id` | caller | At most 4096 bytes. Not checked to exist. | Verbatim. |
| `candidate_object_id` | caller | Empty or exactly 16 bytes. Not checked to exist. | Verbatim. |
| `candidate_digest` | caller | Empty or exactly 32 bytes. Not checked to exist. | Verbatim. |
| `rank`, `similarity`, `reranker_score`, `threshold` | caller | No range check. | As given. They are the numbers from the list the caller saw (in Python, copied from the candidate dict unless overridden) and are never recomputed. A negative score reads as not recorded, `None` in Python. |
| `sequence` | store | | The next number of one counter shared by every family, taken under lock inside the transaction. |
| `recorded_at_us` | store | | The store's clock when the record was assembled, with the same meaning and caveats as `committed_at_us`. |

The consequence for reading outcomes is the same as for decisions. A verdict of `rejected` with the reason
`tenant mismatch` is a declared pair: the store will record it from a caller that ran the check and from one that did
not, and it cannot tell them apart. What it witnesses is that this verdict was recorded, at this sequence, at this
reading of its clock.

In Python the split can be made with the names alone:

```python
rec = store.get_decision("dec-0001")
stamped = {k: rec[k] for k in ("sequence", "committed_at_us", "key", "output_digest")}
declared = {k: rec[k] for k in ("decision_id", "policy_revision", "parent_decision_id", "input_digests", "note")}
read_time = {k: rec[k] for k in ("resolved", "missing_digests", "queued_for_repair")}

out = store.list_outcomes("support-faq@v3", limit=1)[0]
stamped = {k: out[k] for k in ("sequence", "recorded_at_us")}
declared = {k: v for k, v in out.items() if k not in stamped}
```

## Reading, failing, and repairing

- **`GetDecision`** resolves every digest the record references. If a body is missing it still returns the record
  and a `DecisionCheck` naming the missing digests, increments a persisted `dangling_reads` counter, and puts the
  record on the repair queue. **`VerifyDecision`** is the failing variant: it returns `Corruption` instead.
  Read-time failure is the guard. It catches the first bad read under real traffic, counts how widespread the
  problem is, and queues the repair before the second bad read.
- **`SweepDecisions(max_records)`** is the drain. It first re-checks every queued record (records whose bodies
  have since arrived leave the queue and count as `repairs`), then continues a full walk of all records in
  write order from where the previous sweep stopped, queueing anything dangling that nobody has read yet.
  Walking in write order means a body committed after its reference in a fast burst is found before it ages out.
  The cursor is persisted as the walk goes: atomically with every record it queues, and every
  `decision_sweep_cursor_interval` records otherwise (default 256), so a sweep interrupted by a crash resumes from
  its last checkpoint rather than re-examining what it had already verified.
- **Integrity debt** is the queue size after a sweep. If it grows faster than sweeps drain it, that delta is the
  number to alert on. It is reported by `GetHealth` (`decision_queue_size`), by the sweep result, and as the
  gauge `prestige.decision.queue_size`.

Bodies arrive by any `Put` whose content key matches; the key under which they arrive is irrelevant. Bodies
disappear when their last key is deleted or they are evicted, which turns a resolved record into a dangling one;
the guard and the sweep report that the same way.

## C++

```cpp
prestige::Decision d;
d.decision_id = "dec-2026-10-05-0001";
d.policy_revision = "policy@3f9c2a";
d.parent_decision_id = "dec-2026-10-05-0000";
std::string input;
store->Digest(observation_bytes, &input);       // content key of a value the decision consumed
d.input_digests = {input};

store->PutWithDecision("memory/summary/42", summary_bytes, d);   // value + record, one transaction

prestige::DecisionRecord rec;
prestige::DecisionCheck check;
store->GetDecision("dec-2026-10-05-0001", &rec, &check);         // check.resolved, check.missing_digests
store->VerifyDecision("dec-2026-10-05-0001");                    // Corruption if anything is missing

prestige::DecisionSweepStats st;
store->SweepDecisions(10000, &st);                               // st.repaired, st.queue_size (integrity debt)

std::vector<prestige::DecisionRecord> recent;
store->ListDecisions(&recent, 100, /*after_sequence=*/0);        // write order
```

## Python

```python
import prestige

with prestige.open("./store") as store:
    inputs = [store.digest(observation)]
    store.put("memory/summary/42", summary,
              decision=prestige.Decision("dec-0001", "policy@3f9c2a",
                                         parent_decision_id="dec-0000", input_digests=inputs))

    rec = store.get_decision("dec-0001")          # rec["resolved"], rec["missing_digests"], rec["queued_for_repair"]
    store.get_decision("dec-0001", strict=True)   # raises CorruptionError when a body is missing
    store.sweep_decisions(max_records=10000)      # {"checked", "dangling", "repaired", "queue_size", ...}
    store.list_decisions(limit=100)               # write order
    store.get_health()["decision_queue_size"]     # integrity debt
```

## Storage

Records live in their own column family, `prestige_decisions`, keyed by a one-byte prefix: `d` + decision id for
the record, `s` + sequence for the write-order index, `q` + sequence for the repair queue, `c` for the sweep
cursor and `m` + name for the persisted counters. Existing stores gain the column family on first open. Sequence
numbers are allocated in memory from the highest one on disk, so they are monotone within a process and continue
across restarts; a retried transaction keeps the number it was given, and a failed one leaves a gap.

Decision records are exact-mode only for now. Deleting a key never deletes a record: records are the audit trail.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `prestige.decision.recorded_total` | counter | Records committed with a write |
| `prestige.decision.replayed_total` | counter | Replays recognised as the same write |
| `prestige.decision.get_total` | counter | `GetDecision` calls |
| `prestige.decision.dangling_read_total` | counter | Reads that found a missing body |
| `prestige.decision.sweep_checked_total` | counter | Records verified by sweeps |
| `prestige.decision.sweep_checkpoint_total` | counter | Times a sweep persisted its cursor (with a queued record, every interval, at the end) |
| `prestige.decision.repaired_total` | counter | Queued records that resolved during a sweep |
| `prestige.decision.queue_size` | gauge | Integrity debt after the last sweep |

## Design credit

The field set (decision id, policy revision, input hashes, parent decision), the rule that anything whose absence
cannot be detected must live inside the same write, and the severity-sequenced repair (read-time guard first,
write-order sweep as the drain, queue growth as the integrity-debt metric) follow a public design exchange with
**c3po-clawd** on Moltbook in October 2026, who agreed to be credited by name. Their review of the implementation
added the rule that the sweep persists its cursor with each repair batch rather than once at the end of the call.
