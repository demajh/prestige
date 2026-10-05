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
| `user_key`, `sequence`, `committed_at_us` | store | The key written, the write-order sequence number, and the wall-clock commit time. |

## Why commitments and not bodies

A missing body is detectable: the digest is in the record, the store has no object for it, and a reader can say
exactly what is absent. A missing hash is not detectable from anything, because the write would contain no trace
that the decision depended on particular inputs. So the hashes must land in the same transaction as the mutation,
and the bodies can be deferred I/O with a receipt. prestige's existing content key (the SHA-256 that drives
deduplication) already serves as the input snapshot hash; what the record adds is the pointer from the decision
back to that key, the policy revision, and the parent decision.

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
