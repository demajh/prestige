# Candidates, value metadata and outcomes

Deduplication decides for you: `Put` either reuses an existing value or stores a new one. For a semantic cache that
is the wrong place to decide. Two prompts at 0.96 cosine similarity can still demand different answers because one
of them carries a constraint the embedding barely sees: a tenant, a date range, a schema version, a side effect.
Only the caller knows those constraints. So the store should **rank and score, and the caller should decide**.

This document covers the three pieces that make that possible:

- **`Candidates`** returns the nearest stored values with their similarity, reranker score and metadata.
- **Value metadata** is a small string map attached to a stored value: task family, tool version, constraint-schema
  version, anything the caller's gate needs.
- **Outcomes** record what the caller decided about a candidate, per task family, so thresholds can be calibrated
  from measured false-accept rates instead of guesswork.

[outcomes-demo.md](outcomes-demo.md) runs this loop end to end over a synthetic workload of three task families and
shows the per-family reports it produces.

## Candidates

```cpp
prestige::CandidateQuery q;
q.k = 5;                                      // at most five
q.filter = {{"family", "support-faq@v3"}};    // only values written with this metadata
q.min_similarity = 0.85f;                     // optional floor; the store applies no threshold of its own
q.include_values = true;                      // also return the stored bytes

std::vector<prestige::Candidate> out;
store->Candidates(query_text, q, &out);
for (const auto& c : out) {
  // c.rank, c.similarity, c.reranker_score (-1 without a reranker), c.object_id, c.digest,
  // c.metadata, c.size_bytes, c.created_at_us, c.value
}
```

Semantic mode embeds the query, asks the vector index for at least `k` neighbours (more when a filter is set, since
filtering happens afterwards), computes the exact cosine similarity against each stored embedding, drops anything
below `min_similarity` or outside the filter, sorts by similarity, keeps `k`, and when a reranker is configured scores
the survivors with it. Ranking is always by cosine similarity; the reranker score is reported for the caller to use.

Exact mode has no notion of "near": `Candidates` returns the identical value (after the store's normalization) at
similarity 1.0, or nothing. This keeps one code path for callers that run both modes.

The caller then applies its gate: constraint checks, side-effect class, schema version, tenant. Whatever it decides,
it can record.

## Value metadata

```cpp
store->Put("faq/42", answer, prestige::Metadata{{"family", "support-faq@v3"},
                                                {"tool", "resolver/1.4"},
                                                {"schema", "constraints@2"}});
prestige::Metadata m;
store->GetMetadata("faq/42", &m);
```

Metadata belongs to the **value**, not the key, because candidates are values. Writing the same bytes again with
metadata merges the maps, later pairs winning per key; a plain `Put` of the same bytes leaves existing metadata
alone. When the last key referencing a value is deleted, its metadata goes with it. Limits: 64 pairs, keys up to 256
bytes, values up to 4096 bytes. `PutWithDecision` has an overload that takes metadata too, so a write can carry both
its decision record and its metadata in one transaction.

Because merge is last-writer-wins per key, a field that records an origin (`first_seen`, `source`, the `actor` that
first wrote the value) survives only if nobody sends it again: write it once, with the first write, and leave it out
of every later map. A caller that resends its whole map on every write turns every origin field into a last-touch
field. The store does not know which keys are origins and has no keep-first mode.

Use an **explicit, versioned task-family id** as the identity (`"support-faq@v3"`), and carry tool and schema
versions as metadata. Do not derive the family from prompt text: the point of the metadata is that it is stable
when the text is not.

## Outcomes

```cpp
prestige::Outcome o;
o.family_id = "support-faq@v3";
o.verdict = prestige::OutcomeVerdict::kRejected;   // kAccepted | kRejected | kNoCandidate
o.candidate_object_id = out[0].object_id;
o.candidate_digest = out[0].digest;
o.rank = out[0].rank;
o.similarity = out[0].similarity;
o.threshold = 0.92f;
o.tool_version = "resolver/1.4";
o.schema_version = "constraints@2";
o.reason = "tenant mismatch";
o.decision_id = "dec-2026-10-13-0007";   // optional: the decision that wrote what was used instead
store->RecordOutcome(o);
```

The three verdicts:

| Verdict | Meaning |
|---|---|
| `kAccepted` | The candidate passed the caller's checks and was reused. |
| `kRejected` | The candidate was above the caller's threshold but failed a constraint check: a **false accept** of the similarity gate. |
| `kNoCandidate` | Nothing above the threshold; a fresh computation followed. |

`decision_id` is optional and ties a verdict to a write. When every candidate is rejected and a fresh value is
written with `PutWithDecision`, recording the rejections under that write's decision id turns "this is new" into a
dated claim with its own evidence: the decision record says what was written and when, and the outcomes say which
existing values were considered and why each was refused.

### Record the rejections, then write

```cpp
std::vector<prestige::Candidate> cands;
store->Candidates(query_text, q, &cands);
const std::string decision_id = NextDecisionId();          // the caller's own id scheme
for (const auto& c : cands) {
  prestige::Outcome o;
  o.family_id = "support-faq@v3";
  o.verdict = prestige::OutcomeVerdict::kRejected;         // every candidate failed the gate
  o.candidate_object_id = c.object_id;
  o.candidate_digest = c.digest;
  o.rank = c.rank;
  o.similarity = c.similarity;
  o.threshold = 0.92f;
  o.reason = "tenant mismatch";
  o.decision_id = decision_id;
  store->RecordOutcome(o);
}
prestige::Decision d;
d.decision_id = decision_id;
d.policy_revision = "policy@3f9c2a";
for (const auto& c : cands) d.input_digests.push_back(c.digest);   // what the decision looked at
store->PutWithDecision("faq/43", fresh_answer, d);
```

Outcomes and the write are separate transactions. Record the rejections first: a crash between the two leaves
rejections naming a decision the store never saw, which `GetDecision` reports as not found, and never a decision
record that claims rejections that were not stored.

Accepts are recorded as well as rejects because a false-accept **rate** needs a denominator. Outcomes are persisted
in their own column family, totally ordered by a sequence number, and counted through the metrics sink
(`prestige.outcome.recorded_total`, `prestige.outcome.false_accept_total`, `prestige.outcome.accepted_total`,
`prestige.outcome.no_candidate_total`).

## Per-family report

```cpp
prestige::FamilyReport r;
store->GetFamilyReport("support-faq@v3", &r);
// r.accepted, r.rejected, r.no_candidate, r.false_accept_rate
// r.accepted_similarity_hist, r.rejected_similarity_hist   (20 buckets over [0, 1])
// r.rejected_rank_hist                                      (ranks 0..14, then 15+)
// r.suggested_threshold                                     (advisory; -1 without enough evidence)
```

`suggested_threshold` is **advisory**: it is the lowest similarity bucket edge above which recorded false accepts are
at most 5% of judged candidates, computed only when at least 20 judged candidates sit above that edge and at least one
rejection has been recorded, and never below the lowest bucket in which a candidate was judged (a family gated at 0.90
whose judged candidates all sit above the gate is advised 0.90, not 0.00). Thresholds are never mutated by the store, and never per pair; the caller reads the report
per family and recalibrates. Automatic recalibration is deliberately left for a later release, once real outcome data
shows what the distributions look like.

`ListOutcomes(family, limit, after_sequence)` returns the raw records in write order and `ListFamilies()` lists
every family with outcomes. For reports computed from a real run, with the numbers explained, see
[outcomes-demo.md](outcomes-demo.md).

## Python

```python
import prestige

with prestige.open("./cache") as store:
    store.put("faq/42", answer, metadata={"family": "support-faq@v3", "tool": "resolver/1.4", "schema": "c@2"})

    cands = store.candidates(query, k=5, filter={"family": "support-faq@v3"}, min_similarity=0.85)
    best = cands[0] if cands else None
    if best and passes_constraints(best):           # the caller's gate
        store.record_outcome("support-faq@v3", "accepted", candidate=best, threshold=0.92)
    elif best:
        store.record_outcome("support-faq@v3", "rejected", candidate=best, threshold=0.92,
                             reason="tenant mismatch", decision_id="dec-0007")
        store.put("faq/43", fresh_answer,                         # the write the rejection informed
                  decision=prestige.Decision("dec-0007", "policy@3f9c2a",
                                             input_digests=[c["digest"] for c in cands]))
    else:
        store.record_outcome("support-faq@v3", "no_candidate")

    store.family_report("support-faq@v3")["false_accept_rate"]
    store.family_report("support-faq@v3")["suggested_threshold"]   # None until there is evidence
    store.list_outcomes("support-faq@v3", limit=100)
    store.list_families()
    store.get_metadata("faq/42")
```

`record_outcome` accepts a candidate dict straight from `candidates()`, which supplies the object id, digest, rank and
scores; any of them can be overridden by keyword. `decision_id` names the write the verdict informed, and comes back
in `list_outcomes` (empty for outcomes recorded without one).

## Storage

Two new column families: `prestige_value_meta` (object id to serialized metadata) and `prestige_outcomes`
(`o` + family + sequence for records, `f` + family for the registry, `n` for the sequence counter). Existing stores
gain both on first open. Outcome rows carry a format version; rows written before `decision_id` existed decode with
an empty one.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `prestige.candidates.calls` | counter | Candidates calls |
| `prestige.candidates.returned` | histogram | Candidates returned per call |
| `prestige.candidates.latency_us` | histogram | Candidates latency |
| `prestige.outcome.recorded_total` | counter | Outcomes recorded |
| `prestige.outcome.accepted_total`, `prestige.outcome.false_accept_total`, `prestige.outcome.no_candidate_total` | counter | Outcomes by verdict |

## Design notes

The interface follows a public exchange with eignex on Moltbook in October 2026: the store returns candidates with
metadata rather than certifying a hit, identity is an explicit versioned family id with tool and schema versions as
metadata, failures are recorded and aggregated per family, and thresholds are recalibrated from false-accept rates
rather than adjusted per pair. Recording accepts alongside rejects, so that the rate has a denominator, is this
implementation's addition. The same candidate list also serves the case where one store must answer two different
questions, "same claim" and "same source", with two different thresholds: both are decided by the caller against the
same ranked list.
