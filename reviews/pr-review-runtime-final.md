# Final runtime hardware classification review

Scope: read-only review of the revised SM-count classification, consumers of
CardClass, artifact identity/serialization, and scheduler memo tracing. No tests
were rerun in this review; parent reports fusion/GPU tests passed separately.

Result: no concrete correctness issue found in the reviewed changes.

## Hardware facts and walk reuse

- ServingDeployment.RUNTIME_CARD_FIELDS declares sm_count independently of
  dispatch predicate vocabulary. _card_walk_columns includes those declared
  fields; _check_card_fact_classification checks the GPU-sheet field and
  ServingDeployment.CARD_FIELDS alignment.
- ServingDeployment.at obtains SM count from GPUSpec, and ServingDeployment.of
  rejects nonpositive/noninteger/boolean counts. Unknown remains None. The
  materializer consumes that deployment fact; it does not substitute B300's
  count for unknown hardware.
- CardClass carries a sorted tuple of runtime facts in addition to dispatch and
  dtype peaks. NamedTuple equality/hash consequently distinguish identical
  compute capability and peak availability with different SM counts.
- CardClass.key includes those same named runtime facts. Positive SM counts
  cannot alias the key's None spelling. Removing unused when.sm_count does not
  remove SM count from structural identity.
- Card.walks_alike compares complete CardClass values; restamping refuses a
  different class. ServingWalkKey.of_card persists CardClass.key, and
  ServingProjection compares that same key before projection. Existing artifact
  keys lacking the runtime fact are different keys, not silently compatible.
- Repository searches found no production CardClass two-field tuple unpacking,
  positional decoder, or alternative serialization dropping runtime. Current
  production construction is centralized in card_class; consumers use complete
  equality/hash or key().

## Memo tracing

Both windowed_work and windowed_counts now use backend.code_trace.memo.memoize.
The helper records the wrapped computation's miss trace and merges it into an
active recorder on a cache hit. This covers the symbolic-work construction memo
and numeric periodic-scheduler memo without changing their cache inputs.
The exact/float/vector compiler mappings all call sparse_window, which delegates
to the same trace-aware windowed_counts wrapper. No functools memo remains in
these runtime modules. This is a static verification of the intended trace
path, not a claim that an additional dynamic trace audit was executed here.

Previously verified runtime limits remain explicit: source-selected owner
scenario, flattened verify rows, no guessed SM count, and a ValueError outside
SGLang's 48 KiB windowed-precompute admission instead of pretending the native
scheduler fallback follows the same schedule.
