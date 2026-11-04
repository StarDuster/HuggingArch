"""The evaluator: the spec graph walked into derived facts, layered.

The retired evaluator walked a spec graph into one mutable result tree: a
class carrying twenty-nine fields that span seven unrelated concerns, produced
by a walker whose per-frame state lives in mutable attributes on `self`. This
package is that machinery re-derived from its actual structure:

    node         what a node must provide, as two protocols
    Program      the immutable spec — components, ops, structural queries
    Binding      the dim environment, as a chained immutable scope
    Scope        one frame of the walk, threaded as a parameter
    Evaluator    the walk itself, with no mutable state at all
    Layer[T]     every derived fact, keyed on node identity
    transforms   Spec → Spec, Graph → Graph, Binding → Binding

The first is the newest and the reason is the same as the rest: the layering
above was real but unwritten. Every function took `node: Any` and reached for
fields with `getattr(x, "…", None)`, so the contract existed only as the union
of a hundred defensive accesses — one of which named a field that has never
existed on the type it was reading.

The swap this package was built for has happened: `engine` is deleted, and
every production number — flops, params, shapes, HBM access, KV storage and
reads, weight storage, the formula payloads — comes from the walk above.
Validation and rendering may project read-only reports from those layers;
inference never consumes such a report as a second evaluation tree.

(`schema` and `spec_tree` are not debts: they are the authored input this
package consumes, and survived the swap as exactly that. Dim expressions
evaluate at `eval.dim.eval_dim_expr`; `loader` uses that name for its own
resolution and does not re-export it. `transforms`' two `loader` imports
are the `ResolvedSpec` ingestion path, input-side like `spec_tree`.
`materialize` is gone: bytes come from `Analysis.tensors()` /
`iter_checkpoint()`. `Extent` is `Dim`.)
"""
