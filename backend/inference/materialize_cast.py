"""The activation cast, as a node of the graph the planner hands on.

A quantised matmul does not read the activation its edge carries: it reads a
narrow copy, and a kernel makes one. The frameworks launch that kernel —
vLLM's `QuantFP8` / `per_token_group_quant_fp8`
(`BlockScaledMMLinearKernel.py:119`, with the GEMM at `:132`), sglang's
`per_token_group_quant_fp8` (`fp8_utils.py:1355`) — so it is a launch, and a
launch is a node (`CLAUDE.md` §3).

Before this pass there was no node. The cost model carried the change as a
NUMBER on the consumer (`Traffic.convert_bytes`) and six plans declared
an output contract (`quantises_output`, since retired) to say a producer had
absorbed it, which made the
absorption a second mechanism beside the one fusion already is. The
consequences were everywhere a node is read: the inspector and the
computation graph drew one node where the hardware runs two, the per-module
breakdown attributed the cast's bytes to the GEMM, and the measured corpus
billed one row for both launches — 6,196 rows came out above the card's own
bandwidth and the SOL gate refused them.

WHERE IT RUNS, and why there. Between `materialize_comm` and
`Framework.rewrite` (`serving.rewrite_serving_graph`): after the collectives
because a cast sits on a dataflow edge that a collective may have moved, and
BEFORE fusion because fusion is what absorbs it. A plan whose naive pattern
names `quantise_activation` as a member swallows the cast into its launch —
an activation epilogue that writes fp8
(`silu_mul_per_token_group_quant_fp8_colmajor`), a norm that writes its
query quantised (`fused_q_kv_rmsnorm_quant`) — and one that does not leaves
it standing as its own node, priced by the same machinery as every other
node and measured as the `quantise_activation` cell. The plan takes it as an
optional sink (`silu_and_mul`: a BF16 deployment has no cast and the plan
matches without it) or a required one (`fused_q_kv_rmsnorm_quant`, whose
launch exists only on the quantised path), and the match's closure keeps a
cast out of a launch whose wide output something else still reads.

HOW IT IS FOUND (`docs/quantization.md`, "Activation casts in the serving graph"). Three steps, no case:

  1. Naive graft: every conversion the walk reports gets a cast on the
     consumer's own input edge.
  2. Canonicalise, to a fixed point: casts of one tensor to one format are
     one cast (CSE: one quantised copy per tensor and format, whoever reads
     it — DeepSeek-V3's shared and routed experts read one copy of the
     post-norm row; a framework that quantises it twice runs a launch the
     graph does not model); and a cast moves above an
     element copy that keeps the scale-group axis (`traffic.ElementCopy`:
     views, non-reducing collectives, whole-row gathers such as the MoE
     dispatch `permute`) when it is all that reads the copy's output
     (`traffic.TensorReaders`). The
     routed experts' cast so lands on the dispatch's token input, once per
     token, and the dispatch and its exchange move the narrow record.
  3. Fusion, after this pass, merges launches only.

WHERE IT STANDS. Right after the tensor's writer, in the writer's scope: a
fusion binds one scope, so only there can the writer's launch (an epilogue
that writes the format) take it in. The tree holds one copy of a block
template for all its layers, so a writer in another template (the residual
thread from the layer before) has no scope here, and a tensor no node writes
(a model input, a stream read by name, an isolated module's input) has no
writer at all: the copy
then stands in the lowest scope holding every reader, before the first node
there that holds one, and reads what its first reader's input was bound to
(`CastGraft.rewrite`). Standing in the writer's graph, it is in that graph's MoE
dispatch zone exactly when a node there reading the writer would be
(`spec_tree.build.reads_into_dispatch_zone`). Where the edge from it to a
reader enters a component, the component's own inputs name the tensors its
body reads: once no node inside reads the writer's tensor any more, its input
reads the copy; while something still does, it is handed the copy beside the
writer's tensor (`_enter_regions`). A grafted collective's payload width is
restated over the grafted graph (`exchange_widths`).

WHAT IT DOES NOT DECIDE. Which edges need a cast is not this pass's
judgement: `ResolvedPrecision.conversion` answers it from the deployment's
required format against the format the edge carries — the format a
quantised weight's matmul eats, or the one a reader of a quantised cached
tensor quantises the operand it multiplies that tensor with to
(`cache.paired_records`: the DSA indexer's query, against its FP8 or MXFP4
keys) — and `traffic.derive` already reports the answer per leaf
(`Traffic.cast`). This pass reads that and grafts; it has no opinion of its
own to drift from.

A STORED TRANSFORM, the rule beside the conversion. A cache format may
rotate the tensor before writing it (`CacheFormat.transform`,
`fp8_pow2_hadamard128`: GLM-5.3-Flash's index keys, which vLLM rotates
inside its pool+quant+store launch, kpool_compress.py:219/:582, and sglang
inside its store launches). A rotation is a launch of an op, so it is a node
(`CacheFormat.transform_op`), grafted twice:

  * on the edge INTO the cache, ahead of the leaf that lands the tensor
    (`cache.landed_by`), which must be a store marker (`kv_stream_marker`,
    an identity on the values) — the rotation is of what is stored;
  * on the OTHER operand of every product that reads the cached tensor: a
    cached rotated key scored against an
    unrotated query is a different number, and the stacks rotate the query
    into the cached basis (vLLM `fwht128_quant_fp8`, sglang
    `rotate_activation`). Which operand pairs with the cached one is the
    reader's declaration, not its name: the input whose declared innermost
    axis is the same contracted axis (in neither output) as the cached
    operand's.

A reader the cached tensor reaches any other way — elementwise, or
contracted along another axis — cannot run in a rotated basis, and the
format is refused for that model rather than priced as if it could. A
model with no transform format grafts nothing here. Which edges is the
walk's answer (`cache.transform_edges`), as which conversions is
(`ResolvedPrecision.conversion`); this pass grafts.

WHAT IT DOES NOT STATE. The grafted node's tensor and what moving it
costs are its readers': the cast copies the operand their kernels are
handed (every reader of one copy is handed the same tensor, or the graft is
refused). How many times a step runs it is its writer's: one copy per
tensor written, however many readers take it, read off the writer it
stands after (`CastGraft.runs_with`, or its readers where it stands before
them). `cast_facts` reads these off the layers serving merges, rather than
this pass publishing a second answer that could drift from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from backend.arch.cache_formats import CacheFormat
from backend.arch.eval import cache as cache_mod
from backend.arch.eval import traffic as traffic_layer
from backend.arch.eval.flops import record_binding
from backend.arch.eval.layer import Layer, Rewrite
from backend.arch.eval.program import Program
from backend.arch.eval.traffic import (ElementCopy, TensorReaders,
                                       payload_width)
from backend.arch.eval.values import Cost, Shape
from backend.arch.layout.materialize_comm import (FlowRef, PayloadRole,
                                                  refs_reading,
                                                  rewrite_refs)
from backend.arch.spec_tree.build import reads_into_dispatch_zone
from backend.arch.spec_tree.nodes import InputRef

# The conversion op this pass grafts.
CAST_OP = "quantise_activation"

#: Every op this pass can graft: the conversion, and each stored transform's
#: op (`CacheFormat.TRANSFORMS`, their one definition). `ModelFootprint`
#: reads it: a deployment can add these leaves after the spec walk, so a
#: second list of them would drift from the pass that materialises them.
GRAFTED_OPS = frozenset(
    {CAST_OP} | {op for op in CacheFormat.TRANSFORMS.values() if op})


@dataclass(frozen=True)
class CastSite:
    """One grafted node — a cast, or a stored transform's op: its node, the
    leaf it stands before, and which of that leaf's inputs it stands on
    (`Shape.ins` position)."""

    node: Any
    #: Uid of the leaf whose input the cast stands on.
    consumer: int
    #: Which of that leaf's inputs.
    index: int


@dataclass(frozen=True)
class CastGraft:
    """The pre-fusion program with its casts materialised."""

    program: Any
    #: The grafted sites, in insertion order.
    sites: Tuple[CastSite, ...]
    #: Each grafted node's uid -> the uids whose step count it runs at, as
    #: placed (`_Pending.home`): the writer it stands after — once per
    #: tensor written, however many readers take the copy — or, where it
    #: stands before its first reader (the writer is in another template),
    #: its readers, which must then agree.
    runs_with: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    #: What each grafted node reads where no node of the graph writes its
    #: tensor (a model input): the input of its first reader it took over,
    #: `(node uid, 0) -> (reader uid, input)` — the walk bound that one
    #: (`cache.Edges.reads`), and nothing bound the graft's.
    rewrite: Rewrite = field(default_factory=Rewrite)

    def __bool__(self) -> bool:
        return bool(self.sites)

    @property
    def nodes(self) -> Tuple[Any, ...]:
        """The grafted nodes, each once, in insertion order. A node stands
        on several sites when readers share the copy (a gated FFN's two
        projections), so a node is not a site."""
        return tuple({site.node.uid: site.node for site in self.sites}.values())


@dataclass(frozen=True)
class CastFacts:
    """A graft's nodes stated on the layers serving merges, per rank and
    global: a cast moves one tensor and computes two passes over it, stores
    nothing, and runs when its consumer runs."""

    local_cost: Layer[Any]
    local_shapes: Layer[Any]
    full_cost: Layer[Any]
    full_shapes: Layer[Any]
    instances: Layer[int]
    #: The substitution records the per-rank FLOPs figures were priced from
    #: (`cost_plan.flops_bindings` for a walked leaf), so a row's formula
    #: values line binds the cast's own axes rather than folding to a bare
    #: number.
    flops_bindings: Layer[Any]


def _folded_dims(out: Tuple[Any, ...]) -> Dict[str, Any]:
    """The cast's own dims from the tensor it casts.

    Elementwise, so the leading axes fold into one: `[B, s, d]` and
    `[B, h, s, d]` are the same work over the same elements, and the op's
    own `flops_fn` reads `B`, `s_ctx` and `dim`. `dim` is the innermost
    extent because that is the axis a per-group scale lies along.
    """
    if not out:
        return {"B": 1, "s_ctx": 1, "dim": 1}
    rows: Any = 1
    for extent in out[:-1]:
        rows = rows * extent
    return {"B": 1, "s_ctx": rows, "dim": out[-1]}


def _name(reader: str, index: int, what: str = "cast") -> str:
    """A name that says what the node does for whom (`reader`, its name),
    and on which operand."""
    base = f"{reader}__{what}"
    return base if index == 0 else f"{base}{index}"


def _root_siblings(tree: Any, node: Any) -> list:
    """The root list holding a parentless node: the model's top level, or
    its block template's roots."""
    candidates = [tree.top_level, *(roots for _meta, roots in tree.blocks)]
    for siblings in candidates:
        if any(candidate is node for candidate in siblings):
            return siblings
    raise ValueError(
        f"{node.display_path}: a root consumer is attached to no root "
        "list of the tree")


def _after(siblings: list, home: Any, *, before: bool) -> Any:
    """The sibling a cast is inserted in front of: `home` itself when the
    cast stands before the consumer it is made for (`before`), else the one
    after `home`, the producer it re-encodes — None at the end of the list."""
    if before:
        return home
    index = next(i for i, sibling in enumerate(siblings) if sibling is home)
    return siblings[index + 1] if index + 1 < len(siblings) else None


def _enter_regions(program: Any, cast: Any, consumers: Tuple[Any, ...]
                   ) -> None:
    """Carry the edge from `cast` into the components it enters.

    The consumers' own inputs already read the cast. Between the cast's
    scope and a consumer the edge may cross component boundaries (the
    expert bank reading `permute`, an attention block reading its norm):
    each is a region, and its entry lists every tensor that crosses into
    it. Walking outermost first, a region no node inside still reads the
    wide tensor from is the region the copy now feeds: its entry and every
    nested entry are rewritten to the cast, as the comm graft rewrites a
    region (`refs_reading` / `rewrite_refs`, CLAUDE.md 2.8), and nothing
    nested needs more. A region whose body reads both (DeepSeek-V3's MoE
    block: its router and `permute` read the norm, its shared expert the
    copy) is handed both: its entry keeps the wide tensor, and the copy
    crosses in beside it as one more input of the call (`_hand_in`)."""
    semantic = FlowRef.read_by(cast.inputs_obj[0])
    for consumer in consumers:
        chain = []
        ancestor = consumer.parent
        while ancestor is not None and ancestor is not cast.parent:
            chain.append(ancestor)
            ancestor = ancestor.parent
        for region in reversed(chain):              # outermost first
            targets = refs_reading(region, semantic.node_uid, semantic.port,
                                   semantic.external)
            if targets and all(program.by_uid(t.node_uid).children
                               for t in targets):
                rewrite_refs(program.by_uid, targets, semantic, cast,
                             f"cast {cast.display_path}")
                break
            _hand_in(region, cast)


def _hand_in(region: Any, cast: Any) -> None:
    """List `cast` among the tensors `region`'s call hands its body, unless
    it already is: one more input, bound to a formal of the cast's own name
    so the call's inputs stay one to one with its formals (the frame the
    walk binds a body by, `Evaluator._callee_aliases`). Its body's readers already
    name the cast; the entry states that the copy crosses the boundary."""
    refs = tuple(region.inputs_obj or ())
    if any(ref.src is cast for ref in refs):
        return
    if cast.name in region.formals:
        raise ValueError(
            f"{region.display_path}: a formal is already named "
            f"{cast.name!r}, the cast it would be handed")
    region.inputs_obj = (*refs, InputRef(src=cast, port=""))
    region.formals = (*region.formals, cast.name)


def _shape_and_cost(node: Any, out: Tuple[Any, ...]) -> Tuple[Any, Any]:
    """One cast node's shape and cost over a tensor of dims `out`, from the
    op's own declaration — the one statement of them, per rank and global
    alike."""

    out = tuple(out)
    return (Shape(ins=(out,), out=out),
            Cost(flops=node.declaration.flops_fn(_folded_dims(out))))


def cast_facts(graft: CastGraft, *, local_shapes: Any, full_shapes: Any,
               instances: Any) -> CastFacts:
    """State a graft's nodes on the layers serving merges them into.

    The cast's tensor is the operand its readers' kernels are handed
    (`Shape.ins` at the input the cast stands on, the same for every reader
    of one copy), per rank and global, and its count is where the graft placed it
    (`CastGraft.runs_with`): the writer it stands after, already recounted
    for the deployment's stage selection, so a cast whose writer this stage
    does not run is counted zero with it.
    """

    local_costs: Dict[int, Any] = {}
    local_shape_map: Dict[int, Any] = {}
    full_costs: Dict[int, Any] = {}
    full_shape_map: Dict[int, Any] = {}
    counts: Dict[int, int] = {}
    bindings: Dict[int, Any] = {}
    for site in graft.sites:
        local = local_shapes.values[site.consumer].ins[site.index]
        full = full_shapes.values[site.consumer].ins[site.index]
        uid = site.node.uid
        stated = (local_shape_map.get(uid), full_shape_map.get(uid))
        local_shape_map[uid], local_costs[uid] = (
            _shape_and_cost(site.node, local))
        full_shape_map[uid], full_costs[uid] = (
            _shape_and_cost(site.node, full))
        if stated[0] is not None and stated != (local_shape_map[uid],
                                                full_shape_map[uid]):
            # One copy read by several consumers (CSE) is one tensor: its
            # readers must be handed the same one.
            raise ValueError(
                f"{site.node.display_path}: its readers disagree on the "
                f"tensor they read it as: {stated} vs "
                f"{(local_shape_map[uid], full_shape_map[uid])}")
        if uid not in counts:
            runs = {instances.values[u] for u in graft.runs_with[uid]}
            if len(runs) != 1:
                raise ValueError(
                    f"{site.node.display_path}: it stands before readers "
                    f"that run a different number of times per step "
                    f"({sorted(runs)}), so no one count is the copy's")
            counts[uid] = runs.pop()
        record: Dict[Any, Any] = {}
        for axis in local:
            record_binding(record, axis)
        if record:
            bindings[site.node.uid] = record
    program = graft.program
    return CastFacts(
        local_cost=Layer(program, local_costs),
        local_shapes=Layer(program, local_shape_map),
        full_cost=Layer(program, full_costs),
        full_shapes=Layer(program, full_shape_map),
        instances=Layer(program, counts),
        flops_bindings=Layer(program, bindings),
    )


def _insert(tree: Any, node: Any, home: Any, *, before: bool) -> None:
    """Put a grafted `node` in `home`'s sibling list: in front of `home`
    (`before`, the node stands for the consumer it is made for) or right
    after it (the producer it re-encodes)."""
    parent = home.parent
    if parent is not None:
        tree.insert_child(parent, node,
                          before=_after(parent.children, home, before=before))
        return
    # A root consumer — a model-level entry like the MTP head's
    # projections — has no parent whose children list holds it; its
    # siblings are a root list of the tree itself (the same reach
    # `materialize_comm._siblings_of` makes for a root collective's anchor).
    siblings = _root_siblings(tree, home)
    node.parent = None
    following = _after(siblings, home, before=before)
    siblings.insert(len(siblings) if following is None
                    else siblings.index(following), node)


def _make(tree: Any, *, name: str, op: str, home: Any, producer: Any,
          port: str, out: Tuple[Any, ...], output_format: Any = None,
          output_dtype: Any = None, external: str = "") -> Any:
    return tree.make_node(
        name=name,
        op=op,
        is_component=False,
        from_custom_op=False,
        role=home.role,
        template=home.template,
        component_path=home.component_path,
        dims=_folded_dims(out),
        inputs=(),
        inputs_obj=(InputRef(src=producer, port=port,
                             external_sid="" if producer is not None
                             else external),),
        children=[],
        provenance="cast",
        output_format=output_format,
        output_dtype=output_dtype,
        # It stands in the producer's graph when `home` is the producer,
        # and is in that graph's dispatch zone by the rule every node there
        # is (`reads_into_dispatch_zone`); standing in a consumer's template
        # it reads across the template boundary, where no zone reaches.
        in_dispatch_zone=(home is producer
                          and reads_into_dispatch_zone(producer)),
    )


def _port_of(ref: Any) -> str:
    """The producer port an input ref reads, `""` for the primary."""
    return "" if ref.port in (None, "out") else ref.port


@dataclass
class _Pending:
    """A cast the graft will make, while it is canonicalised: the tensor it
    copies (`source` on `port`; None for a tensor no node in the graph
    writes but named on the edge — a model input, stream, or auxiliary
    input), the format it writes, and
    the leaf inputs that read the copy. A stored transform (a rotation,
    `op`) is grafted by the same rule and writes no format."""

    source: Any
    port: str
    target: Any
    target_dtype: Any
    readers: list
    #: The GEMM input it was first grafted for, `(leaf, index)`: what it is
    #: named for, wherever canonicalisation moves it.
    origin: Tuple[Any, int]
    op: str = CAST_OP
    #: The input's name where no node writes it (`InputRef.external_sid`).
    external: str = ""

    @property
    def key(self) -> Tuple[Any, ...]:
        """Its identity under CSE: one tensor (its writer's output, or the
        name of an input no node writes), one format, in the template its
        readers are in. Every reader in that template that
        wants the tensor in that format reads the one copy — a gated FFN's
        two projections, the shared expert and the routed experts of one
        MoE layer, an attention's `wq_b` and its indexer's. A framework that
        quantises one tensor twice (vLLM's shared-expert linear beside its
        MoE prepare step) runs a launch this graph does not bill: the
        graph models the better form, not that inefficiency."""
        tensor = self.external if self.source is None else self.source.uid
        # The template the copy stands in, its readers': the tree holds one
        # copy of a block template for all its layers, so a node in one
        # template is no node another template's readers can read.
        return (tensor, self.port, self.op, self.target, self.target_dtype,
                self.readers[0][0].template)

    def commute(self, shapes: Any, readers: TensorReaders) -> bool:
        """`cast(X(y)) -> X(cast(y))`: move above the copy X that writes
        the tensor while X keeps the scale-group axis (`ElementCopy.
        commutes`, read off its declarations), this copy is all that reads
        X's output (`TensorReaders.only`) — onto X's input, written by a
        node or by none (a named input no node writes). Quantising rows before they are copied is quantising the
        copies — once per row rather than once per copy, and a collective
        after the copy moves the narrow record. Whether it moved."""

        moved = False
        while self.source is not None and not self.port:
            node = self.source
            copy = ElementCopy.of(node, shapes)
            if copy is None or not copy.commutes:
                break
            (index,) = copy.inputs
            ref = tuple(node.inputs_obj or ())[index]
            if not readers.only(node, "", self.readers):
                break
            self.source, self.port = ref.src, _port_of(ref)
            self.external = "" if ref.src is not None else ref.external_sid
            self.readers = [(node, index)]
            moved = True
        return moved

    @property
    def after_writer(self) -> bool:
        """Whether it stands right after its writer (`home`): a node of the
        graph writes the tensor, in its readers' template."""
        return (self.source is not None
                and self.source.template == self.readers[0][0].template)

    def home(self, tree: Any) -> Any:
        """The node it stands beside (`graft`): its writer; or, where no
        writer shares its readers' template — the writer in another one, or
        none in the graph (a model input) — the node it stands before in
        the lowest scope holding every reader: the first, in that scope's
        order, of the nodes there that hold one. Every reader's scope
        reaches it, and it runs before each."""
        if self.after_writer:
            return self.source
        chains = [_chain(reader) for reader, _index in self.readers]
        depth = 0
        while (depth < min(len(chain) for chain in chains) - 1
               and all(chain[depth] is chains[0][depth] for chain in chains)):
            depth += 1
        holding = {id(chain[depth]) for chain in chains}
        scope = chains[0][depth].parent
        siblings = (scope.children if scope is not None
                    else _root_siblings(tree, chains[0][depth]))
        return next(node for node in siblings if id(node) in holding)

    def runs_with(self, program: Any,
                  grafted: Mapping[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """The uids whose step count it runs at (`CastGraft.runs_with`):
        the leaf that writes its tensor — a node grafted before it (a
        rotation the cast reads) running at what that one runs at — or,
        standing before its readers, those readers."""
        if not self.after_writer:
            return tuple(reader.uid for reader, _index in self.readers)
        if self.source.uid in grafted:
            return grafted[self.source.uid]
        if self.source.children:
            return (program.output_leaf(self.source,
                                        self.port or "out")[0].uid,)
        return (self.source.uid,)

    def graft(self, tree: Any, shape: Tuple[Any, ...]) -> Any:
        """Make the node and point every reader at it. It stands right
        after the tensor's writer, in the writer's scope, where the
        writer's launch can take it in (an epilogue that writes the
        format). The IR holds one tree per block template, every node of
        which exists in each of its layers, so a writer in another template
        (the residual thread from the layer before), or none in the graph (a
        model input, an isolated module's input), has no scope here to
        share: the copy then stands in the lowest scope holding every
        reader (`home`)."""
        home = self.home(tree)
        made_for, index = self.origin
        what = "cast" if self.op == CAST_OP else self.op.split("_", 1)[0]
        name = _name(made_for.name, index, what)
        siblings = (home.parent.children if home.parent is not None
                    else _root_siblings(tree, home))
        if any(sibling.name == name for sibling in siblings):
            # Another node of the same name in the scope it stands in: named
            # by the path from this scope down to the leaf it is made for.
            path = [made_for.name]
            ancestor = made_for.parent
            while ancestor is not None and ancestor is not home.parent:
                path.append(ancestor.name)
                ancestor = ancestor.parent
            name = _name(".".join(reversed(path)), index, what)
        node = _make(tree, name=name, op=self.op, home=home,
                     producer=self.source, port=self.port, out=shape,
                     external=self.external,
                     output_format=self.target,
                     output_dtype=(None if self.target is not None
                                   or self.target_dtype is None
                                   else {"": self.target_dtype}))
        _insert(tree, node, home, before=home is not self.source)
        for reader, at in self.readers:
            refs = list(reader.inputs_obj or ())
            if (refs[at].src is not self.source or (
                    self.source is None
                    and refs[at].external_sid != self.external)):
                raise ValueError(
                    f"{reader.display_path}: input {at} does not read the "
                    f"edge the cast is on")
            refs[at] = InputRef(src=node, port="")
            reader.inputs_obj = tuple(refs)
        return node


def _chain(node: Any) -> list:
    """`node` and its ancestors, outermost first."""
    chain = [node]
    while chain[0].parent is not None:
        chain.insert(0, chain[0].parent)
    return chain


def _merged(pending: list) -> list:
    """Common-subexpression elimination over the pending casts: those with
    one key (`_Pending.key`) are one cast, read by all their readers."""
    by_key: Dict[Any, _Pending] = {}
    for cast in pending:
        kept = by_key.get(cast.key)
        if kept is None:
            by_key[cast.key] = cast
            continue
        kept.readers.extend(r for r in cast.readers if r not in kept.readers)
    return list(by_key.values())


def materialize_casts(analysis: Any, materialized: Any) -> CastGraft:
    """Graft the `quantise_activation` nodes for the casts no producer
    absorbed — one per tensor and target format, every consumer converting
    that tensor to that format reading the one copy — and each stored
    transform's op on the edges its format puts it on
    (`cache.transform_edges`).

    `materialized` is the comm materialisation whose tree is already this
    deployment's private copy, so the graft lands in it rather than copying
    the program a second time. The program handed on is rebuilt over the
    grafted tree: a Program's indexes are computed on first read, and the
    derivation below has already read the ones the graft changes
    (`by_uid`, `stack_of`) — every later reader must compute them with the
    casts in.
    """

    tree = materialized.tree
    program = materialized.program
    flows = traffic_layer.derive(analysis, program=program)

    sites: list = []
    nodes: list = []

    # The stored transforms first: a rotation comes before the quantiser
    # that writes the rotated values narrow (both stacks rotate, then
    # quantise), so a cast on the same operand reads the rotation.
    runs_with: Dict[int, Tuple[int, ...]] = {}
    unbound: Dict[Tuple[int, int], Tuple[int, int]] = {}
    transforms: list = []
    for (uid, index), op in sorted(
            cache_mod.transform_edges(analysis).items()):
        consumer = program.by_uid(uid)
        reading = tuple(consumer.inputs_obj or ())[index]
        if reading.src is None:
            raise ValueError(
                f"{consumer.display_path}: input {index} names no producer "
                f"to rotate")
        transforms.append(_Pending(
            source=reading.src, port=_port_of(reading), target=None,
            target_dtype=None, readers=[(consumer, index)],
            origin=(consumer, index), op=op))
    shared_transforms = []
    for transform in _merged(transforms):
        reader, index = transform.readers[0]
        node = transform.graft(
            tree, tuple(analysis.shapes.values[reader.uid].ins[index]))
        runs_with[node.uid] = transform.runs_with(program, runs_with)
        shared_transforms.append(node)
        for reader, index in transform.readers:
            sites.append(CastSite(node=node, consumer=reader.uid,
                                  index=index))

    # 1. Naive graft: one pending cast per conversion, on the consumer's
    # own input edge. Sorted so the graft is the same tree whichever order a mapping iterates in: two processes
    # building the same deployment must produce the same uids, or a cached
    # payload from one does not describe the other.
    shapes = Layer(program, {**analysis.shapes.values,
                             **materialized.shapes.values})
    pending: list = []
    for uid in sorted(flows.values):
        cast = flows.values[uid].cast
        if cast is None:
            continue
        consumer = program.by_uid(uid)
        refs = tuple(consumer.inputs_obj or ())
        if not 0 <= cast.index < len(refs):
            raise ValueError(
                f"{consumer.display_path}: the cast names input "
                f"{cast.index}, which is outside its {len(refs)} inputs")
        reading = refs[cast.index]
        if reading.src is None and not reading.external_sid:
            raise ValueError(
                f"{consumer.display_path}: input {cast.index} has neither a "
                f"producer nor a name, so no copy of it can be placed or "
                f"shared")
        # An edge no node in the graph writes (a named model, stream, or
        # auxiliary input) is copied by a launch of its own all the same:
        # it stands in its readers' scope (`_Pending.home`).
        pending.append(_Pending(
            source=reading.src, port=_port_of(reading),
            target=cast.target, target_dtype=cast.target_dtype,
            readers=[(consumer, cast.index)],
            origin=(consumer, cast.index),
            external="" if reading.src is not None
            else reading.external_sid))

    # 2. Canonicalise to a fixed point: merge identical casts (CSE), then
    # move each above an element copy it commutes with. A tensor read past
    # the graph's edges (`TensorReaders`: cached, context-carrying, a stream
    # a later layer reads, a block's or a model-level output) is pinned
    # wide: a cast does not move above a copy that writes one.
    readers = TensorReaders.of(analysis, program)
    while True:
        pending = _merged(pending)
        moved = False
        for cast in pending:
            moved = cast.commute(shapes, readers) or moved
        if not moved:
            break

    for cast in pending:
        reader, index = cast.readers[0]
        node = cast.graft(tree, tuple(shapes.values[reader.uid].ins[index]))
        runs_with[node.uid] = cast.runs_with(program, runs_with)
        if cast.source is None:
            unbound[(node.uid, 0)] = (reader.uid, index)
        nodes.append(node)
        for reader, index in cast.readers:
            sites.append(CastSite(node=node, consumer=reader.uid,
                                  index=index))

    if not sites:
        return CastGraft(program=program, sites=())
    consumers_of: Dict[int, list] = {}
    for site in sites:
        consumers_of.setdefault(site.node.uid, []).append(
            program.by_uid(site.consumer))
    for node in (*shared_transforms, *nodes):
        _enter_regions(program, node, tuple(consumers_of[node.uid]))
    return CastGraft(
        program=Program(resolved=program.resolved, ir=tree,
                        schedule=program.schedule),
        sites=tuple(sites), runs_with=runs_with,
        rewrite=Rewrite(inputs=unbound))


def exchange_widths(analysis: Any, graft: CastGraft,
                    materialized: Any) -> Dict[int, float]:
    """The element width each grafted collective moves its payload at, read
    again over the grafted graph: a collective moves the one tensor it
    reads, and the graft may have changed what that tensor is — the
    exchange after a MoE dispatch whose tokens were quantised before it now
    moves the narrow elements and their scales, as vLLM dispatches `a1q`
    with its scales (`naive_dp_ep.py:181`). The rule the collective's width
    was chosen by when it was grafted (`traffic.payload_width`), over the
    graph as it now stands; a collective it answers nothing for keeps the
    width it was grafted with."""

    if not graft:
        return {}
    program = graft.program
    shapes = dict(analysis.shapes.values)
    shapes.update(materialized.shapes.values)
    for site in graft.sites:
        shapes[site.node.uid] = _shape_and_cost(
            site.node, shapes[site.consumer].ins[site.index])[0]
    layer = Layer(program, shapes)
    out: Dict[int, float] = {}
    for event in materialized.events:
        if event.payload_role != PayloadRole.FLOW_BUFFER:
            # An auxiliary buffer (DCP's LSE statistic) is no copy of the
            # tensor its node's edge carries; its width is its own.
            continue
        uid = materialized.node_uid_by_event[event.event_id]
        node = program.by_uid(uid)
        if node is None:
            continue
        width = payload_width(node, program, layer, event.payload_shape)
        if width is not None:
            out[uid] = width
    return out
