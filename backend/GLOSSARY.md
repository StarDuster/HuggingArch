# HuggingArch — Concept glossary

Defines only the **irreducible domain nouns** — the handful of concepts a reader
cannot infer from a well-named symbol. Operations (fuse, gather, collect, set up,
replace) are named with plain verbs in the code and are deliberately **not** listed
here: inventing a term for an ordinary verb is exactly the bloat this file exists
to avoid.

Companion to [`arch/dims.yaml`](arch/dims.yaml) (dimension
symbols) and [`arch/DESIGN.md`](arch/DESIGN.md) (the spec model —
primitive / component / adapter / model tiers, roles, alias adapters, and the
stream glossary for cross-layer dataflow). Anchors
are `file · symbol`, not line numbers.

---

## 1. The three identities of a leaf

Every leaf carries three **distinct** ids; conflating them is a recurring bug.

| id | identifies | anchor |
|----|----|----|
| `spec_id` | structural identity in the spec IR — stable across compose/expand; the key the engine's evaluation table and the fusion matcher use | `arch/spec_id_compose.py` |
| `aggregation_id` | the bucket a per-leaf cost rolls up to (many spec_ids → one) — `= compose(template, display_path)` | `arch/spec_id_compose.py · compose_aggregation_id` |
| `display_path` | the human dotted path (`model.layers.0.self_attn.q_proj`); presentation only, never a logic key | idem |

## 2. Device-mesh layout

How parallel axes nest on the physical grid — the cost model's core. `region` /
`base_stride` are defined once in `arch/layout/mesh.py`; `fabric` / `inner_group`
live in `inference/comm_cost.py`; `world` and `mesh` are used in their ordinary sense.

| term | meaning |
|----|----|
| `region` | a **per-component sub-mesh**. TP may differ per component (`tp_attention ≠ tp_dense_ffn ≠ tp_moe_expert`), so the layout is a *set* of sub-meshes, not one uniform grid |
| `base_stride` | rank spacing between consecutive members of a collective's group `= inner_group` size. TP is innermost → `1`; a CP collective over TP → `tp_attention` |
| `inner_group` | the group **nested inside** a collective (∏ of the axis sizes inner to it); its size is `base_stride` |
| `fabric` | which interconnect a collective crosses — `intra` (one NVLink/PCIe node) vs `inter` (cross-node RDMA); chosen from `base_stride` + topology, and it picks which bandwidth the roofline divides by. Not `lane`: in GPU code that is a thread inside a warp |

## 3. Attention category — one axis, one canonical field

The analysis keys on *which attention mechanism*, historically fragmented across
six spellings. Canonical going forward — nothing should introduce a seventh:

- **`cache.family`** (read back as `kv_label`) — the **cost / KV** category (`MLA / GQA / MHA / MQA / SWA /
  GLA-Linear`); drives per-token KV bytes, the KV shard divisor, and
  attention-component matching.
- **`attn_display`** — the **display** name the UI reads (`= attn_variant or cache.family`).

`attn_type` / `attention_type` / `attention_kind` are legacy aliases of the above;
read `cache.family` for anything cost-bearing and `attn_display` for anything
user-facing. `attn_variant` is NOT legacy — it is the authored display-override
field and the preferred source of `attn_display`.

## 4. The spec discriminator

Full mechanics live in `arch/DESIGN.md`; pinned here only because it is easy
to misread.

- **`role`** — a component's structural function. Component-declared roles are
  a closed set of six: `attention / ffn_dense / ffn_moe / router / block /
  aux_block` (`BlockGraph.role`); slot names in a model's `components:` map and
  call-site overrides may additionally be `attention_swa` / `norm`
  (`GraphNode.role`). Every name is classified once, in
  [`arch/roles.yaml`](arch/roles.yaml). The one rigorous
  categorization discriminator (prefer it over ad-hoc `*_type` / `*_kind`).
