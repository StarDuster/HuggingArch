import { useState } from "react";
import { fetchJson } from "../api";
import { precisionBody } from "../precision";
import type { GraphPhase, GraphTarget } from "./ComputeGraph";

/** The answer of `/inference/bench/plan-jobs`: the jobs.jsonl text the
 *  harness's own writer produced, and the ops of this step that carry no
 *  kernel signature. */
type PlanJobs = {
  jsonl: string;
  job_count: number;
  dropped: string[];
  filename: string;
};

/** The request body for `/inference/bench/plan-jobs` — the step on screen, as
 *  the estimate reported it.
 *
 *  Every field is an input of `kernel_bench.plan.build_jobs_over_points` under
 *  the name that builder reads it by; the numbers are the estimate's own
 *  (`graphWorkload`), which is why a job exported here is keyed at the shapes
 *  the per-op table showed. Same four workload fields `computeGraphQuery`
 *  sends, so the graph drawn and the jobs exported describe one step.
 */
export function benchJobsBody(target: GraphTarget, phase: GraphPhase): Record<string, unknown> {
  const { par, workload: w } = target;
  const body: Record<string, unknown> = {
    model_id: target.modelId.trim(),
    // The estimate's card. Framework dispatch can make a mixed-precision key
    // card-dependent, so a queue plan is never card-free.
    gpu_sku: target.gpuSku,
    phase,
    // The deployment whole: the mesh the user chose, its CP scheme, and the
    // routing skew the estimate priced the expert cells at. Not a flattened
    // TP degree — the per-rank shape of every job comes from all of it.
    parallel: {
      ...par,
      ...(target.cpScheme ? { cp_scheme: target.cpScheme } : {}),
      ...(typeof target.epLoadFactor === "number" && Number.isFinite(target.epLoadFactor)
        ? { ep_load_factor: target.epLoadFactor }
        : {}),
    },
    ...precisionBody(target.precision),
  };
  if (target.framework) body.framework = target.framework;
  if (phase === "encode") {
    // The encode step is the media's, with no language-model workload.
    if (!target.encode) throw new Error("an encode export needs the encoder's step");
    body.media = target.encode.media;
    body.batch_size = target.encode.batch;
    return body;
  }
  for (const [k, v] of Object.entries({
    avg_input_tokens: w.avgInputTokens,
    avg_output_tokens: w.avgOutputTokens,
    prefix_cached_tokens: w.prefixCachedTokens,
    speculative_draft_tokens: w.speculativeDraftTokens,
    batch_size: w.batch[phase],
  })) if (typeof v === "number" && Number.isFinite(v)) body[k] = Math.round(v);
  return body;
}

/** One op this export does not carry, named as the planner named it. The
 *  planner's note is `<op> [<node>]: <why>` for a drop it can explain, and a
 *  bare op name for one that simply has no signature; the op is what a reader
 *  needs first. */
function droppedOp(note: string): string {
  const head = note.split(":")[0] ?? note;
  return (head.split("[")[0] ?? head).trim() || note;
}

/** Download the step on screen as Kernel Bench jobs — the same `jobs.jsonl`
 *  `hbench plan` writes, so the shapes just planned can be measured on a GPU
 *  box without re-typing them as CLI flags.
 *
 *  The bytes come from the server (the harness's own writer): re-serializing
 *  the job dicts here would be a second statement of the file format, and the
 *  harness reads this file back line by line.
 */
export function BenchJobsExport({ target, phase }: { target: GraphTarget; phase: GraphPhase }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlanJobs | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    setPlan(null);
    try {
      const data = await fetchJson<PlanJobs>("/inference/bench/plan-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(benchJobsBody(target, phase)),
      });
      setPlan(data);
      const blob = new Blob([data.jsonl], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Each op once, in the order the planner reported it; the full note is the
  // tooltip, because a drop with a reason says which projection is missing.
  const ops: string[] = [];
  for (const note of plan?.dropped ?? []) {
    const op = droppedOp(note);
    if (!ops.includes(op)) ops.push(op);
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-2 text-xs" data-bench-export="">
      <button
        type="button"
        id={`infer-bench-export-${phase}`}
        className="px-2 py-1 border border-gray-300 rounded text-gray-600 hover:bg-amber-50 hover:border-amber-300 disabled:opacity-50"
        title="Download this step as Kernel Bench jobs (jobs.jsonl) — the same file `hbench plan` writes, for the deployment and workload on screen."
        disabled={busy}
        onClick={() => void run()}
      >
        {busy ? "Planning…" : "Export bench jobs"}
      </button>
      {error ? (
        <span id={`infer-bench-export-error-${phase}`} role="alert" className="text-red-700">
          Export failed: {error}
        </span>
      ) : plan ? (
        <span id={`infer-bench-export-note-${phase}`} className="text-gray-500">
          {plan.job_count} job{plan.job_count === 1 ? "" : "s"} → {plan.filename}
          {ops.length > 0 ? (
            <>
              {" · "}
              <span
                className="text-amber-700"
                data-bench-export-dropped=""
                title={plan.dropped.join("\n")}
              >
                no kernel signature, not exported: {ops.join(", ")}
              </span>
            </>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}
