import { useEffect, useState } from "react";
import { fetchJson } from "./api";

/** The `ParallelConfig` knobs the Inference page edits. How many GPUs a
 *  deployment occupies is the backend's answer (`/inference/parallel-shape`). */
export type Parallel = {
  pp: number;
  tp_attention: number;
  tp_dense_ffn: number;
  tp_moe_expert: number;
  ep_moe: number;
  cp: number;
  dp_attention: number;
  dp_replicas: number;
};

export const EMPTY_PARALLEL: Parallel = {
  pp: 1,
  tp_attention: 1,
  tp_dense_ffn: 1,
  tp_moe_expert: 1,
  ep_moe: 1,
  cp: 1,
  dp_attention: 1,
  dp_replicas: 1,
};

/** Same four axes as vanilla `#infer-scaling` / `estimate-parallel-sweep`. */
export const PAR_SWEEP_MODES = ["dp_ep", "cp_ep", "tp_ep", "tp_tp"] as const;
export type ParSweepMode = (typeof PAR_SWEEP_MODES)[number];

export function axis(n: number): number {
  return Math.max(1, Math.floor(n) || 1);
}

/** The page's knobs of a deployment the backend sent (a sweep point's
 *  `parallel`, an estimate's `parallel`). */
export function parallelOf(d: Partial<Record<keyof Parallel, unknown>> | null | undefined): Parallel {
  const out = { ...EMPTY_PARALLEL };
  for (const k of Object.keys(EMPTY_PARALLEL) as (keyof Parallel)[]) out[k] = axis(Number(d?.[k] ?? 1));
  return out;
}

/** How many GPUs a deployment occupies and what makes it invalid —
 *  `ParallelConfig`'s own answers (`/inference/parallel-shape`). */
export type ParallelShape = { cluster_size: number; per_stage_size: number; errors: string[] };

/** The backend's shape of the deployment being edited, re-asked a beat after
 *  the last edit; null until it answers for THIS deployment — an answer is
 *  kept with the deployment it was asked for, so an edit never shows the
 *  previous deployment's GPU count while the new one is on its way. With
 *  `modelId`, the shape is the mesh that model runs: a dense model has no
 *  expert region, so its expert knobs span no ranks. */
export function useParallelShape(par: Parallel, modelId?: string): ParallelShape | null {
  const key = JSON.stringify(par);
  const model = (modelId || "").trim();
  const asked = `${model}|${key}`;
  const [answer, setAnswer] = useState<{ asked: string; shape: ParallelShape } | null>(null);
  useEffect(() => {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      fetchJson<ParallelShape>("/inference/parallel-shape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parallel: JSON.parse(key), ...(model ? { model_id: model } : {}) }),
        signal: ctrl.signal,
      })
        .then((s) => { if (!ctrl.signal.aborted) setAnswer({ asked, shape: s }); })
        .catch(() => { if (!ctrl.signal.aborted) setAnswer(null); });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [key, model, asked]);
  return answer?.asked === asked ? answer.shape : null;
}

/** ρ = max/mean expert load as the user typed it: the number when it is one
 *  `ParallelConfig.ep_load_factor` accepts (finite, ≥ 1), else null. A value
 *  that does not parse is an error to show, not a 1.0 to assume. */
export function parseLoadFactor(text: string): number | null {
  const t = text.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) && n >= 1 ? n : null;
}

/** What is wrong with a typed ρ, or null when it parses. */
export function loadFactorError(text: string): string | null {
  return parseLoadFactor(text) === null
    ? `MoE load imbalance ρ must be a number ≥ 1 (max/mean expert tokens); got "${text}".`
    : null;
}
