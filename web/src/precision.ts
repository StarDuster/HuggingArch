/** Deployment precision profiles on the client: which named profile a page
 *  holds, how a request carries it, and how it lives in a page URL.
 *
 *  The profiles themselves (labels, descriptions, canonical rules) come from
 *  `GET /precision/presets`; nothing here names or describes one. A page
 *  holds a preset id; "as shipped" is the absence of a `precision` field, so
 *  a default request (and its cache key) is the same as one sent without
 *  any profile. */

/** The id of the profile every page starts on: each model as it ships. */
export const AS_SHIPPED = "as_shipped";

/** One `GET /precision/presets` entry. */
export type PrecisionPreset = {
  id: string;
  label: string;
  description: string;
  profile_key: string;
  canonical: unknown;
};

/** One row of a response's `precision.effective`: an item the profile
 *  resolves (weights of a role, an edge, a cache slot …), as shipped and as
 *  it runs, in the backend's own words. */
export type PrecisionEffectiveRow = {
  category: string;
  selector: string;
  as_shipped: string;
  effective: string;
  source: string;
  changed: boolean;
};

/** A response's `precision` block: the profile the numbers were computed
 *  under and what it changed on this model. */
export type PrecisionReport = {
  profile_key: string;
  label: string;
  preset: string | null;
  canonical: unknown;
  overridden?: boolean;
  effective?: PrecisionEffectiveRow[];
  inapplicable?: { rule: string; reason: string }[];
};

/** The request field for `presetId`: omitted for as shipped. */
export function precisionBody(presetId: string): { precision?: { preset: string } } {
  return presetId && presetId !== AS_SHIPPED ? { precision: { preset: presetId } } : {};
}

/** Set the `precision` query parameter (JSON text of the request field) on a
 *  GET query; as shipped leaves it absent. */
export function setPrecisionQuery(qs: URLSearchParams, presetId: string): URLSearchParams {
  const body = precisionBody(presetId);
  if (body.precision) qs.set("precision", JSON.stringify(body.precision));
  else qs.delete("precision");
  return qs;
}

/** The request fields that carried precision before profiles. The backend
 *  refuses them by name; a saved config that still has them is read without
 *  them. */
export const RETIRED_PRECISION_FIELDS = [
  "act_dtype_bytes", "act_dtype_per_role", "kv_dtype", "kv_quant_granularity",
  "kv_scale_dtype", "weight_dtype_bytes", "weight_quant_granularity",
  "weight_scale_dtype", "weight_dtype_per_role", "compute_dtype_bytes",
] as const;

/** `cfg` without the retired precision fields, and whether it had any. */
export function withoutRetiredPrecision<T extends Record<string, unknown>>(cfg: T): { cfg: T; dropped: boolean } {
  let dropped = false;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(cfg)) {
    if ((RETIRED_PRECISION_FIELDS as readonly string[]).includes(k)) dropped = true;
    else out[k] = v;
  }
  return { cfg: out as T, dropped };
}

/** Said beside the profile buttons when a link or draft carried precision
 *  settings from before profiles. */
export const PREDATES_PROFILES_NOTICE =
  "This link's precision settings predate precision profiles and were not applied; showing As shipped.";

/** The preset id a page URL hash holds (`prec=`); as shipped when absent. */
export function readPrecisionHash(hash: string): string {
  const v = new URLSearchParams(hash.replace(/^#/, "")).get("prec");
  return v ? v : AS_SHIPPED;
}

/** Write `prec=<preset id>`; drop it for as shipped. Leaves other keys. */
export function mergePrecisionHash(existingHash: string, presetId: string): string {
  const params = new URLSearchParams(existingHash.replace(/^#/, ""));
  if (presetId && presetId !== AS_SHIPPED) params.set("prec", presetId);
  else params.delete("prec");
  const next = params.toString();
  return next ? `#${next}` : "";
}
