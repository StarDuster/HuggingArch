/** In-tab cache for Scaling-panel sweep POSTs, and the sweep's two answers.

A sweep body has two answers (docs/auto_sweep.md §7.5): its SOL answer
(`sol_only`, or a `theoretical_only` body) reads no measurement — a function
of the body alone, cached here and on the server, so switching a dropdown back
to a combination already drawn does not wait on HTTP; and, when its mode reads
measurements, its measured answer — what Kernel Bench holds now, which the tab
cannot know, so it is asked of the server live every time and never held.
The page draws the SOL answer, then the measured one over it.

Keyed by path + JSON body. Memory only — a reload or a new `code_version`
from a redeploy starts empty. LRU-capped so a long click-around of GPU ×
parallel × degree cannot grow without bound.
*/
import { fetchJson, type StillComputing } from "./api";

const MAX_ENTRIES = 64;
const store = new Map<string, unknown>();

/** Whether a sweep body reads no measurements: its SOL answer alone. */
export function isTheory(body: unknown): boolean {
  const b = body as { selection_mode?: string; sol_only?: boolean } | null;
  return b?.sol_only === true || b?.selection_mode === "theoretical_only";
}

/** The body's SOL answer: the same body, asked for its SOL alone. */
export function solOnly<B extends object>(body: B): B & { sol_only: true } {
  return { ...body, sol_only: true };
}

export function sweepCacheKey(path: string, body: unknown): string {
  return `${path}:${JSON.stringify(body)}`;
}

export function sweepCacheGet<T>(key: string): T | undefined {
  if (!store.has(key)) return undefined;
  const value = store.get(key);
  store.delete(key);
  store.set(key, value);
  return value as T;
}

export function sweepCacheSet(key: string, value: unknown): void {
  if (store.has(key)) store.delete(key);
  store.set(key, value);
  while (store.size > MAX_ENTRIES) {
    const oldest = store.keys().next().value;
    if (oldest === undefined) break;
    store.delete(oldest);
  }
}

/** The tab's answer for a SOL body; never one for a measured body. */
export function readSweepCache<T>(path: string, body: unknown): T | undefined {
  return isTheory(body) ? sweepCacheGet<T>(sweepCacheKey(path, body)) : undefined;
}

export async function fetchSweepCached<T>(
  path: string,
  body: unknown,
  init: RequestInit,
  onStillComputing?: (info: StillComputing) => void,
): Promise<T> {
  const hit = readSweepCache<T>(path, body);
  if (hit !== undefined) return hit;
  const data = await fetchJson<T>(path, {
    ...init,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    body: JSON.stringify(body),
  }, onStillComputing);
  if (isTheory(body)) sweepCacheSet(sweepCacheKey(path, body), data);
  return data;
}
