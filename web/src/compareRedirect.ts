/** Retired /compare.html hash → Inspector compare or Inference sweep.
 *  Keys and dests match vanilla `frontend/compare.html`.
 */

export function legacyCompareDest(hash: string): string {
  let h = (hash || "").replace(/^#\/?/, "");
  const q = h.indexOf("?");
  if (q >= 0 && /^[a-z]+$/.test(h.slice(0, q))) h = h.slice(q + 1);
  const p = new URLSearchParams(h);
  if (p.get("sub") === "inference") {
    p.set("sub", "sweep");
    return `/inference.html#${p.toString()}`;
  }
  const out = new URLSearchParams();
  out.set("sub", "compare");
  let models: string[] = [];
  if (p.get("m")) {
    models = p.get("m")!.split("|");
  } else {
    if (p.get("a")) models.push(p.get("a")!);
    if (p.get("b")) models.push(p.get("b")!);
  }
  if (models.length) out.set("m", models.join("|"));
  return `/inspector.html#${out.toString()}`;
}
