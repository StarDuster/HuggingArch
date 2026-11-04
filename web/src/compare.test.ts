// A Compare link: the models, each row's deployment, and the phase. The
// payload is base64 in the URL, so nothing about it is readable on the page —
// what is pinned here is that the writer and reader agree, that the row count
// stays inside its bounds, and that old links keep meaning what they meant.
import { describe, expect, test } from "vitest";

import {
  draftTokensFromLegacyWindow, encodeSweep, hydrateSweep, mergeSweep, mergeSweepPhase,
  parseModels, readSweepParams, readSweepPhase, SWEEP_MAX_ROWS, SWEEP_MIN_ROWS,
  type SweepRow,
} from "./compare";

const params = (hash: string) => new URLSearchParams(hash.replace(/^#/, ""));

describe("encodeSweep / hydrateSweep", () => {
  test("the rows come back as they went in", () => {
    const rows: SweepRow[] = [
      { model_id: "deepseek-ai/DeepSeek-V3", cfg: { gpu_sku: "H200-SXM", parallel: { tp_attention: 8, ep_moe: 8 } } },
      { model_id: "Qwen/Qwen3-235B-A22B", cfg: null },
    ];
    const template = { gpu_sku: "H100-SXM" };
    const { sub, m, s } = encodeSweep(rows, template);
    expect(sub).toBe("sweep");
    const back = hydrateSweep(m, s);
    expect(back.rows).toEqual(rows);
    expect(back.template).toEqual(template);
    expect(back.droppedPrecision).toBe(false);
  });

  test("the template is not copied onto rows that have no cfg of their own", () => {
    const { m, s } = encodeSweep(
      [{ model_id: "a", cfg: null }, { model_id: "b", cfg: null }], { gpu_sku: "H100-SXM" });
    expect(hydrateSweep(m, s).rows.map((r) => r.cfg)).toEqual([null, null]);
  });

  test("the row count stays between its bounds", () => {
    expect(hydrateSweep("only-one", "").rows).toHaveLength(SWEEP_MIN_ROWS);
    expect(hydrateSweep("", "").rows.map((r) => r.model_id)).toEqual(["", ""]);
    const many = Array.from({ length: SWEEP_MAX_ROWS + 3 }, (_, i) => `m${i}`).join("|");
    expect(hydrateSweep(many, "").rows).toHaveLength(SWEEP_MAX_ROWS);
    expect(hydrateSweep(many, "").rows.map((r) => r.model_id))
      .toEqual(Array.from({ length: SWEEP_MAX_ROWS }, (_, i) => `m${i}`));
  });

  test("a link whose payload is damaged loads as no cfgs, not as a crash", () => {
    expect(hydrateSweep("a|b", "not-base64!!").rows).toEqual([
      { model_id: "a", cfg: null }, { model_id: "b", cfg: null },
    ]);
    expect(hydrateSweep("a|b", "not-base64!!").template).toBeNull();
  });

  test("a link's precision settings are dropped and the drop is reported", () => {
    const { m, s } = encodeSweep(
      [{ model_id: "a", cfg: { gpu_sku: "H100-SXM", kv_dtype: "fp8" } }, { model_id: "b", cfg: null }],
      null);
    const back = hydrateSweep(m, s);
    // Compare prices every model as shipped, so the field cannot ride along.
    expect(back.rows[0].cfg).toEqual({ gpu_sku: "H100-SXM" });
    expect(back.droppedPrecision).toBe(true);
  });

  test("a saved speculative window becomes the drafted-token count it meant", () => {
    const { m, s } = encodeSweep([{ model_id: "a", cfg: { speculative_K: 4 } }, { model_id: "b", cfg: null }], null);
    const cfg = hydrateSweep(m, s).rows[0].cfg;
    expect(cfg).toEqual({ speculative_draft_tokens: 3 });
    expect(draftTokensFromLegacyWindow(4)).toBe(3);
    expect(draftTokensFromLegacyWindow(1)).toBe(0);
    // A window of 0 is not a negative count of drafts.
    expect(draftTokensFromLegacyWindow(0)).toBe(0);
    expect(draftTokensFromLegacyWindow("")).toBeUndefined();
    expect(draftTokensFromLegacyWindow(null)).toBeUndefined();
  });

  test("models are split on the bar and trimmed", () => {
    expect(parseModels(" a | b ")).toEqual(["a", "b"]);
    expect(parseModels("")).toEqual([""]);
  });
});

describe("mergeSweep", () => {
  test("only sub/m/s are the sweep's to write", () => {
    const hash = mergeSweep("#p=prefill&dec=1&model=old", { sub: "sweep", m: "a|b", s: "PAYLOAD" });
    expect(params(hash).get("p")).toBe("prefill");
    expect(params(hash).get("dec")).toBe("1");
    expect(params(hash).get("model")).toBe("old");
    expect(readSweepParams(hash)).toEqual({ sub: "sweep", m: "a|b", s: "PAYLOAD" });
  });

  test("leaving the sweep clears its three keys and nothing else", () => {
    const hash = mergeSweep("#sub=sweep&m=a|b&s=P&p=prefill", null);
    expect(readSweepParams(hash)).toEqual({});
    expect(params(hash).get("p")).toBe("prefill");
    expect(mergeSweep("#sub=sweep&m=a&s=P", null)).toBe("");
  });

  test("an empty part of the sweep is dropped rather than written empty", () => {
    const hash = mergeSweep("#sub=sweep&m=a|b&s=P", { sub: "sweep", m: "", s: "" });
    expect(params(hash).has("m")).toBe(false);
    expect(params(hash).has("s")).toBe(false);
    expect(readSweepParams("#sub=sweep&m=&s=")).toEqual({ sub: "sweep" });
  });
});

describe("sweep phase", () => {
  test("decode is the default and is elided; only prefill is written", () => {
    expect(readSweepPhase("")).toBe("decode");
    expect(readSweepPhase("#p=decode")).toBe("decode");
    expect(readSweepPhase("#p=something-else")).toBe("decode");
    expect(readSweepPhase("#p=prefill")).toBe("prefill");
    expect(params(mergeSweepPhase("#sub=sweep", "prefill")).get("p")).toBe("prefill");
    expect(params(mergeSweepPhase("#sub=sweep&p=prefill", "decode")).has("p")).toBe(false);
    expect(params(mergeSweepPhase("#sub=sweep&p=prefill", null)).has("p")).toBe(false);
    // The round trip: whatever was written reads back as the same phase.
    expect(readSweepPhase(mergeSweepPhase("", "prefill"))).toBe("prefill");
    expect(readSweepPhase(mergeSweepPhase("#p=prefill", "decode"))).toBe("decode");
  });

  test("the sweep's own keys survive a phase change", () => {
    const hash = mergeSweepPhase("#sub=sweep&m=a|b&s=P", "prefill");
    expect(readSweepParams(hash)).toEqual({ sub: "sweep", m: "a|b", s: "P" });
  });
});
