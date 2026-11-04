// The readings that appear on every page. What is pinned here is the part a
// refactor can change without any page failing to render: where one unit
// hands over to the next, how many digits survive, and what an absent number
// prints as.
import { describe, expect, test } from "vitest";

import {
  binaryUnit, compressRanges, escapeHtml, formatBytes, formatCount, formatFlops,
  formatGiB, formatNumber, formatSI, formatTokens, siAxisLabel, siPrefix,
} from "./format";

describe("siPrefix / formatSI", () => {
  test("a prefix starts at its own power of ten, not above it", () => {
    expect(siPrefix(999)).toEqual({ scale: 1, prefix: "" });
    expect(siPrefix(1000)).toEqual({ scale: 1e3, prefix: "K" });
    expect(siPrefix(1e6)).toEqual({ scale: 1e6, prefix: "M" });
    expect(siPrefix(1e9)).toEqual({ scale: 1e9, prefix: "G" });
    expect(siPrefix(1e12)).toEqual({ scale: 1e12, prefix: "T" });
    expect(siPrefix(1e15)).toEqual({ scale: 1e15, prefix: "P" });
    // Peta is the largest prefix: beyond it the mantissa grows.
    expect(siPrefix(1e18)).toEqual({ scale: 1e15, prefix: "P" });
  });

  test("the prefix follows the magnitude, so a negative value reads like its twin", () => {
    expect(siPrefix(-2.5e9)).toEqual({ scale: 1e9, prefix: "G" });
    expect(formatSI(-2.5e9, "B", 1)).toBe("-2.5 GB");
  });

  test("two digits by default, and the unit carries the prefix", () => {
    expect(formatSI(2.13e12, "FLOPs")).toBe("2.13 TFLOPs");
    expect(formatSI(1000, "B")).toBe("1.00 KB");
    expect(formatSI(999, "B")).toBe("999.00 B");
    expect(formatSI(1.23456e9, "B", 3)).toBe("1.235 GB");
    expect(formatBytes(2e9)).toBe("2.00 GB");
  });

  test("zero is a number and prints as one; only a non-number prints as a dash", () => {
    expect(formatSI(0, "B")).toBe("0.00 B");
    expect(formatSI(NaN, "B")).toBe("—");
    expect(formatSI(Infinity, "B")).toBe("—");
  });
});

describe("siAxisLabel", () => {
  test("one unit for the whole axis, taken from the axis max", () => {
    const label = siAxisLabel(3e12, "FLOPs");
    // Not each tick's own prefix: 5e11 reads in the axis's tera, as 0.5.
    expect(label(1e12)).toBe("1 TFLOPs");
    expect(label(2e12)).toBe("2 TFLOPs");
    expect(label(5e11)).toBe("0.5 TFLOPs");
    expect(siAxisLabel(900, "B")(300)).toBe("300 B");
  });

  test("zero is bare, with no unit to read", () => {
    expect(siAxisLabel(3e12, "FLOPs")(0)).toBe("0");
  });
});

describe("formatCount", () => {
  test("a prefix above a thousand, the count itself below one", () => {
    expect(formatCount(187.11e6)).toBe("187.11 M");
    expect(formatCount(2.77e12)).toBe("2.77 T");
    expect(formatCount(999.5)).toBe("999.5");
    expect(formatCount(1000)).toBe("1.00 K");
  });

  test("zero counts as a count; absent prints as a dash", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(null)).toBe("—");
    expect(formatCount(undefined)).toBe("—");
    expect(formatCount(NaN)).toBe("—");
  });
});

describe("formatTokens", () => {
  test("exact, with thousands separators", () => {
    expect(formatTokens(4096)).toBe("4,096");
    expect(formatTokens(1048576)).toBe("1,048,576");
    expect(formatTokens(999)).toBe("999");
  });

  test("zero tokens is zero, not a dash", () => {
    expect(formatTokens(0)).toBe("0");
    expect(formatTokens(null)).toBe("—");
    expect(formatTokens(undefined)).toBe("—");
  });
});

describe("formatFlops", () => {
  test("small counts stay whole; a million and up take a prefix", () => {
    expect(formatFlops(999999)).toBe("999999");
    expect(formatFlops(1e6)).toBe("1.00 M");
    expect(formatFlops(-2e9)).toBe("-2.00 G");
    expect(formatFlops(0)).toBe("0");
    expect(formatFlops(null)).toBe("—");
  });
});

describe("formatGiB / binaryUnit", () => {
  test("bytes held read in binary units, from 1 KiB up", () => {
    expect(formatGiB(1024)).toBe("1.00 KiB");
    expect(formatGiB(1.5 * 1024 ** 3)).toBe("1.50 GiB");
    expect(formatGiB(2 * 1024 ** 4)).toBe("2.00 TiB");
    // A thousand bytes is not a KiB: below 1024 the count stays bytes.
    expect(formatGiB(1000)).toBe("1000 B");
    expect(formatGiB(512)).toBe("512 B");
    expect(formatGiB(512.5)).toBe("512.50 B");
    expect(formatGiB(NaN)).toBe("—");
  });

  test("an axis of bytes held is drawn in the unit of its max", () => {
    expect(binaryUnit(2e9)).toEqual({ scale: 1024 ** 3, unit: "GiB" });
    expect(binaryUnit(900)).toEqual({ scale: 1, unit: "B" });
  });
});

describe("formatNumber", () => {
  test("B for billions, not G — this is the counts-with-letters reading", () => {
    expect(formatNumber(2.5e9)).toBe("2.50B");
    expect(formatNumber(1e12)).toBe("1.00T");
    expect(formatNumber(1500)).toBe("1.50K");
    expect(formatNumber(12.345)).toBe("12.35");
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(null)).toBe("—");
  });
});

describe("compressRanges", () => {
  test("consecutive indices collapse, in order, however they arrive", () => {
    expect(compressRanges([3, 1, 2, 7])).toBe("1-3, 7");
    expect(compressRanges([0])).toBe("0");
    expect(compressRanges([1, 2, 4, 5, 6])).toBe("1-2, 4-6");
    // A run of one in the middle is written as itself, not as "1-1".
    expect(compressRanges([1, 3, 4])).toBe("1, 3-4");
    expect(compressRanges([])).toBe("");
  });

  test("the caller's list keeps its order", () => {
    const given = [3, 1, 2];
    compressRanges(given);
    expect(given).toEqual([3, 1, 2]);
  });
});

describe("escapeHtml", () => {
  test("every one of the five, and absent is empty", () => {
    expect(escapeHtml(`<a href="x" title='y'>&</a>`))
      .toBe("&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;");
    expect(escapeHtml(null)).toBe("");
    expect(escapeHtml(undefined)).toBe("");
    expect(escapeHtml(0)).toBe("0");
  });
});
