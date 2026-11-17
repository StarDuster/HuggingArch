import { afterEach, describe, expect, test, vi } from "vitest";

import { currentHref, readLastInspectorHref, rememberInspectorHref } from "./lastVisited";

const storage = () => {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
  };
};

afterEach(() => vi.unstubAllGlobals());

describe("last Inspector destination", () => {
  test("the model and inner view survive a top-level navigation", () => {
    const dispatchEvent = vi.fn();
    vi.stubGlobal("window", {
      location: {
        origin: "https://huggingarch.test",
        pathname: "/inspector.html",
        search: "",
        hash: "#model=acme%2FFoo&view=diagram",
      },
      dispatchEvent,
    });
    vi.stubGlobal("sessionStorage", storage());
    vi.stubGlobal("CustomEvent", class<T> { detail: T; constructor(_name: string, init: { detail: T }) { this.detail = init.detail; } });

    expect(currentHref()).toBe("/inspector.html#model=acme%2FFoo&view=diagram");
    rememberInspectorHref();
    expect(readLastInspectorHref()).toBe("/inspector.html#model=acme%2FFoo&view=diagram");
    expect(dispatchEvent).toHaveBeenCalledOnce();
  });

  test("a stored non-Inspector or cross-origin URL is never followed", () => {
    const held = storage();
    vi.stubGlobal("window", {
      location: { origin: "https://huggingarch.test", pathname: "/", search: "", hash: "" },
      dispatchEvent: vi.fn(),
    });
    vi.stubGlobal("sessionStorage", held);
    held.setItem("ha:last-inspector-url", "https://example.test/steal");
    expect(readLastInspectorHref()).toBe("/");
    held.setItem("ha:last-inspector-url", "/inference.html#model=x");
    expect(readLastInspectorHref()).toBe("/");
  });
});
