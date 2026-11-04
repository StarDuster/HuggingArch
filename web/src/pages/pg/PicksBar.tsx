import type { Ref } from "react";

/** Compact compare-mode config row: every selected entry, sticky under the
 *  page header. Chips wrap onto further lines instead of clipping off-screen.
 *  Binding can be changed here, and — when the model calls the component in
 *  more than one place — the call it is evaluated at; × removes; + Add opens
 *  the component catalog. Structural changes stage until Benchmark commits
 *  the whole selection, so adding several comparisons does not fire a run
 *  per chip. */
export type PicksBarItem = {
  key: string;
  name: string;
  modelId: string | null;
  modelLabel: string | null;
  models: { id: string; label: string }[];
  canRebind: boolean;
  // The call the entry is evaluated at, and the calls it can name.
  call: string | null;
  calls: { call: string; label: string }[];
};

/** A chip's dropdowns draw their own arrow in reserved room at the right, so
 *  a long model name ends in "…" before the arrow instead of running under
 *  it (the native arrow claims no room once the padding is zeroed). */
const CHIP_SELECT = "text-11 h-5 leading-5 bg-transparent border-0 text-amber-800 font-normal pl-0 pr-3.5 py-0 appearance-none truncate cursor-pointer";
const CHEVRON = {
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23b45309' stroke-width='1.5'/%3E%3C/svg%3E")`,
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right center",
  backgroundSize: "8px 5px",
} as const;

export function PicksBar({
  items, onRebind, onPickCall, onRemove, onAdd, barRef,
}: {
  items: PicksBarItem[];
  onRebind: (key: string, modelId: string | null) => void;
  onPickCall: (key: string, call: string) => void;
  onRemove: (key: string) => void;
  onAdd: () => void;
  barRef?: Ref<HTMLDivElement>;
}) {
  return (
    <div
      id="pg-picks"
      ref={barRef}
      className="sticky top-14 z-20 -mx-[var(--page-gutter)] px-[var(--page-gutter)] py-1 mb-3 flex flex-wrap items-center gap-1.5 min-w-0 bg-white/95 backdrop-blur border-b border-gray-200"
    >
      {items.map((e) => (
        <div
          key={e.key}
          data-entry={e.key}
          className="shrink-0 flex items-center gap-1 text-xs px-2 py-1 rounded-full border border-amber-500 bg-amber-50 text-amber-800"
        >
          <span className="font-medium whitespace-nowrap">{e.name}</span>
          {e.canRebind && e.models.length > 0 ? (
            <>
              <span className="text-amber-400">·</span>
              <select
                value={e.modelId ?? ""}
                onChange={(ev) => onRebind(e.key, ev.target.value || null)}
                title="Bind this column to a shipped model — its dims become that model's parameterization (edits reset)"
                className={`${CHIP_SELECT} max-w-[9rem]`} style={CHEVRON}
              >
                {e.modelId == null && <option value="">bind…</option>}
                {e.models.map((m) => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
            </>
          ) : e.modelLabel ? (
            <>
              <span className="text-amber-400">·</span>
              <span className="whitespace-nowrap">{e.modelLabel}</span>
            </>
          ) : null}
          {e.calls.length > 1 && (
            <>
              <span className="text-amber-400">·</span>
              <select
                data-call-pick={e.key}
                value={e.call ?? ""}
                onChange={(ev) => { if (ev.target.value) onPickCall(e.key, ev.target.value); }}
                title="The model calls this component in more than one place: the call this column is evaluated at"
                className={`${CHIP_SELECT} max-w-[12rem]`} style={CHEVRON}
              >
                {e.call == null && <option value="">at which call…</option>}
                {e.calls.map((c) => (
                  <option key={c.call} value={c.call}>{c.label}</option>
                ))}
              </select>
            </>
          )}
          <button type="button" title="Remove" aria-label={`Remove ${e.name}`} onClick={() => onRemove(e.key)}
            className="text-amber-400 hover:text-amber-700 leading-none">×</button>
        </div>
      ))}
      <button
        id="pg-add-pick"
        type="button"
        onClick={onAdd}
        className="shrink-0 text-xs px-2.5 py-1 rounded-full border border-gray-300 text-gray-600 hover:bg-gray-50"
      >
        + Add
      </button>
    </div>
  );
}
