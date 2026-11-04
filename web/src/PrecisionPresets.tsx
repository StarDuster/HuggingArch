/** The precision switch shared by Inference and Playground: one button per
 *  named profile from `GET /precision/presets`, and what a response says its
 *  profile changed. Every label and value shown is the backend's string. */
import { useEffect, useState } from "react";
import { fetchJson } from "./api";
import type { PrecisionPreset, PrecisionReport } from "./precision";

let presetsRequest: Promise<PrecisionPreset[]> | null = null;

function loadPresets(): Promise<PrecisionPreset[]> {
  presetsRequest ??= fetchJson<{ presets?: PrecisionPreset[] }>("/precision/presets")
    .then((j) => (Array.isArray(j.presets) ? j.presets : []))
    .catch((e) => {
      presetsRequest = null; // a later mount asks again
      throw e;
    });
  return presetsRequest;
}

/** The named profiles, once the backend has listed them (null before). */
function usePrecisionPresets(): { presets: PrecisionPreset[] | null; error: string } {
  const [presets, setPresets] = useState<PrecisionPreset[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    loadPresets()
      .then((p) => { if (live) setPresets(p); })
      .catch((e) => { if (live) setError(`Failed to load precision profiles: ${e instanceof Error ? e.message : String(e)}`); });
    return () => { live = false; };
  }, []);
  return { presets, error };
}

/** The profile buttons; `notice` is a one-line message beside them (an old
 *  link's settings that were not applied). */
export function PrecisionPresets({ value, onChange, notice, idPrefix }: {
  value: string;
  onChange: (presetId: string) => void;
  notice?: string | null;
  idPrefix: string;
}) {
  const { presets, error } = usePrecisionPresets();
  return (
    <div id={`${idPrefix}-precision`} className="text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-gray-600">Precision</span>
        {(presets || []).map((p) => (
          <button
            key={p.id}
            type="button"
            data-preset={p.id}
            aria-pressed={p.id === value}
            title={p.description}
            onClick={() => onChange(p.id)}
            className={`px-2.5 py-1 rounded-lg border transition ${p.id === value
              ? "border-amber-500 bg-amber-500 text-white"
              : "border-gray-300 bg-white text-gray-700 hover:bg-amber-50 hover:border-amber-300"}`}
          >
            {p.label}
          </button>
        ))}
        {error ? <span className="text-red-500">{error}</span> : null}
      </div>
      {notice ? <div data-precision-notice className="mt-1 text-amber-700">{notice}</div> : null}
    </div>
  );
}

/** The profile a response was computed under and, when it changed anything
 *  on this model, each changed item as shipped → as run. */
export function PrecisionApplied({ report, id }: { report: PrecisionReport | null | undefined; id?: string }) {
  if (!report) return null;
  const changed = report.overridden ? (report.effective || []).filter((r) => r.changed) : [];
  return (
    <div id={id} className="text-xs text-gray-600">
      <span className="text-gray-500">Precision:</span>{" "}
      <span className="font-medium text-gray-800">{report.label}</span>
      {changed.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {changed.map((r, i) => (
            <li key={`${r.category}-${r.selector}-${i}`} className="font-mono text-gray-700">
              <span className="text-gray-400 font-sans">{r.category}</span> {r.selector}: {r.as_shipped} → {r.effective}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
