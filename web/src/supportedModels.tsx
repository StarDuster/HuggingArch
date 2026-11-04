// The model-id autocomplete shared by Inference and Inspector: the
// `/supported-models` list, its filter/sort order, and the origin badge each
// dropdown row carries.
import { useEffect, useState } from "react";
import { fetchJson } from "./api";

export type SupportedModel = { model_id: string; source?: string };

type SupportedModelsResponse = {
  items?: Array<{ id?: string; model_id?: string; source?: string }>;
  model_ids?: string[];
};

function readSupportedModels(j: SupportedModelsResponse): SupportedModel[] {
  if (Array.isArray(j.items) && j.items.length) {
    return j.items.map((m) => ({
      model_id: m.model_id || m.id || "",
      source: m.source,
    })).filter((m) => m.model_id);
  }
  return (j.model_ids || []).filter(Boolean).map((id) => ({ model_id: id, source: "built-in" }));
}

/** `/supported-models`, fetched once on mount; `loading` until it lands or fails. */
export function useSupportedModels(onError?: (e: unknown) => void): { models: SupportedModel[]; loading: boolean } {
  const [models, setModels] = useState<SupportedModel[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    fetchJson<SupportedModelsResponse>("/supported-models")
      .then((j) => setModels(readSupportedModels(j)))
      .catch((e) => {
        setModels([]);
        onError?.(e);
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return { models, loading };
}

function sourceRank(s: string | undefined): number {
  return s === "custom" ? 0 : s === "built-in" ? 1 : 2;
}

/** Same order as vanilla `filterAndSort` in dropdowns.js. */
export function filterAndSortModels(items: SupportedModel[], filter: string): SupportedModel[] {
  const q = (filter || "").toLowerCase().trim();
  return items
    .filter((it) => !q || it.model_id.toLowerCase().includes(q))
    .slice()
    .sort((a, b) => {
      const d = sourceRank(a.source) - sourceRank(b.source);
      return d !== 0 ? d : a.model_id.localeCompare(b.model_id);
    });
}

/** Same three badges as vanilla `sourceBadge` in dropdowns.js. */
export function SourceBadge({ source }: { source: string | undefined }) {
  if (source === "custom") {
    return <span className="ml-2 text-10 font-medium px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">custom</span>;
  }
  if (source === "built-in") {
    return <span className="ml-2 text-10 font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">built-in</span>;
  }
  if (source === "cached") {
    return <span className="ml-2 text-10 font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">cached</span>;
  }
  return null;
}

/** The rows of a model dropdown: "Loading…" while fetching, "No matches", or one clickable row per model. */
export function ModelOptions({
  items,
  action,
  index,
  onSelect,
  loading,
}: {
  items: SupportedModel[];
  action: string;
  index?: number;
  onSelect: (modelId: string) => void;
  loading?: boolean;
}) {
  if (items.length === 0) {
    return <div className="px-3 py-2 text-sm text-gray-400">{loading ? "Loading…" : "No matches"}</div>;
  }
  return (
    <>
      {items.map((it) => (
        <div
          key={it.model_id}
          className="flex items-center justify-between px-3 py-1.5 text-sm text-gray-700 hover:bg-amber-50 cursor-pointer"
          data-action={action}
          data-index={index}
          data-model={it.model_id}
          onClick={() => onSelect(it.model_id)}
        >
          <span className="truncate">{it.model_id}</span>
          <SourceBadge source={it.source} />
        </div>
      ))}
    </>
  );
}
