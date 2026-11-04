// An engine-blocked spec as the pages show it: the badge on a record or a
// run, the gap table in a record's drawer, and the banner above an analysis
// the Inspector serves from such a record. Shared by the account page, the
// admin drawer and the Inspector, so a blocked spec reads the same wherever
// it appears.

import type { ReactNode } from "react";
import { DataTable, type ColumnDef } from "./DataTable";
import { gapsTitle, GENERATION_STATUS_BADGE, type EngineGap } from "./specStatus";

/** The violet pill; hover lists every gap. `engineChanged`: the engine this
 *  server runs differs from the one that blocked the record. */
export function EngineBlockedBadge({ gaps, engineChanged }: {
  gaps?: EngineGap[] | null;
  engineChanged?: boolean;
}) {
  const badge = GENERATION_STATUS_BADGE.blocked_by_engine;
  const title = [badge.title, gapsTitle(gaps)].filter(Boolean).join("\n");
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap" data-engine-blocked>
      <span className={`px-1.5 py-px rounded-md text-10 font-semibold tracking-wide ${badge.cls}`} title={title}>
        {badge.label}
      </span>
      {engineChanged ? (
        <span
          className="px-1.5 py-px rounded-md border border-violet-200 bg-white text-10 font-medium text-violet-600"
          title="The analysis engine changed since this spec was blocked; a re-check may now pass."
          data-engine-changed
        >
          engine updated — re-check
        </span>
      ) : null}
    </span>
  );
}

/** One row per gap: what the engine lacks, where, and on which deployments. */
export function EngineGapTable({ gaps }: { gaps?: EngineGap[] | null }) {
  if (!gaps?.length) return null;
  const violetHead = "bg-violet-50 text-violet-700";
  const columns: ColumnDef<EngineGap>[] = [
    { key: "capability", wrap: true, header: "Capability", headerClassName: violetHead, className: "text-gray-800", sortValue: (g) => g.capability },
    {
      key: "node", wrap: true,
      header: "Node",
      headerClassName: violetHead,
      className: "font-mono text-gray-600 break-all",
      render: (g) => g.node || "—",
      sortValue: (g) => g.node,
    },
    {
      key: "deployments", wrap: true,
      header: "Deployments",
      headerClassName: violetHead,
      className: "text-gray-600",
      render: (g) => (g.deployments || []).join(", "),
      sortValue: (g) => (g.deployments || []).join(", "),
    },
    {
      key: "stages", wrap: true,
      header: "Stages",
      headerClassName: violetHead,
      className: "text-gray-600",
      render: (g) => (g.stages || []).join(", "),
      sortValue: (g) => (g.stages || []).join(", "),
    },
    {
      key: "raisedAt", wrap: true,
      header: "Raised at",
      headerClassName: violetHead,
      className: "font-mono text-gray-500",
      render: (g) => `${g.site || "—"}${g.exception ? ` (${g.exception})` : ""}`,
      sortValue: (g) => g.site,
    },
  ];
  return (
    <DataTable
      columns={columns}
      rows={gaps}
      rowKey={(_g, i) => i}
      framed={false}
      className="border border-violet-200 rounded-lg"
      rowProps={(g) => ({ title: g.detail || undefined })}
    />
  );
}

/** The banner of a spec the engine blocks: each capability with the
 *  deployments and stages where validate met it. */
export function EngineBlockedBanner({ title, gaps, note, children }: {
  title: string;
  gaps?: EngineGap[] | null;
  /** Under the gap list: what this page's own analysis hit, or a save that failed. */
  note?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="mb-4 bg-violet-50 border border-violet-200 rounded-lg p-4" data-engine-blocked-banner>
      <div className="flex items-start justify-between gap-4">
        <div className="text-violet-900 min-w-0">
          <div className="font-medium">{title}</div>
          <ul className="mt-1 text-sm text-violet-800 space-y-0.5">
            {(gaps || []).map((g, i) => (
              <li key={i} title={g.detail || undefined}>
                <span className="font-medium">{g.capability}</span>
                {g.node ? <> at <code className="font-mono text-xs">{g.node}</code></> : null}
                <span className="text-violet-600"> — {(g.deployments || []).join(", ")} ({(g.stages || []).join(", ")})</span>
              </li>
            ))}
          </ul>
          {note ? <div className="mt-2 text-sm text-violet-900">{note}</div> : null}
        </div>
        {children ? <div className="flex items-center gap-2 shrink-0">{children}</div> : null}
      </div>
    </div>
  );
}
