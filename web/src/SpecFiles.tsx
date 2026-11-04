// The two kinds of file a spec generation produces — the model spec and the
// new components it introduces — placed at the paths its pull request uses
// for them: the spec at backend/arch/models/<id>.yaml (github.py
// `_bundle_files`), each component at its own path. Not the whole PR: the
// snapshot shows in the drawer's metadata section, and what the PR splices
// into ops_registry.py / dims.yaml exists only once the PR is built. A
// collapsible directory tree picks the file; one file shows at a time.
// Shared by the account drawer and both admin drawers.
import { useEffect, useMemo, useState } from "react";

export type SpecFile = { path?: string; body?: string };

type File = { path: string; body: string; spec: boolean };
type Dir = { name: string; path: string; dirs: Dir[]; files: File[] };

function buildTree(files: File[]): Dir {
  const root: Dir = { name: "", path: "", dirs: [], files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let cur = root;
    for (const part of parts.slice(0, -1)) {
      const path = cur.path ? `${cur.path}/${part}` : part;
      let next = cur.dirs.find((d) => d.name === part);
      if (!next) {
        next = { name: part, path, dirs: [], files: [] };
        cur.dirs.push(next);
      }
      cur = next;
    }
    cur.files.push(f);
  }
  // A directory whose only entry is one directory reads as one row
  // (backend/arch/components/drafts), the way code hosts show it.
  const compact = (d: Dir): Dir => {
    let cur = { ...d, dirs: d.dirs.map(compact) };
    while (cur.name && cur.files.length === 0 && cur.dirs.length === 1) {
      const only = cur.dirs[0];
      cur = { ...only, name: `${cur.name}/${only.name}` };
    }
    return cur;
  };
  return compact(root);
}

function leaf(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg className={`w-3 h-3 shrink-0 text-gray-400 transition-transform ${open ? "rotate-90" : ""}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M7 5l6 5-6 5V5z" />
    </svg>
  );
}

function DirRows({ dir, depth, closed, onToggle, selected, onSelect }: {
  dir: Dir; depth: number; closed: Set<string>; onToggle: (p: string) => void;
  selected: string; onSelect: (p: string) => void;
}) {
  const pad = (d: number) => ({ paddingLeft: `${8 + d * 12}px` });
  return (
    <>
      {dir.dirs.map((d) => {
        const open = !closed.has(d.path);
        return (
          <div key={`d:${d.path}`}>
            <button type="button" style={pad(depth)} onClick={() => onToggle(d.path)}
              className="w-full flex items-center gap-1 pr-2 py-1 text-left text-11 font-mono text-gray-600 hover:bg-gray-100 truncate"
              title={d.path}>
              <Chevron open={open} />
              <span className="truncate">{d.name}/</span>
            </button>
            {open && <DirRows dir={d} depth={depth + 1} closed={closed} onToggle={onToggle} selected={selected} onSelect={onSelect} />}
          </div>
        );
      })}
      {dir.files.map((f) => (
        <button key={`f:${f.path}`} type="button" style={pad(depth)} onClick={() => onSelect(f.path)}
          className={`w-full flex items-center gap-1.5 pr-2 py-1 text-left text-11 font-mono truncate ${
            f.path === selected ? "bg-amber-100 text-gray-900" : "text-gray-700 hover:bg-gray-100"}`}
          title={f.path}>
          <span className="w-3 shrink-0" />
          <span className="truncate">{leaf(f.path)}</span>
          {f.spec && <span className="ml-auto shrink-0 text-10 px-1.5 rounded bg-gray-200 text-gray-600 font-sans">spec</span>}
        </button>
      ))}
    </>
  );
}

export function SpecFiles({ modelId, specYaml, components }: {
  modelId?: string; specYaml?: string; components?: SpecFile[];
}) {
  const files = useMemo<File[]>(() => {
    const out: File[] = [];
    if (specYaml != null) {
      const safe = (modelId || "model").replace(/\//g, "__");
      out.push({ path: `backend/arch/models/${safe}.yaml`, body: specYaml, spec: true });
    }
    for (const c of components || []) {
      if (c.path) out.push({ path: c.path, body: c.body || "", spec: false });
    }
    return out;
  }, [modelId, specYaml, components]);
  const tree = useMemo(() => buildTree(files), [files]);
  const [selected, setSelected] = useState("");
  const [closed, setClosed] = useState<Set<string>>(new Set());
  // Land on the spec (or the first file) when a different set of files
  // opens — keyed on the paths, so a re-fetch of the same record (a drawer
  // refreshing while a review runs) keeps the file the reader picked.
  const listing = files.map((f) => f.path).join("\n");
  useEffect(() => { setSelected(listing.split("\n")[0] || ""); setClosed(new Set()); }, [listing]);
  const current = files.find((f) => f.path === selected) || files[0];
  if (!current) return null;
  const toggle = (p: string) => setClosed((s) => {
    const next = new Set(s);
    if (next.has(p)) next.delete(p); else next.add(p);
    return next;
  });
  const lines = current.body ? current.body.split("\n").length : 0;
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden flex flex-col sm:flex-row min-h-[16rem] max-h-[70vh]">
      <nav className="sm:w-64 shrink-0 border-b sm:border-b-0 sm:border-r border-gray-200 bg-gray-50 overflow-auto py-1 max-h-40 sm:max-h-none">
        <div className="px-2 pb-1 text-10 uppercase tracking-wide text-gray-400">{files.length} file{files.length === 1 ? "" : "s"}</div>
        <DirRows dir={tree} depth={0} closed={closed} onToggle={toggle} selected={current.path} onSelect={setSelected} />
      </nav>
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-200 bg-white text-11">
          <span className="font-mono text-gray-700 truncate" title={current.path}>{current.path}</span>
          <span className="ml-auto shrink-0 text-gray-400">{lines} lines</span>
          <button type="button" className="shrink-0 text-gray-500 hover:text-gray-900"
            onClick={() => void navigator.clipboard?.writeText(current.body)}>Copy</button>
        </div>
        <pre className="flex-1 overflow-auto text-11 font-mono p-3 m-0 whitespace-pre text-gray-800">{current.body || "(empty)"}</pre>
      </div>
    </div>
  );
}
