// SourceAnchor — a framework binding's `source_anchor` as the backend cites
// it (`AuditedAt.cite`): the sentence split into plain runs and the places it
// names, each place a permalink to the audited commit. A place shows its file
// name and lines; the full repository path rides the link's title.

export type CitedAnchor = {
  text: string;
  segments: { text: string; url?: string }[];
};

export function SourceAnchor({ anchor }: { anchor: CitedAnchor }) {
  return (
    <>
      {anchor.segments.map((s, i) =>
        s.url ? (
          <a key={i} href={s.url} title={s.text} target="_blank" rel="noopener"
             className="text-blue-600 hover:underline whitespace-nowrap">
            {s.text.slice(s.text.lastIndexOf("/") + 1)}
          </a>
        ) : (
          <span key={i}>{s.text}</span>
        ),
      )}
    </>
  );
}
