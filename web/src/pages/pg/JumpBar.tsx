/** In-page jumps for the Playground's sections. Sticks under the page
 *  header and the compare picks bar (`--pg-picks-h`, 0 when the bar is
 *  absent) so the rest of the page is one tap away. */
const LINKS: { id: string; label: string }[] = [
  { id: "pg-structure", label: "Structure" },
  { id: "pg-summary", label: "Summary" },
  { id: "pg-kv", label: "KV" },
  { id: "pg-latency", label: "Latency" },
  { id: "pg-details", label: "Details" },
];

export function JumpBar({ present }: { present?: ReadonlySet<string> }) {
  const links = present ? LINKS.filter((l) => present.has(l.id)) : LINKS;
  if (links.length === 0) return null;
  return (
    <nav
      id="pg-jump"
      className="sticky z-[19] -mx-1 px-1 py-1.5 mb-3 flex gap-2 overflow-x-auto bg-gray-50/95 backdrop-blur border-b border-gray-200 text-xs top-[calc(3.5rem+var(--pg-picks-h,0px))]"
    >
      {links.map((l) => (
        <a
          key={l.id}
          href={`#${l.id}`}
          className="shrink-0 px-2 py-1 rounded text-gray-600 hover:text-gray-900 hover:bg-white"
        >
          {l.label}
        </a>
      ))}
    </nav>
  );
}
