import type { ReactNode } from "react";

/** A Playground block: an id the jump bar can aim at, a title, the basis
 *  its numbers are on (`scope`: "per step" where the prefill/decode toggle
 *  moves them, otherwise what they are fixed to), and an optional footnote.
 *  The chrome is the same `card` shell the page already uses. */
export function Section({
  id, title, scope, footnote, lead, children,
}: {
  id: string;
  title: ReactNode;
  scope?: string;
  footnote?: ReactNode;
  /** First line of the body (serving contract, phase toggle, …). */
  lead?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="card p-4 sm:p-6 mb-6 min-w-0 scroll-mt-[calc(3.5rem+var(--pg-picks-h,0px)+2.75rem)]">
      {/* The scope note drops under the title when both don't fit on one line,
          instead of squeezing the title into a narrow column. */}
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 mb-2">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        {scope != null && (
          <span className="text-11 text-gray-400 min-w-0" data-scope>{scope}</span>
        )}
      </div>
      {lead}
      {children}
      {footnote != null && (
        <div className="mt-2 text-11 text-gray-500">{footnote}</div>
      )}
    </section>
  );
}
