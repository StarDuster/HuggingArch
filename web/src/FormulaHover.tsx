// Per-op formula popover: the shared `formulaHTML` renderer (formula /
// values / per layer / = / note rows) inside the anchored shell that keeps
// the FORMULA header and the "{x}-bound" verdict badge. Inference's summary
// cards, memory budget, per-op table, and Playground Details all hover
// numbers through this.
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { BOUND_PILL, isBound } from "./bound";
import { formulaHTML, type FormulaFields } from "./tooltip";

/** A value that answers a hover: a light dotted rule that darkens under the
 *  pointer, so a table of them does not read as a page of links. */
export const HOVER_HINT = "underline decoration-dotted decoration-gray-300 underline-offset-2 hover:decoration-current";

export function FormulaHover({ value, formula, cls }: {
  value: string;
  formula?: FormulaFields;
  cls?: string;
}) {
  const [open, setOpen] = useState(false);
  const hideTimer = useRef<number | null>(null);
  const popRef = useRef<HTMLDivElement | null>(null);
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  // The card hangs right-aligned under the value, in a portal on <body> at a
  // fixed position: a value inside a scrolling table frame would otherwise
  // have its card clipped by that frame. Kept inside the viewport, flipped
  // above the value when there is no room below, and re-placed on scroll.
  useLayoutEffect(() => {
    const pop = popRef.current;
    const anchor = anchorRef.current;
    if (!open || !pop || !anchor) return;
    const place = () => {
      const margin = 8;
      const a = anchor.getBoundingClientRect();
      const r = pop.getBoundingClientRect();
      const left = Math.min(Math.max(a.right - r.width, margin), window.innerWidth - margin - r.width);
      const below = a.bottom + 4;
      const top = below + r.height > window.innerHeight - margin && a.top - 4 - r.height >= margin
        ? a.top - 4 - r.height
        : below;
      pop.style.left = `${left}px`;
      pop.style.top = `${top}px`;
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open]);
  const has = !!(formula && (formula.symbolic || formula.substituted || formula.note
    || formula.result != null || formula.result_bytes != null || formula.description || formula.label));
  const show = () => {
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    hideTimer.current = null;
    setOpen(true);
  };
  const hide = () => {
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setOpen(false), 180);
  };
  if (!has) return <span className={cls}>{value}</span>;
  const substituted = formula!.substituted || "";
  const boundMatch = substituted.match(/=\s*([a-z]+)-bound\s*$/i);
  const bound = boundMatch?.[1];
  // The bound verdict rides the badge; the values row keeps the equation.
  const fields: FormulaFields = {
    ...formula!,
    substituted: boundMatch ? substituted.slice(0, boundMatch.index).trim() : substituted,
  };
  return (
    <span ref={anchorRef} className="relative inline-block" onMouseEnter={show} onMouseLeave={hide}>
      <span
        className={`${cls || ""} ${HOVER_HINT}`}
        data-formula-pop=""
      >
        {value}
      </span>
      {open ? createPortal(
        <div
          ref={popRef}
          className="fixed z-50 left-0 top-0 w-[min(24rem,calc(100vw-32px))] max-h-[70vh] overflow-x-hidden overflow-y-auto bg-white border border-gray-200 rounded-xl shadow-2xl text-left text-11 font-sans font-normal not-italic leading-snug whitespace-normal"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-100 bg-gray-50">
            <span className="text-10 font-semibold uppercase tracking-wide text-gray-400">Formula</span>
            {bound ? (
              <span className={`px-1.5 py-px rounded-full text-10 font-semibold uppercase tracking-wide ${
                isBound(bound) ? BOUND_PILL[bound] : "bg-gray-50 text-gray-600 border border-gray-200"
              }`}>{bound}-bound</span>
            ) : null}
          </div>
          <div className="p-3" dangerouslySetInnerHTML={{ __html: formulaHTML(fields) }} />
        </div>,
        document.body,
      ) : null}
    </span>
  );
}
