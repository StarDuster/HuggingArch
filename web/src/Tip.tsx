// The shared hover popover (`#ha-tip`, mounted once by App as <TipPopover>).
// An element gets a tip by spreading `tipHandlers(html)` onto itself; React
// owns those listeners, so they go away with the element.
import type { MouseEvent as ReactMouseEvent, WheelEvent as ReactWheelEvent } from "react";

const tipEl = () => document.getElementById("ha-tip");
function positionTip(x: number, y: number) {
  const t = tipEl();
  if (!t || t.classList.contains("hidden")) return;
  const tw = t.offsetWidth, th = t.offsetHeight;
  const pad = 12;
  const margin = 8;
  let left = x + pad;
  let top = y + pad;
  if (left + tw > window.innerWidth - margin) left = x - tw - pad;
  if (top + th > window.innerHeight - margin) top = y - th - pad;
  left = Math.min(Math.max(margin, left), Math.max(margin, window.innerWidth - tw - margin));
  top = Math.min(Math.max(margin, top), Math.max(margin, window.innerHeight - th - margin));
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}
let tipHideTimer: ReturnType<typeof setTimeout> | null = null;
function cancelHideTip() {
  if (tipHideTimer !== null) { clearTimeout(tipHideTimer); tipHideTimer = null; }
}
function scheduleHideTip() {
  cancelHideTip();
  tipHideTimer = setTimeout(() => {
    const t = tipEl();
    // The card may have appeared under the cursor (a tall formula). :hover
    // is true even when mouseenter never fired, so keep it open to scroll.
    if (t && t.matches(":hover")) { tipHideTimer = null; return; }
    t?.classList.add("hidden");
    tipHideTimer = null;
  }, 200);
}
function showTipAt(x: number, y: number, html: string) {
  const t = tipEl();
  if (!t) return;
  cancelHideTip();
  t.scrollTop = 0;
  t.innerHTML = html;
  t.classList.remove("hidden");
  positionTip(x, y);
}

/** The one popover every tip renders into. Hovering it keeps it open, so its
 *  text can be selected and a tall formula can be scrolled. */
export function TipPopover() {
  const onWheel = (e: ReactWheelEvent<HTMLDivElement>) => {
    e.stopPropagation();
  };
  return (
    <div
      id="ha-tip"
      className="hidden fixed z-50 max-w-md max-h-[70vh] overflow-x-hidden overflow-y-auto px-3 py-2 bg-white border border-gray-200 rounded-lg shadow-lg text-11 mono"
      style={{ pointerEvents: "auto", userSelect: "text", lineHeight: 1.55, overscrollBehavior: "contain" }}
      onMouseEnter={cancelHideTip}
      onMouseLeave={scheduleHideTip}
      onWheel={onWheel}
    />
  );
}

/** Hover handlers that show `html` at the cursor; none when there is no html.
 *  The card stays put after enter so the pointer can move onto it to scroll. */
export function tipHandlers(html: string | null | undefined) {
  if (!html) return {};
  return {
    onMouseEnter: (e: ReactMouseEvent) => showTipAt(e.clientX, e.clientY, html),
    onMouseLeave: scheduleHideTip,
  };
}
