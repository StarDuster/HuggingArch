import { useEffect, useRef, useState } from "react";

export type ComboOption = { value: string; label: string };

/** A small searchable dropdown — the native <select> popover is unusable on
 *  mobile and unstyled everywhere. Button + popover with a filter input when
 *  the option list is long; Esc / outside click closes. */
export function ComboBox({
  value, options, onChange, placeholder = "—", className = "", buttonClassName = "",
  allowCustom = false,
}: {
  value: string;
  options: ComboOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  buttonClassName?: string;
  /** Type a value that is not in `options` (endpoints with no /models catalog). */
  allowCustom?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    if (open) {
      setQuery(allowCustom ? value : "");
      if (allowCustom || options.length > 8) searchRef.current?.focus();
    }
  }, [open, options.length, allowCustom, value]);

  const current = options.find((o) => o.value === value);
  const q = query.trim().toLowerCase();
  const shown = q ? options.filter((o) => o.label.toLowerCase().includes(q) || o.value.toLowerCase().includes(q)) : options;
  const typed = query.trim();
  const custom = allowCustom && typed && !options.some((o) => o.value === typed);

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        className={`w-full flex items-center justify-between gap-2 px-2 py-1.5 text-xs border border-gray-200 rounded bg-white hover:border-gray-300 focus:outline-none focus:border-amber-500 ${buttonClassName}`}
        onClick={() => setOpen(!open)}
      >
        <span className={`truncate ${current || (allowCustom && value) ? "text-gray-800" : "text-gray-400"}`}>
          {current ? current.label : (allowCustom && value ? value : placeholder)}
        </span>
        <svg className={`w-3 h-3 shrink-0 text-gray-400 transition-transform ${open ? "rotate-180" : ""}`} viewBox="0 0 12 12" fill="none">
          <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="absolute z-50 mt-1 w-full min-w-[12rem] bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
          {(allowCustom || options.length > 8) && (
            <div className="p-1.5 border-b border-gray-100">
              <input
                ref={searchRef}
                className="w-full px-2 py-1 text-xs border border-gray-200 rounded focus:outline-none focus:border-amber-500 font-mono"
                placeholder={allowCustom ? "type a model id…" : "Filter…"}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && allowCustom && typed) {
                    e.preventDefault();
                    onChange(typed);
                    setOpen(false);
                  }
                }}
              />
            </div>
          )}
          <div className="max-h-60 overflow-y-auto py-1">
            {shown.map((o) => (
              <button
                key={o.value || "__default"}
                type="button"
                className={`w-full text-left px-3 py-2 text-xs truncate hover:bg-amber-50 ${o.value === value ? "text-amber-700 font-medium bg-amber-50/60" : "text-gray-700"}`}
                onClick={() => { onChange(o.value); setOpen(false); }}
              >
                {o.label}
              </button>
            ))}
            {custom ? (
              <button
                type="button"
                className="w-full text-left px-3 py-2 text-xs hover:bg-amber-50 text-amber-800"
                onClick={() => { onChange(typed); setOpen(false); }}
              >
                Use <span className="font-mono">{typed}</span>
              </button>
            ) : null}
            {!shown.length && !custom && (
              <div className="px-3 py-2 text-xs text-gray-400">
                {allowCustom ? "type a model id and press Enter" : "no matches"}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
