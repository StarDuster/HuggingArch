// DimSymbol — a dim symbol whose meaning/binding rides a hover tooltip
// (`#ha-tip`) instead of an inline paragraph. Long glossary prose under every
// input made panels unreadable (the old Gaps panel); the dotted underline
// marks it as hoverable, same convention as the Inspector's formula tooltips.
import { escapeHtml } from "./format";
import { tipHandlers } from "./Tip";

export function DimSymbol({ symbol, meaning, binding, className }: {
  symbol: string;
  meaning?: string;
  binding?: string;
  className?: string;
}) {
  const html = [
    meaning ? `<div class="text-gray-700">${escapeHtml(meaning)}</div>` : "",
    binding ? `<div class="mt-1 text-10 text-gray-400 font-mono">${escapeHtml(binding)}</div>` : "",
  ].filter(Boolean).join("");
  const hoverable = !!(meaning || binding);
  return (
    <span
      {...tipHandlers(html)}
      className={className ?? "font-mono text-gray-800"}
      style={hoverable ? { textDecoration: "underline dotted" } : undefined}
    >
      {symbol}
    </span>
  );
}
