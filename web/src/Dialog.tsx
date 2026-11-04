import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";

/** In-page replacements for `window.confirm` / `prompt` / `alert`: the same
 *  awaitable question, drawn as the site's own dialog instead of the
 *  browser's. One `DialogHost` (mounted once in App) shows them one at a
 *  time, in the order they were asked. */
type Ask = {
  title: ReactNode;
  message?: ReactNode;
  /** The affirmative button; "OK" unless said. */
  confirmLabel?: string;
  /** An action that destroys something: the button is red. */
  danger?: boolean;
};

/** A checkbox the question carries, answered along with it. */
type Option = { label: ReactNode; checked?: boolean };

type Request = { id: number } & (
  | (Ask & { kind: "confirm"; resolve: (ok: boolean) => void })
  | (Ask & { kind: "choice"; option: Option; resolve: (checked: boolean | null) => void })
  | (Ask & { kind: "prompt"; defaultValue: string; resolve: (value: string | null) => void })
  | (Ask & { kind: "alert"; resolve: () => void }));
type Asked = Request extends infer R ? (R extends Request ? Omit<R, "id"> : never) : never;

let queue: Request[] = [];
let asked = 0;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};
const push = (r: Asked) => {
  queue = [...queue, { ...r, id: ++asked } as Request];
  emit();
};
const settle = () => {
  queue = queue.slice(1);
  emit();
};

/** True when the user confirms, false when they cancel (button, Esc or a
 *  click outside). */
export function confirmDialog(ask: Ask): Promise<boolean> {
  return new Promise((resolve) => push({ ...ask, kind: "confirm", resolve }));
}

/** Confirm with a checkbox: the box's state when confirmed, null when
 *  cancelled. */
export function confirmChoice(ask: Ask & { option: Option }): Promise<boolean | null> {
  return new Promise((resolve) => push({ ...ask, kind: "choice", resolve }));
}

/** The text entered, or null when cancelled. */
export function promptDialog(ask: Ask & { defaultValue?: string }): Promise<string | null> {
  return new Promise((resolve) =>
    push({ ...ask, kind: "prompt", defaultValue: ask.defaultValue ?? "", resolve }));
}

/** Resolves once dismissed. */
export function alertDialog(ask: Ask): Promise<void> {
  return new Promise((resolve) => push({ ...ask, kind: "alert", resolve }));
}

/** A failed action's error, said in the dialog. */
export function alertError(error: unknown): Promise<void> {
  return alertDialog({ title: "That didn't work", message: error instanceof Error ? error.message : String(error) });
}

export function DialogHost() {
  const current = useSyncExternalStore(subscribe, () => queue[0] ?? null);
  return current ? <DialogBox key={current.id} request={current} /> : null;
}

function DialogBox({ request }: { request: Request }) {
  const [value, setValue] = useState(request.kind === "prompt" ? request.defaultValue : "");
  const [checked, setChecked] = useState(request.kind === "choice" ? !!request.option.checked : false);
  const inputRef = useRef<HTMLInputElement>(null);
  const okRef = useRef<HTMLButtonElement>(null);

  const finish = (ok: boolean) => {
    if (request.kind === "confirm") request.resolve(ok);
    else if (request.kind === "choice") request.resolve(ok ? checked : null);
    else if (request.kind === "prompt") request.resolve(ok ? value : null);
    else request.resolve();
    settle();
  };

  // Focus once, on opening: the input (its text selected) or the button.
  useEffect(() => {
    if (request.kind === "prompt") inputRef.current?.select();
    else okRef.current?.focus();
  }, [request.kind]);
  // Re-bound each render so Esc settles with the current input value.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); finish(false); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const okClass = request.danger
    ? "bg-red-600 hover:bg-red-700 text-white"
    : "bg-amber-500 hover:bg-amber-600 text-white";
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-gray-900/30 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) finish(false); }}>
      <form role="dialog" aria-modal="true" data-dialog={request.kind}
        className="w-full max-w-md rounded-2xl bg-white shadow-xl ring-1 ring-gray-900/5 p-5"
        onSubmit={(e) => { e.preventDefault(); finish(true); }}>
        <h2 className="text-base font-semibold text-gray-900">{request.title}</h2>
        {request.message != null && (
          <div className="mt-1.5 text-sm text-gray-600 whitespace-pre-line break-words">{request.message}</div>
        )}
        {request.kind === "choice" && (
          <label className="mt-3 flex items-start gap-2 text-sm text-gray-700 cursor-pointer">
            <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)}
              className="mt-0.5 accent-amber-500" />
            <span>{request.option.label}</span>
          </label>
        )}
        {request.kind === "prompt" && (
          <input ref={inputRef} value={value} onChange={(e) => setValue(e.target.value)}
            className="mt-3 w-full text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400" />
        )}
        <div className="mt-5 flex justify-end gap-2">
          {request.kind !== "alert" && (
            <button type="button" onClick={() => finish(false)}
              className="text-sm px-3.5 py-1.5 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50">
              Cancel
            </button>
          )}
          <button ref={okRef} type="submit"
            className={`text-sm px-3.5 py-1.5 rounded-lg font-medium ${okClass}`}>
            {request.confirmLabel ?? "OK"}
          </button>
        </div>
      </form>
    </div>
  );
}
