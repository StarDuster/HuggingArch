import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Small markdown renderer with Tailwind styling — Tailwind's preflight strips
 *  default element styles, so every tag gets an explicit class here. Used for
 *  agent-authored text (review reports) that is already markdown. */
export function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="text-sm font-semibold text-gray-800 mt-3 mb-1" {...p} />,
          h2: (p) => <h2 className="text-13 font-semibold text-gray-800 mt-3 mb-1" {...p} />,
          h3: (p) => <h3 className="text-xs font-semibold text-gray-800 mt-3 mb-1" {...p} />,
          h4: (p) => <h4 className="text-xs font-semibold text-gray-700 mt-2 mb-1" {...p} />,
          p: (p) => <p className="my-1.5" {...p} />,
          ul: (p) => <ul className="list-disc pl-5 my-1.5 space-y-0.5" {...p} />,
          ol: (p) => <ol className="list-decimal pl-5 my-1.5 space-y-0.5" {...p} />,
          li: (p) => <li className="pl-0.5" {...p} />,
          a: (p) => <a className="text-blue-600 underline" target="_blank" rel="noreferrer" {...p} />,
          blockquote: (p) => <blockquote className="border-l-2 border-gray-300 pl-3 my-1.5 text-gray-500" {...p} />,
          hr: () => <hr className="my-3 border-gray-200" />,
          code: ({ className: codeClass, children, ...p }) =>
            codeClass ? (
              <code className={`${codeClass} text-11`} {...p}>{children}</code>
            ) : (
              <code className="bg-gray-100 text-gray-800 rounded px-1 py-0.5 text-11" {...p}>{children}</code>
            ),
          pre: (p) => <pre className="bg-gray-900 text-gray-100 rounded-md p-2.5 my-1.5 overflow-auto text-11 leading-snug" {...p} />,
          table: (p) => (
            <div className="my-1.5 overflow-auto">
              <table className="border-collapse text-11" {...p} />
            </div>
          ),
          th: (p) => <th className="border border-gray-200 bg-gray-50 px-2 py-1 text-left font-medium" {...p} />,
          td: (p) => <td className="border border-gray-200 px-2 py-1 align-top" {...p} />,
          strong: (p) => <strong className="font-semibold" {...p} />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** Markdown text (e.g. an agent review report) rendered by default, with a
 *  corner toggle to see the raw source. */
export function MarkdownReport({ text }: { text: string }) {
  const [raw, setRaw] = useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        className="absolute top-2 right-2 z-10 px-1.5 py-0.5 rounded border border-gray-200 bg-white text-10 text-gray-500 hover:text-gray-700"
        title={raw ? "Render as markdown" : "View raw markdown source"}
        onClick={() => setRaw((v) => !v)}
      >
        {raw ? "Markdown" : "Raw"}
      </button>
      {raw ? (
        <pre className="text-11 whitespace-pre-wrap text-gray-700 p-3 max-h-72 overflow-auto m-0">{text}</pre>
      ) : (
        <Markdown text={text} className="text-11 text-gray-700 p-3 max-h-96 overflow-auto" />
      )}
    </div>
  );
}
