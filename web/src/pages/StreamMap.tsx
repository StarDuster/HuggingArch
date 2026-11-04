/** One inner stream across a card's layers (`stream_map` on the card): per
 *  layer that writes or reads it, whether it writes it and whose write it
 *  reads — a layer index, or the name of the stream's seed. */
export type StreamRow = {
  name: string;
  cells: { layer: number; writes: boolean; from?: number | string }[];
};

/** The colours a kind card's variants are told apart by, here and on their
 *  rows. */
export const VARIANT_COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#64748b"];

const COL = 12;
const LABEL = 90;
const ROW = 24;
const BAND = 10;

/** The layer-to-layer reads inside one card: a column per layer, a row per
 *  stream; a filled dot writes the stream, a hollow one only reads it, and
 *  an arc joins each reader to the layer whose write it reads. `bands`
 *  colours each layer by the variant that runs it. */
export function StreamMap({ layers, rows, bands }: {
  layers: number[];
  rows: StreamRow[];
  bands?: Record<number, number>;
}) {
  const xs = new Map(layers.map((l, i) => [l, LABEL + i * COL + COL / 2]));
  const top = bands ? BAND + 14 : 14;
  const height = top + rows.length * ROW + 4;
  const width = LABEL + layers.length * COL + 4;
  const ticks = layers.filter((l, i) => i === 0 || i === layers.length - 1 || l % 5 === 0);
  // Drawn at its own size, a column a layer: a card of many layers scrolls
  // rather than squeezing its columns below a dot's width.
  return (
    <div style={{ overflowX: "auto", marginTop: 10 }}>
    <svg
      className="arch-stream-map"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      style={{ display: "block" }}
      data-stream-map
    >
      {bands ? layers.map((l) => (
        <rect key={`b${l}`} x={xs.get(l)! - COL / 2 + 0.5} y={0} width={COL - 1} height={BAND} rx={1.5}
          fill={VARIANT_COLORS[(bands[l] ?? 0) % VARIANT_COLORS.length]} opacity={0.75}>
          <title>{`layer ${l}`}</title>
        </rect>
      )) : null}
      {ticks.map((l) => (
        <text key={`t${l}`} x={xs.get(l)} y={top - 3} textAnchor="middle" className="arch-stream-tick">{l}</text>
      ))}
      <text x={0} y={top - 3} className="arch-stream-tick">layer</text>
      {rows.map((row, r) => {
        const y = top + r * ROW + ROW - 7;
        return (
          <g key={row.name} data-stream-row={row.name}>
            <text x={0} y={y + 3} className="arch-stream-label">{row.name}</text>
            <line x1={LABEL - 2} x2={width - 2} y1={y} y2={y} stroke="#e2e8f0" />
            {row.cells.map((c) => {
              const x = xs.get(c.layer);
              if (x === undefined) return null;
              const src = typeof c.from === "number" ? xs.get(c.from) : undefined;
              const lift = src !== undefined ? Math.min(ROW - 8, 3 + Math.abs(x - src) / COL * 1.5) : 0;
              const what = `layer ${c.layer}${c.writes ? ` writes ${row.name}` : ""}${
                c.from !== undefined ? `${c.writes ? "," : ""} reads it from ${typeof c.from === "number" ? `layer ${c.from}` : c.from}` : ""}`;
              return (
                <g key={c.layer}>
                  {src !== undefined ? (
                    <path d={`M${src},${y - 3} Q${(src + x) / 2},${y - 3 - 2 * lift} ${x},${y - 3}`}
                      fill="none" stroke="#d97706" strokeWidth={0.8} opacity={0.7} />
                  ) : c.from !== undefined ? (
                    // Read from outside this card (a seed, an earlier card).
                    <path d={`M${x - COL / 2},${y - 9} L${x},${y - 3}`} stroke="#d97706" strokeWidth={0.8} strokeDasharray="2 1.5" />
                  ) : null}
                  <circle cx={x} cy={y} r={3}
                    fill={c.writes ? "#d97706" : "#ffffff"} stroke="#d97706" strokeWidth={1.2}>
                    <title>{what}</title>
                  </circle>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
    </div>
  );
}
