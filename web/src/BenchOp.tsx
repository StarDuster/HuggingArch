/** A measured row's op, as Kernel Bench names it everywhere: the physical op
 *  (`signature`) its envelope, filters and charts key on, and — where the
 *  model's graph calls it something else (a fused `flash_attention` timing a
 *  `gqa_attention` shape) — that name beneath, so the two never read as two
 *  different ops. */
export function BenchOp({ signature, op }: { signature?: string | null; op?: string | null }) {
  const name = signature || op || "—";
  return (
    <>
      {name}
      {op && signature && op !== signature
        ? <span className="block text-10 text-gray-400" title="The op as the model's graph calls it">as {op}</span>
        : null}
    </>
  );
}
