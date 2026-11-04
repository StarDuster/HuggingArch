import { useEffect, useRef, useState } from "react";

/** True once the element has come within a screen of the viewport, and from
 *  then on. A page of many charts (Kernel Bench draws two per curve group)
 *  would otherwise build every option and ECharts instance on load and hold
 *  the main thread for seconds; off screen, the caller draws a placeholder of
 *  the same height. */
export function useNearViewport() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [near, setNear] = useState(typeof IntersectionObserver === "undefined");
  useEffect(() => {
    if (near || !ref.current) return;
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) setNear(true);
    }, { rootMargin: "100% 0px" });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [near]);
  return [ref, near] as const;
}
