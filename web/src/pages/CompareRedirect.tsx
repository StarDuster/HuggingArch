import { Navigate } from "react-router-dom";
import { legacyCompareDest } from "../compareRedirect";

/** Old shareable /compare.html links. No product UI — hash is remapped. */
export function CompareRedirect() {
  return <Navigate to={legacyCompareDest(window.location.hash)} replace />;
}
