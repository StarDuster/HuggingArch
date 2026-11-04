import { lazy, Suspense, useEffect, type ReactNode } from "react";
import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";
import { isGalleryHash } from "./galleryHash";
import { ErrorBoundary } from "./ErrorBoundary";
import { DialogHost } from "./Dialog";
import { SiteFooter } from "./SiteFooter";
import { TipPopover } from "./Tip";
import { TopNav } from "./TopNav";

// Route-level code splitting: each page is its own chunk, fetched on first
// navigation, so the initial bundle carries only the shell + the landing page's
// dependencies instead of every page's (the 2 MB single-bundle cold start).
// Pages use named exports, so the dynamic import is adapted to a default.
const Gallery = lazy(() => import("./pages/Gallery").then((m) => ({ default: m.Gallery })));
const Inference = lazy(() => import("./pages/Inference").then((m) => ({ default: m.Inference })));
const Corpus = lazy(() => import("./pages/Corpus").then((m) => ({ default: m.Corpus })));
const Fusion = lazy(() => import("./pages/Fusion").then((m) => ({ default: m.Fusion })));
const Inspector = lazy(() => import("./pages/Inspector").then((m) => ({ default: m.Inspector })));
const Playground = lazy(() => import("./pages/Playground").then((m) => ({ default: m.Playground })));
const Privacy = lazy(() => import("./pages/Privacy").then((m) => ({ default: m.Privacy })));
const CompareRedirect = lazy(() => import("./pages/CompareRedirect").then((m) => ({ default: m.CompareRedirect })));
const AccountProfile = lazy(() => import("./pages/AccountProfile").then((m) => ({ default: m.AccountProfile })));
const AccountWorkspace = lazy(() => import("./pages/AccountWorkspace").then((m) => ({ default: m.AccountWorkspace })));
const BenchReview = lazy(() => import("./pages/BenchReview").then((m) => ({ default: m.BenchReview })));
const Admin = lazy(() => import("./pages/Admin").then((m) => ({ default: m.Admin })));
const AuthLogin = lazy(() => import("./pages/AuthLogin").then((m) => ({ default: m.AuthLogin })));
const AuthVerify = lazy(() => import("./pages/AuthVerify").then((m) => ({ default: m.AuthVerify })));
const AuthReset = lazy(() => import("./pages/AuthReset").then((m) => ({ default: m.AuthReset })));

/** Any URL no page answers. The server sends this same app with status 404
 *  for such a URL, so the address stays what the reader typed. */
function NotFound() {
  return (
    <main className="flex-1 page-shell py-16 text-center" data-not-found>
      <p className="text-sm font-semibold text-amber-600">404</p>
      <h1 className="mt-2 text-2xl font-semibold text-gray-900">Page not found</h1>
      <p className="mt-2 text-sm text-gray-500">
        Nothing lives at <code className="font-mono text-gray-700">{window.location.pathname}</code>.
      </p>
      <Link to="/" className="inline-block mt-6 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium">
        Go to the Inspector
      </Link>
    </main>
  );
}

/** `/`: the Inspector. A Gallery link from before the Inspector took `/`
 *  (`/#gpu=…`) goes on to the Gallery's own URL with its view. */
function Landing() {
  const { hash } = window.location;
  if (isGalleryHash(hash)) return <Navigate to={`/gallery.html${hash}`} replace />;
  return <Shell active="inspector" title={null}><Inspector /></Shell>;
}

function Pending() {
  return <div className="flex-1 text-xs text-gray-400 text-center py-10">Loading…</div>;
}

/** The browser tab's title: the page's name, then the site's. The landing
 *  page keeps the site's own title (`web/index.html`). */
const SITE_TITLE = "HuggingArch — understand any Hugging Face model before you deploy it";

function usePageTitle(page: string | null) {
  useEffect(() => {
    document.title = page ? `${page} · HuggingArch` : SITE_TITLE;
  }, [page]);
}

function Shell({
  active,
  title,
  children,
}: {
  active: "gallery" | "inspector" | "inference" | "fusion" | "corpus" | "playground" | null;
  /** Page name for the browser tab; null on the landing page. */
  title: string | null;
  children: ReactNode;
}) {
  usePageTitle(title);
  return (
    <div className="bg-gray-50 min-h-screen flex flex-col">
      <TopNav active={active} />
      <ErrorBoundary>
        <Suspense fallback={<Pending />}>{children}</Suspense>
      </ErrorBoundary>
      <SiteFooter />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <Suspense fallback={<Pending />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/gallery.html" element={<Shell active="gallery" title="Gallery"><Gallery /></Shell>} />
          <Route path="/inference.html" element={<Shell active="inference" title="Inference"><Inference /></Shell>} />
          <Route path="/inspector.html" element={<Shell active="inspector" title="Inspector"><Inspector /></Shell>} />
          <Route path="/playground.html" element={<Shell active="playground" title="Playground"><Playground /></Shell>} />
          <Route path="/fusion.html" element={<Shell active="fusion" title="Fusion"><Fusion /></Shell>} />
          <Route path="/kernel-corpus.html" element={<Shell active="corpus" title="Kernel Bench"><Corpus /></Shell>} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/compare.html" element={<CompareRedirect />} />
          <Route path="/account/profile.html" element={<AccountProfile />} />
          <Route path="/account/workspace.html" element={<AccountWorkspace />} />
          <Route path="/account/bench-review.html" element={<BenchReview />} />
          {/* Retired benchmarks page — folded into the workspace page's #benchmarks panel. */}
          <Route path="/account/my-benchmarks.html" element={<Navigate to="/account/workspace.html#benchmarks" replace />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/admin/" element={<Admin />} />
          <Route path="/admin/index.html" element={<Admin />} />
          <Route path="/auth/login.html" element={<AuthLogin />} />
          <Route path="/auth/verify.html" element={<AuthVerify />} />
          <Route path="/auth/reset-password.html" element={<AuthReset />} />
          <Route path="*" element={<Shell active={null} title="Page not found"><NotFound /></Shell>} />
        </Routes>
      </Suspense>
      </ErrorBoundary>
      <TipPopover />
      <DialogHost />
    </BrowserRouter>
  );
}
