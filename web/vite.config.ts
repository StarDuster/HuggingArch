import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/** Start the critical Latin UI face before the CSS parser discovers it. */
function preloadCriticalFont(): Plugin {
  return {
    name: "preload-critical-source-sans",
    apply: "build",
    enforce: "post",
    generateBundle(_options, bundle) {
      const html = bundle["index.html"];
      if (html?.type !== "asset") return;
      const font = Object.values(bundle).find(
        (asset) =>
          asset.type === "asset" &&
          /^web-assets\/source-sans-3-latin-wght-normal-[^.]+\.woff2$/.test(
            asset.fileName,
          ),
      );
      if (!font) throw new Error("critical Source Sans font was not emitted");
      html.source = String(html.source).replace(
        "</head>",
        `  <link rel="preload" href="/${font.fileName}" as="font" type="font/woff2" crossorigin>\n  </head>`,
      );
    },
  };
}

// Production assets are served by FastAPI at /web-assets. Dev proxies API +
// remaining HTML pages to FastAPI. Top-tab HTML paths, /privacy, /compare.html,
// and /account/* /admin /auth/*.html are React routes.
export default defineConfig({
  plugins: [preloadCriticalFont(), react(), tailwindcss()],
  build: {
    outDir: "dist",
    assetsDir: "web-assets",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // React/router into a stable vendor chunk that caches across deploys.
        // elkjs is intentionally NOT split out: it imports the bare specifier
        // `web-worker`, which fails to resolve if forced into its own chunk —
        // route-level lazy loading already keeps elkjs out of the landing
        // bundle (it rides in the inspector/fusion route chunks that use it).
        manualChunks: {
          "vendor-react": ["react", "react-dom", "react-router-dom"],
          "vendor-echarts": ["echarts", "echarts-for-react"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/gallery": "http://127.0.0.1:8000",
      "/supported-models": "http://127.0.0.1:8000",
      "/analyze": "http://127.0.0.1:8000",
      "/compare": {
        target: "http://127.0.0.1:8000",
        bypass(req) {
          const url = req.url || "";
          if (url.startsWith("/compare.html")) return url;
        },
      },
      "/dim-glossary": "http://127.0.0.1:8000",
      "/generate-spec": "http://127.0.0.1:8000",
      "/refine-spec": "http://127.0.0.1:8000",
      "/spec-status": "http://127.0.0.1:8000",
      "/stop-spec": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
      "/inference": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/static": "http://127.0.0.1:8000",
      "/site-config.js": "http://127.0.0.1:8000",
      "/analytics": "http://127.0.0.1:8000",
      "/fusion/registry": "http://127.0.0.1:8000",
      "/fusion/fired": "http://127.0.0.1:8000",
      "/auth": {
        target: "http://127.0.0.1:8000",
        bypass(req) {
          const path = (req.url || "").split("?")[0];
          if (path === "/auth/login.html" || path === "/auth/verify.html" || path === "/auth/reset-password.html") return req.url;
        },
      },
      "/account": {
        target: "http://127.0.0.1:8000",
        bypass(req) {
          const url = req.url || "";
          if (url.startsWith("/account/profile.html") || url.startsWith("/account/workspace.html") || url.startsWith("/account/my-benchmarks.html") || url.startsWith("/account/bench-review.html")) return url;
        },
      },
      "/admin": {
        target: "http://127.0.0.1:8000",
        bypass(req) {
          const path = (req.url || "").split("?")[0];
          if (path === "/admin" || path === "/admin/" || path.startsWith("/admin/index.html")) return req.url;
        },
      },
    },
  },
});
