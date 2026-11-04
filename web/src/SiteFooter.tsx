/** Visible Privacy + GitHub links. OAuth review needs a real in-flow Privacy href. */

const PRIVACY = "https://www.huggingarch.com/privacy";
const GITHUB = "https://github.com/shenh10/HuggingArch";

export function SiteFooter() {
  return (
    <footer className="border-t border-gray-200 bg-white/40">
      <div className="page-shell py-3 flex items-center justify-between text-11 text-gray-400">
        <div>
          <span className="text-gray-500 font-medium">HuggingArch</span>
          <span className="mx-2 text-gray-300">·</span>
          © <span id="footer-year">{new Date().getFullYear()}</span> All rights reserved
        </div>
        <div className="flex items-center gap-3">
          <a href={PRIVACY} className="text-gray-500 hover:text-amber-600">Privacy</a>
          <a
            href={GITHUB}
            className="text-gray-500 hover:text-amber-600 inline-flex items-center gap-1"
            target="_blank"
            rel="noopener"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
