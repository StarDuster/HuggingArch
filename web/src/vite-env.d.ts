/// <reference types="vite/client" />

declare module "elkjs/lib/elk.bundled.js" {
  import ElkConstructor from "elkjs";
  export default ElkConstructor;
}

interface SiteConfig {
  authEnabled?: boolean;
  supabase?: { url?: string; anon_key?: string };
}

interface Window {
  SITE_CONFIG?: SiteConfig;
}
