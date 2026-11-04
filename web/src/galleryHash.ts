/** The hash keys a Gallery link carries (`pages/Gallery.tsx`): its card,
 *  decode context, KV precision and draft. Its own module so the router can
 *  recognise a Gallery link without loading the Gallery chunk. */
export const GALLERY_HASH_KEYS = ["gpu", "ctx", "kv", "k", "alpha"] as const;

/** Whether `hash` is a Gallery view: Gallery links lived at `/` before the
 *  Inspector took it, and they still land there. */
export function isGalleryHash(hash: string): boolean {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  return GALLERY_HASH_KEYS.some((k) => params.has(k));
}
