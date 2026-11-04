// A user's avatar image, falling back to their initial on an amber disc when
// there is no URL or the image fails to load. `size` carries the box and the
// initial's text size (e.g. "w-6 h-6 text-xs").
import { useState } from "react";

export function AvatarFace({ url, text, size, id }: {
  url?: string | null;
  text: string;
  size: string;
  id?: string;
}) {
  const [failed, setFailed] = useState(false);
  const initial = (text || "?").trim().slice(0, 1).toUpperCase() || "?";
  if (!url || failed) {
    return (
      <span id={id} className={`${size} rounded-full bg-amber-100 text-amber-700 inline-flex items-center justify-center font-semibold`}>
        {initial}
      </span>
    );
  }
  return (
    <img
      id={id}
      src={url}
      alt=""
      referrerPolicy="no-referrer"
      className={`${size} rounded-full object-cover`}
      onError={() => setFailed(true)}
    />
  );
}
