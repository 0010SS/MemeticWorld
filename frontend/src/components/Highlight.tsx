import { useMemo } from "react";

function escape(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Build one case-insensitive regex that matches any of the phrases, tolerant of punctuation between words. */
export function phraseRegex(phrases: string[]): RegExp | null {
  const parts = phrases
    .filter(Boolean)
    .sort((a, b) => b.length - a.length)
    .map((p) => p.split(" ").map(escape).join("[^a-z0-9]+"));
  if (!parts.length) return null;
  return new RegExp(`(?<![a-z0-9'-])(${parts.join("|")})(?![a-z0-9'-])`, "gi");
}

/** Renders text with detected meme phrases marked. */
export function Highlight({ text, regex, strong }: { text: string; regex: RegExp | null; strong?: RegExp | null }) {
  const pieces = useMemo(() => {
    if (!regex) return [text];
    const out: (string | { match: string; key: number })[] = [];
    let last = 0;
    regex.lastIndex = 0;
    for (const m of text.matchAll(regex)) {
      const start = m.index ?? 0;
      if (start > last) out.push(text.slice(last, start));
      out.push({ match: m[0], key: start });
      last = start + m[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }, [text, regex]);

  return (
    <>
      {pieces.map((p, i) => {
        if (typeof p === "string") return <span key={i}>{p}</span>;
        const isStrong = strong ? new RegExp(strong.source, "i").test(p.match) : false;
        return (
          <mark
            key={i}
            className={`rounded px-0.5 text-mark-ink ${isStrong ? "bg-mark font-semibold ring-1 ring-[var(--mark-ink)]/30" : "bg-mark"}`}
          >
            {p.match}
          </mark>
        );
      })}
    </>
  );
}
