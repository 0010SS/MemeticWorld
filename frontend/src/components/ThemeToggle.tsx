"use client";

import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";
const KEY = "memeticworld-theme";

function apply(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    try {
      const saved = localStorage.getItem(KEY) as Theme | null;
      if (saved) {
        setTheme(saved);
        apply(saved);
      }
    } catch {
      /* storage unavailable */
    }
  }, []);

  const next = () => {
    const order: Theme[] = ["system", "light", "dark"];
    const value = order[(order.indexOf(theme) + 1) % order.length];
    setTheme(value);
    apply(value);
    try {
      localStorage.setItem(KEY, value);
    } catch {
      /* storage unavailable */
    }
  };

  return (
    <button onClick={next} className="rounded-md border border-line px-2 py-1.5 text-xs text-ink-2 hover:bg-surface-2" title="Theme">
      {theme === "system" ? "◐ Auto" : theme === "light" ? "☀ Light" : "☾ Dark"}
    </button>
  );
}
