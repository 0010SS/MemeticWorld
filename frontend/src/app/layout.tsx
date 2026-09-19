import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "MemeticWorld",
  description: "A multi-agent campus simulation for watching memes emerge from ordinary conversation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">{children}</body>
    </html>
  );
}
