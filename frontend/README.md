# MemeticWorld frontend

Next.js 16 (App Router) + TypeScript + Tailwind v4. See the repo-root [README](../README.md) and
[CLAUDE.md](../CLAUDE.md).

```bash
npm install
npm run dev     # http://localhost:3000
```

The backend URL defaults to `http://localhost:8000`; override with `NEXT_PUBLIC_API_URL` in
`frontend/.env.local`. Everything on screen is derived from the backend's event log
(`src/lib/replay.ts`); components live in `src/components/`.
