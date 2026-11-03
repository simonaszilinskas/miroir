import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package, which serves it under /static/
// (`miroir.web.app`) and ships it to the demo machine (`miroir.web.pagebuild`).
const ROOT = dirname(fileURLToPath(import.meta.url));
const OUT = join(ROOT, "../src/miroir/web/ui");

// Same rule as `source_digest` in src/miroir/web/pagebuild.py: keep the two in step.
const TOP_FILES = ["index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts"];
const FOLDERS = ["src", "public"];
const SKIPPED = ["public/dsfr/"];

function walk(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

function sourceDigest(root: string): string {
  const names = TOP_FILES.filter((n) => existsSync(join(root, n)));
  for (const folder of FOLDERS) {
    for (const path of walk(join(root, folder))) {
      const name = relative(root, path).split(sep).join("/");
      if (!SKIPPED.some((s) => name.startsWith(s)) && !name.endsWith(".DS_Store")) names.push(name);
    }
  }
  // Python sorts str by code point; so does this comparison.
  names.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const hash = createHash("sha256");
  for (const name of names) {
    hash.update(Buffer.concat([Buffer.from(name), Buffer.from([0])]));
    hash.update(Buffer.concat([readFileSync(join(root, name)), Buffer.from([0])]));
  }
  return hash.digest("hex");
}

function stamp(): Plugin {
  return {
    name: "miroir-source-digest",
    apply: "build",
    closeBundle() {
      writeFileSync(join(OUT, "source.sha256"), sourceDigest(ROOT) + "\n");
    },
  };
}

export default defineConfig({
  base: "/static/",
  plugins: [react(), stamp()],
  build: { outDir: OUT, emptyOutDir: true },
  server: {
    // `npm run dev` against `miroir web dev` on port 8000.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
