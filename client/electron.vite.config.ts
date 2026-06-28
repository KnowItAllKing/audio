import { defineConfig } from "electron-vite";
import { resolve } from "node:path";

export default defineConfig({
  main: {
    build: {
      lib: {
        entry: resolve(__dirname, "src/main/index.ts")
      }
    }
  },
  preload: {
    build: {
      outDir: resolve(__dirname, "out/preload"),
      lib: {
        entry: resolve(__dirname, "src/preload/index.ts")
      },
      rollupOptions: {
        output: {
          // Preload scripts are most compatible as CommonJS.
          format: "cjs",
          entryFileNames: "[name].cjs"
        }
      }
    }
  },
  renderer: {
    root: resolve(__dirname, "src/renderer"),
    build: {
      // Keep renderer output under `out/` so `src/main/index.ts` can load
      // `../renderer/index.html` relative to `out/main`.
      outDir: resolve(__dirname, "out/renderer")
    }
  }
});
