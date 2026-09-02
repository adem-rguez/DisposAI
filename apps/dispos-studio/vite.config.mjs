import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    modulePreload: false,
    // Electron loads this build through file:// while the launcher watches
    // for rebuilds. Stable filenames avoid deleting the asset an open window
    // is still executing after a rebuild.
    rollupOptions: {
      input: {
        main: 'index.html',
        wizard: 'wizard.html',
      },
      output: {
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
