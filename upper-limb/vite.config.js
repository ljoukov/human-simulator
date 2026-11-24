import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  resolve: { dedupe: ['three'] },
  build: {
    modulePreload: { polyfill: false },
    rollupOptions: {
      input: {
        atlas: fileURLToPath(new URL('./index.html', import.meta.url)),
        presentation: fileURLToPath(new URL('./presentation/index.html', import.meta.url)),
        deterioration: fileURLToPath(new URL('./deterioration/index.html', import.meta.url)),
        kneeLoading: fileURLToPath(new URL('./knee-loading/index.html', import.meta.url)),
        hipStudy: fileURLToPath(new URL('./hip-study/index.html', import.meta.url)),
      },
    },
  },
  server: {
    fs: {
      allow: [
        fileURLToPath(new URL('./', import.meta.url)),
        fileURLToPath(new URL('../lower-limb/', import.meta.url)),
        fileURLToPath(new URL('../knee-loading/', import.meta.url)),
        fileURLToPath(new URL('../hip-study/', import.meta.url)),
        fileURLToPath(new URL('../whole-body-study/', import.meta.url)),
      ],
    },
  },
});
