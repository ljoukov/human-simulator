import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';

const root = fileURLToPath(new URL('../', import.meta.url));
const dist = join(root, 'dist');
// The hosted entry points share Three.js. Build an independent atlas module
// so its downloadable file can still run without the hosted shared chunks.
const standalone = await build({
  root,
  configFile: false,
  logLevel: 'error',
  build: {
    write: false,
    modulePreload: { polyfill: false },
    rollupOptions: {
      input: join(root, 'index.html'),
      output: { inlineDynamicImports: true },
    },
  },
});
const files = (Array.isArray(standalone) ? standalone : [standalone]).flatMap((bundle) => bundle.output);
function bundleText(path) {
  const relativePath = path.replace(/^\/+/, '');
  const file = files.find(({ fileName }) => fileName === relativePath);
  const content = file?.type === 'chunk' ? file.code : file?.source;
  if (typeof content !== 'string') throw new Error(`Missing standalone viewer file ${path}.`);
  return content;
}
let html = bundleText('index.html');
const scriptTag = /<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/;
const stylesheetTag = /<link\b[^>]*\brel="stylesheet"[^>]*\bhref="([^"]+)"[^>]*>/;
const scriptPath = html.match(scriptTag)?.[1];
const stylesheetPath = html.match(stylesheetTag)?.[1];
if (!scriptPath || !stylesheetPath) throw new Error('The Vite bundle did not contain both viewer files.');

const js = bundleText(scriptPath);
const css = bundleText(stylesheetPath);
const [model, legModel, threeLicense] = await Promise.all([
  readFile(join(root, 'assets', 'upper-limb.glb')),
  readFile(join(root, '../lower-limb/assets/lower-limb.glb')).catch((error) => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  }),
  readFile(join(root, 'node_modules', 'three', 'LICENSE'), 'utf8'),
]);
html = html.replace(stylesheetTag, () => '<style>' + css + '</style>');
html = html.replace(scriptTag, () => '<script type="module">' + js.replaceAll('</script', '<\\/script') + '</script>');
html = html.replace(/\s*<a id="(?:knee-loading|hip-study|presentation)-link"[^>]*>[^<]*<\/a>/g, '');
const embed = (id, bytes) => '<script type="application/octet-stream" id="' + id + '">'
  + bytes.toString('base64') + '</script>';
const data = embed('embedded-model', model) + (legModel ? '\n' + embed('embedded-leg-model', legModel) : '');
html = html.replace('</body>', () => data + '\n<!-- Three.js\n' + threeLicense + '\n-->\n</body>');
await writeFile(join(dist, 'upper-limb-viewer.html'), html);
if (legModel) {
  await writeFile(join(dist, 'lower-limb-viewer.html'), html.replace('<body>', '<body data-default-region="legs">'));
}
