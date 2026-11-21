import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const dist = join(root, 'dist');
const path = join(dist, 'index.html');
let html = await readFile(path, 'utf8');
const scriptTag = /<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/;
const stylesheetTag = /<link\b[^>]*\brel="stylesheet"[^>]*\bhref="([^"]+)"[^>]*>/;
const scriptPath = html.match(scriptTag)?.[1];
const stylesheetPath = html.match(stylesheetTag)?.[1];
if (!scriptPath || !stylesheetPath) throw new Error('The Vite bundle did not contain both viewer files.');

const [js, css, model, legModel, threeLicense] = await Promise.all([
  readFile(join(dist, scriptPath), 'utf8'),
  readFile(join(dist, stylesheetPath), 'utf8'),
  readFile(join(root, 'assets', 'upper-limb.glb')),
  readFile(join(root, '../lower-limb/assets/lower-limb.glb')).catch((error) => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  }),
  readFile(join(root, 'node_modules', 'three', 'LICENSE'), 'utf8'),
]);
html = html.replace(stylesheetTag, () => '<style>' + css + '</style>');
html = html.replace(scriptTag, () => '<script type="module">' + js.replaceAll('</script', '<\\/script') + '</script>');
const embed = (id, bytes) => '<script type="application/octet-stream" id="' + id + '">'
  + bytes.toString('base64') + '</script>';
const data = embed('embedded-model', model) + (legModel ? '\n' + embed('embedded-leg-model', legModel) : '');
html = html.replace('</body>', () => data + '\n<!-- Three.js\n' + threeLicense + '\n-->\n</body>');
await writeFile(join(dist, 'upper-limb-viewer.html'), html);
if (legModel) {
  await writeFile(join(dist, 'lower-limb-viewer.html'), html.replace('<body>', '<body data-default-region="legs">'));
}
