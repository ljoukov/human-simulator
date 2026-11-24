// The marker stays at its projected study coordinate. Only the text and the
// leader elbows move into the margins, including when the two points coincide.
export function createMassMarkerOverlay(host) {
  const ns = 'http://www.w3.org/2000/svg';
  const overlay = document.createElementNS(ns, 'svg');
  overlay.classList.add('body-stage-mass-markers');
  overlay.setAttribute('role', 'img');
  overlay.setAttribute('aria-label', 'Body center of mass and pelvis origin on the recorded stride');
  overlay.style.cssText =
    'position:absolute;inset:0;pointer-events:none;width:100%;height:100%;overflow:hidden;z-index:1';
  host.append(overlay);

  function add(parent, tag, attributes) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
    parent.append(node);
    return node;
  }
  function annotation(label, color, ring) {
    const group = add(overlay, 'g', { 'data-mass-marker': ring ? 'body' : 'pelvis' });
    const leader = add(group, 'polyline', { fill: 'none', stroke: color, 'stroke-width': 1.2 });
    if (!ring) leader.setAttribute('stroke-dasharray', '3 2');
    const text = add(group, 'text', {
      fill: color,
      'font-family': 'system-ui, sans-serif',
      'font-weight': 600,
      'text-anchor': ring ? 'end' : 'start',
      'dominant-baseline': 'middle',
      stroke: '#283339',
      'stroke-width': 4,
      'stroke-linejoin': 'round',
      'paint-order': 'stroke',
    });
    text.textContent = label;
    const mark = add(group, 'g', {});
    if (ring) {
      add(mark, 'circle', { r: 5.5, fill: 'none', stroke: '#283339', 'stroke-width': 4.5 });
      add(mark, 'circle', { r: 5.5, fill: 'none', stroke: color, 'stroke-width': 2.5 });
    } else {
      add(mark, 'circle', { r: 3.5, fill: color, stroke: '#283339', 'stroke-width': 1.4 });
    }
    return (point, width, height) => {
      const [x, y] = point;
      const visible = x >= 0 && x <= width && y >= 0 && y <= height;
      group.style.display = visible ? 'inline' : 'none';
      if (!visible) return;
      const compact = width < 430;
      const labelX = ring
        ? Math.min(Math.max(compact ? 119 : 124, width * .28), width * .46)
        : width - Math.min(Math.max(compact ? 80 : 95, width * .25), width * .38);
      const labelY = Math.min(height - 20, Math.max(22, y + (ring ? -1 : 1) * (compact ? 24 : 32)));
      const towards = ring ? 1 : -1;
      const lineStart = labelX + 6 * towards;
      const elbow = lineStart + (x - lineStart) * .52;
      leader.setAttribute(
        'points',
        [[lineStart, labelY], [elbow, labelY], [x, y]].map((p) => p.join(',')).join(' '),
      );
      text.setAttribute('x', labelX);
      text.setAttribute('y', labelY);
      text.setAttribute('font-size', compact ? 10.5 : 12.5);
      mark.setAttribute('transform', 'translate(' + x + ' ' + y + ')');
    };
  }
  const body = annotation('Body center of mass', '#f3b68f', true);
  const pelvis = annotation('Pelvis origin', '#e0e5e5', false);
  return {
    show(points, width, height) {
      overlay.style.display = 'block';
      body(points.body, width, height);
      pelvis(points.pelvis, width, height);
    },
    hide() {
      overlay.style.display = 'none';
    },
    dispose() {
      overlay.remove();
    },
  };
}
