// Geometric projection of the same tracked joint centres for devices without WebGL.
// This view uses centre lines rather than presenting them as anatomical surfaces.
export function createLandmarkView(host, singleColor) {
  const canvas = document.createElement('canvas');
  canvas.className = 'body-stage-canvas';
  canvas.setAttribute('aria-label', 'Projection of body joint centres');
  canvas.style.cssText = 'display:block;width:100%;height:100%;background:#283339';
  const context = canvas.getContext('2d');
  if (!context) throw new Error('No anatomy rendering context is available');
  const note = document.createElement('span');
  note.textContent = 'Joint-centre projection';
  note.style.cssText =
    'position:absolute;bottom:5px;left:8px;color:#cfdae0;font:10px sans-serif;pointer-events:none';
  host.append(canvas, note);
  return {
    canvas,
    draw(frames, view, region, height) {
      const width = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
      const dpr = Math.min(devicePixelRatio || 1, 1.5);
      const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(h * dpr);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, h);
      const w = width / frames.length;
      const yaw = { front: 0, back: Math.PI, side: Math.PI / 2, oblique: 0.62 }[view] ?? 0.62;
      const scale = Math.min(
        h / (region === 'body' ? height * 1.25 : height * .61),
        w / (region === 'body' ? 1.06 : .7),
      );
      let massMarkers = null;
      frames.forEach((frame, i) => {
        const pose = frame.segments;
        const v = (name) => pose[name].position;
        const center = region === 'hip' ? v('femur') : region === 'knee' ? v('tibia') : [0, height * .55, 0];
        function point(p) {
          return [
            w * (i + .5) + ((p[0] - center[0]) * Math.sin(yaw) - (p[2] - center[2]) * Math.cos(yaw)) * scale,
            h * .52 - (p[1] - center[1]) * scale,
          ];
        }
        if (frame.massMarkers && frames.length === 1) {
          massMarkers = { body: point(frame.massMarkers.body), pelvis: point(frame.massMarkers.pelvis) };
        }
        function line(a, b, color, weight = 2) {
          const [x, y] = point(a), [x2, y2] = point(b);
          context.strokeStyle = color;
          context.lineWidth = weight;
          context.beginPath();
          context.moveTo(x, y);
          context.lineTo(x2, y2);
          context.stroke();
        }
        context.save();
        context.beginPath();
        context.rect(w * i, 0, w, h);
        context.clip();
        const main = i === 1 ? '#e5ac8b' : singleColor ?? '#a6c5d9';
        const edge = '#9aafb6';
        for (const [leg, arm] of [['', '_r'], ['_l', '_l']]) {
          line(v('femur' + leg), v('tibia' + leg), main, 4);
          line(v('tibia' + leg), v('talus' + leg), main, 3);
          line(v('talus' + leg), v('toes' + leg), edge, 3);
          line(v('humerus' + arm), v('ulna' + arm), edge, 3);
          line(v('ulna' + arm), v('hand' + arm), edge, 2);
          line(v('femur' + leg), v('humerus' + arm), main, 3);
        }
        line(v('humerus_r'), v('humerus_l'), main, 3);
        line(v('femur'), v('femur_l'), main, 3);
        const shoulders = v('humerus_r').map((a, j) => (a + v('humerus_l')[j]) / 2);
        const head = frame.headPosition ?? [shoulders[0], shoulders[1] + height * .115, shoulders[2]];
        line(shoulders, head, main, 2);
        const [x, y] = point(head);
        context.strokeStyle = main;
        context.lineWidth = 2;
        context.beginPath();
        context.ellipse(x, y, scale * .055, scale * .078, 0, 0, Math.PI * 2);
        context.stroke();
        for (const key of ['femur', 'femur_l', 'tibia', 'tibia_l', 'talus', 'talus_l', 'humerus_r', 'humerus_l']) {
          const [x, y] = point(v(key));
          context.fillStyle = main;
          context.beginPath();
          context.arc(x, y, 3, 0, Math.PI * 2);
          context.fill();
        }
        context.restore();
      });
      return massMarkers;
    },
    dispose() {
      canvas.remove();
      note.remove();
    },
  };
}
