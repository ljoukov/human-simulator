// The traces span the complete movement cycle, on a shared millimetre scale.
// The parent supplies the selected and reference readings at the current phase.
export function createDeepTravelReadout(paneNode) {
  const panel = document.createElement('section');
  panel.id = 'deep-travel';
  panel.setAttribute('aria-labelledby', 'deep-travel-title');
  panel.innerHTML = `
    <div class="travel-heading">
      <h4 id="deep-travel-title">Modeled IT band travel</h4>
      <span class="travel-site">Trochanteric bursa</span><span class="travel-unit">mm</span>
    </div>
    <div class="travel-row">
      <dl class="travel-values">
        <div class="travel-selected"><dt></dt><dd id="deep-travel-current">—</dd></div>
        <div class="travel-reference"><dt>100% ref.</dt><dd id="deep-travel-reference">—</dd></div>
        <div class="travel-difference"><dt>Difference</dt><dd id="deep-travel-difference">—</dd></div>
      </dl>
      <span class="travel-unmatched">No matched reference</span>
    </div>
    <svg class="travel-traces" viewBox="0 0 144 44" preserveAspectRatio="none" aria-hidden="true">
      <line class="travel-zero" x1="2" x2="142" y1="41" y2="41"/>
      <path class="travel-selected-trace"/>
      <path class="travel-reference-trace"/>
      <line class="travel-phase" y1="2" y2="41"/>
    </svg>`;
  paneNode.append(panel);
  const current = panel.querySelector('#deep-travel-current');
  const reference = panel.querySelector('#deep-travel-reference');
  const difference = panel.querySelector('#deep-travel-difference');
  const selectedLabel = panel.querySelector('.travel-selected dt');
  const noReference = panel.querySelector('.travel-unmatched');
  const traces = panel.querySelector('.travel-traces');
  const selectedPath = traces.querySelector('.travel-selected-trace');
  const referencePath = traces.querySelector('.travel-reference-trace');
  const cursor = traces.querySelector('.travel-phase');

  function setClip(clip, activeValues, referenceValues) {
    const paired = Boolean(referenceValues?.length);
    selectedLabel.textContent = Number.isFinite(clip.hipCapacityFraction)
      ? Math.round(100 * clip.hipCapacityFraction) + '% capacity' : 'Selected clip';
    reference.parentElement.hidden = difference.parentElement.hidden = !paired;
    noReference.hidden = paired;
    panel.title = 'Modeled displacement of the iliotibial band along the trochanteric bursa from its neutral position, in millimetres.' +
      (paired ? ' Difference is the selected walk minus the 100% capacity walk at the same stride phase.' : '');
    const yMax = Math.max(1, ...(activeValues ?? []), ...(referenceValues ?? []));
    function tracePath(values) {
      if (!values || values.length < 2) return '';
      return values.map((mm, i) => {
        const x = 2 + 140 * i / (values.length - 1);
        const y = 41 - 38 * mm / yMax;
        return (i ? 'L' : 'M') + x.toFixed(2) + ' ' + y.toFixed(2);
      }).join('');
    }
    selectedPath.setAttribute('d', tracePath(activeValues));
    referencePath.setAttribute('d', tracePath(referenceValues));
    traces.style.visibility = activeValues?.length ? '' : 'hidden';
    setPhase(0, undefined, undefined);
  }

  function setPhase(phase, selectedMm, referenceMm) {
    const selected = Number.isFinite(selectedMm) ? selectedMm.toFixed(1) : '—';
    const normal = Number.isFinite(referenceMm) ? referenceMm.toFixed(1) : '—';
    let delta = '—';
    if (Number.isFinite(selectedMm) && Number.isFinite(referenceMm)) {
      const value = Number((selectedMm - referenceMm).toFixed(1));
      delta = (value > 0 ? '+' : value < 0 ? '−' : '') + Math.abs(value).toFixed(1);
    }
    if (current.textContent !== selected) current.textContent = selected;
    if (reference.textContent !== normal) reference.textContent = normal;
    if (difference.textContent !== delta) difference.textContent = delta;
    const x = (2 + 140 * phase).toFixed(2);
    cursor.setAttribute('x1', x);
    cursor.setAttribute('x2', x);
  }
  return {setClip, setPhase};
}
