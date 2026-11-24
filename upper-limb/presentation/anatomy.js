import './anatomy.css';

export function mountAnatomy(slot) {
  const frame = document.createElement('iframe');
  frame.className = 'anatomy-workbench';
  frame.title = 'Synchronized full-body motion and three anatomical views of the hip';
  frame.allowFullscreen = true;
  frame.src = '../hip-study/?embed=motion';
  const holder = document.createElement('div');
  holder.className = 'anatomy-viewport';
  holder.append(frame);
  slot.replaceChildren(holder);

  function connectKeyboard() {
    frame.contentDocument?.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      if (!['ArrowLeft', 'ArrowRight', 'PageDown', 'PageUp', 'Home', 'End'].includes(event.key)) return;
      if (event.target.closest('input,select,canvas,button,[role="dialog"]')) return;
      if (frame.contentDocument.fullscreenElement) return;
      event.preventDefault();
      document.body.dispatchEvent(new KeyboardEvent('keydown', {key: event.key, bubbles: true, cancelable: true}));
    });
  }
  frame.addEventListener('load', connectKeyboard);
  return {
    dispose() {
      frame.removeEventListener('load', connectKeyboard);
      frame.remove();
    },
  };
}
