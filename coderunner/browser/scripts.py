"""Fixed page scripts the broker runs in an isolated world; the model never supplies script."""

TEXT_LIMIT = 10000          # longer page text is cut, and the action is then treated as sensitive

# Broker-owned constant scripts. They run in an isolated world, so page scripts
# cannot tamper with the prototypes they use; the model never supplies script.
DESCRIBE_JS = r"""function(closest) {
  let el = this.nodeType === 1 ? this : this.parentElement;
  if (el && closest) el = el.closest('a,button,input,select,textarea,summary,[role],[onclick]') || el;
  if (!el) return null;
  let truncated = false;
  const clean = s => {
    s = String(s || '').replace(/\s+/g, ' ').trim();
    if (s.length > LIMIT) { truncated = true; s = s.slice(0, LIMIT); }
    return s;
  };
  const text = e => [e.getAttribute('aria-label'), e.getAttribute('title'), e.innerText,
                     e.tagName === 'INPUT' ? e.value : ''].map(clean).filter(Boolean).join(' / ');
  const type = (el.getAttribute('type') || '').toLowerCase();
  const field = el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && !/^(submit|button|reset|image)$/.test(type));
  const form = el.form || el.closest('form');
  const submit = form && form.querySelector('button[type=submit],input[type=submit],input[type=image],button:not([type])');
  return {tag: el.tagName.toLowerCase(), type, autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
          text: field ? clean(el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.name) : text(el),
          submit: submit ? text(submit) : '', truncated};
}""".replace("LIMIT", str(TEXT_LIMIT))
FOCUSED_JS = r"""(() => {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  return el && el !== document.body && el !== document.documentElement ? el : null;
})()"""
CONTAINS_JS = "function(other) { for (let n = other; n; n = n.parentNode || n.host) if (n === this) return true; return false; }"
# Choose an option of a native <select>, whose popup is drawn by the OS where mouse events cannot reach it.
SELECT_OPTION_JS = r"""function(choice) {
  if (!this || this.tagName !== 'SELECT') return {ok: false, error: 'not a dropdown (<select>); click it instead'};
  const want = String(choice).trim().toLowerCase();
  const options = Array.from(this.options).filter(o => !o.disabled);
  const label = o => (o.label || o.text || '').trim();
  const match = options.find(o => label(o).toLowerCase() === want) || options.find(o => o.value.toLowerCase() === want)
    || options.find(o => label(o).toLowerCase().includes(want));
  if (!match) return {ok: false, error: 'no such option', options: options.slice(0, 60).map(label)};
  this.value = match.value;
  this.dispatchEvent(new Event('input', {bubbles: true}));
  this.dispatchEvent(new Event('change', {bubbles: true}));
  return {ok: true, selected: label(match)};
}"""

SELECT_ALL_JS = r"""function() {
  try { if (typeof this.select === 'function') { this.select(); return; } } catch (e) {}
  const range = document.createRange(); range.selectNodeContents(this);
  const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
}"""
# Resolves once the page has stopped changing its content (or did not start), so results include what a
# click or typed text brings in from the network. Attribute changes are ignored: animations never stop.
SETTLE_JS = r"""function(start, quiet, most) {
  return new Promise(resolve => {
    const began = performance.now();
    let last = null;
    const observer = new MutationObserver(() => { last = performance.now(); });
    observer.observe(document, {subtree: true, childList: true, characterData: true});
    const check = () => {
      const now = performance.now();
      if (now - began >= most || (last === null ? now - began >= start : now - last >= quiet)) {
        observer.disconnect();
        resolve(last !== null);
      } else {
        setTimeout(check, 50);
      }
    };
    setTimeout(check, 50);
  });
}"""
# Scrolls whatever scrolls at the middle of the window (a results list, a dialog, or the page) by script:
# synthetic wheel events wait for the compositor and hang when the window is hidden or busy.
SCROLL_JS = r"""function(dx, dy) {
  const scrolls = el => {
    const style = getComputedStyle(el);
    return (dy && /(auto|scroll|overlay)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1)
      || (dx && /(auto|scroll|overlay)/.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1);
  };
  let el = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
  while (el && el !== document.body && el !== document.documentElement && !scrolls(el)) el = el.parentElement;
  if (!el || el === document.body || el === document.documentElement) el = document.scrollingElement || document.documentElement;
  const before = [el.scrollLeft, el.scrollTop];
  el.scrollBy({left: dx, top: dy, behavior: 'instant'});
  const range = Math.max(el.scrollHeight - el.clientHeight, 1);
  return {moved: el.scrollLeft !== before[0] || el.scrollTop !== before[1],
          percent: Math.round(100 * el.scrollTop / range)};
}"""
# The element at a point and its ancestors, nearest first (through shadow roots), for actions a model
# places by coordinates on a screenshot.
ANCESTORS_JS = r"""function() {
  const out = [];
  for (let n = this.nodeType === 1 ? this : this.parentElement; n && out.length < 40;
       n = n.parentElement || (n.getRootNode && n.getRootNode().host) || null) out.push(n);
  return out;
}"""
