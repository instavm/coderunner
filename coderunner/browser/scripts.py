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
# Whether a hit is the element or inside it; a click on a control's label is a click on the control. "Inside" is
# as shown and as a click's event travels: content a web component places in a slot of its own is inside the
# slot's element (Rotten Tomatoes' poster sits in a link through two such slots) though it is not its child.
CONTAINS_JS = r"""function(other) {
  for (let n = other; n; n = n.assignedSlot || n.parentNode || n.host) if (n === this) return true;
  for (const label of this.labels || []) if (label.contains(other)) return true;
  return false;
}"""
# Choose an option of a native <select>, whose popup is drawn by the OS where mouse events cannot reach it.
SELECT_OPTION_JS = r"""function(choice) {
  if (!this || this.tagName !== 'SELECT') return {ok: false, error: 'not a dropdown (<select>); click it instead'};
  // Sites pad option text with runs of non-breaking spaces ("1 \u00a0 \u00a0 (₹\u00a01,242)"), which a model
  // cannot type back: all whitespace counts as one space on both sides.
  const norm = t => String(t || '').replace(/\s+/g, ' ').trim();
  const want = norm(choice).toLowerCase();
  const options = Array.from(this.options).filter(o => !o.disabled);
  const label = o => norm(o.label || o.text);
  const match = options.find(o => label(o).toLowerCase() === want) || options.find(o => o.value.toLowerCase() === want)
    || options.find(o => label(o).toLowerCase().includes(want));
  if (!match) return {ok: false, error: 'no such option', options: options.slice(0, 60).map(label)};
  this.value = match.value;
  this.dispatchEvent(new Event('input', {bubbles: true}));
  this.dispatchEvent(new Event('change', {bubbles: true}));
  return {ok: true, selected: label(match)};
}"""

OPTIONS_JS = r"""function() {
  return this && this.tagName === 'SELECT'
    ? Array.from(this.options).filter(o => !o.disabled).slice(0, 60).map(o => (o.label || o.text || '').replace(/\s+/g, ' ').trim())
    : null;
}"""

# A slider is set, not typed into. A range input takes the value through the element's own setter and hears
# the events a drag sends (in the isolated world a framework's own value tracker still sees the change); a
# slider drawn with ARIA reports where it stands, and is moved with arrow keys by the caller.
SLIDER_JS = r"""function(value) {
  const aria = (name) => this.getAttribute('aria-' + name);
  if (this.tagName === 'INPUT' && this.type === 'range' && value !== null) {
    const min = Number(this.min || 0), max = Number(this.max || 100);
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(this, String(Math.min(max, Math.max(min, Number(value)))));
    this.dispatchEvent(new Event('input', {bubbles: true}));
    this.dispatchEvent(new Event('change', {bubbles: true}));
  }
  const range = this.tagName === 'INPUT' && this.type === 'range';
  const number = (text) => (text === null || text === '' ? null : Number(text));
  return {range, now: number(range ? this.value : aria('valuenow')),
          min: number(range ? (this.min || '0') : aria('valuemin')),
          max: number(range ? (this.max || '100') : aria('valuemax')),
          text: aria('valuetext') || ''};
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
# The page's main repeated list (search results, products, flights, movie tiles, table rows): the group of
# siblings built alike (tag and classes, numbers in class names aside: item-3, s-result-4) with the most text. Scored on
# textContent, which costs no layout; only the chosen group is measured and read as shown.
LIST_JS = r"""function(start, count, limit) {
  const sig = e => e.tagName + '.' + Array.from(e.classList).map(c => c.replace(/\d+/g, '#')).sort().join('.');
  const clean = s => String(s || '').replace(/\s+/g, ' ').trim();
  // Lists built from web components live inside open shadow roots too (Rotten Tomatoes' tiles).
  const parents = [];
  const walk = root => {
    for (const el of root.querySelectorAll('*')) {
      parents.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document.body || document.documentElement);
  // An item is shown when it or a child has a box: wrappers with display: contents have none of their own.
  // Only shown items count, so a hidden menu with more text (Rotten Tomatoes' header) cannot win.
  const boxed = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const shownItem = e => boxed(e) || Array.from(e.children).some(boxed);
  let best = null, bestScore = 0;
  for (const parent of parents) {
    if (parent.children.length < 3) continue;
    const groups = new Map();
    for (const kid of parent.children) {
      if (/^(SCRIPT|STYLE|TEMPLATE|NOSCRIPT)$/.test(kid.tagName)) continue;
      const key = sig(kid);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(kid);
    }
    for (const items of groups.values()) {
      if (items.length < 3) continue;
      const sized = items.filter(e => clean(e.textContent).length >= 15);
      if (sized.length < 3) continue;
      const shown = sized.filter(shownItem);
      if (shown.length < 3) continue;
      const score = shown.reduce((a, e) => a + Math.min(clean(e.textContent).length, 400), 0);
      if (score > bestScore) { bestScore = score; best = shown; }
    }
  }
  if (!best) return {total: 0, items: [], links: []};
  // An item's link is its first one that is shown and read out: a product's picture link is often hidden from
  // screen readers (Amazon), and so has no number in the page text.
  const read = a => !a.closest('[aria-hidden="true"]') && boxed(a);
  const shown = best.slice(start, start + count);
  const all = shown.map(e => (e.matches('a[href]') ? [e] : []).concat(Array.from(e.querySelectorAll('a[href]'))));
  const links = all.map(found => found.find(read) || null);
  const items = shown.map((e, i) => ({text: clean(e.innerText || e.textContent).slice(0, limit),
                                      href: (links[i] || all[i][0] || {}).href || ''}));
  return {total: best.length, items, links};      // the links themselves, for the numbers the page text gives them
}"""
# The element at a point and its ancestors, nearest first (through slots and shadow roots, as for CONTAINS_JS),
# for actions a model places by coordinates on a screenshot. A label stands for its control: a click on a label
# is a click on the control, and sites draw radios and checkboxes inside the label beside a hidden input.
ANCESTORS_JS = r"""function() {
  const out = [];
  for (let n = this.nodeType === 1 ? this : this.parentElement; n && out.length < 40;
       n = n.assignedSlot || n.parentElement || (n.getRootNode && n.getRootNode().host) || null) {
    out.push(n);
    if (n.localName === 'label' && n.control) out.push(n.control);
  }
  return out;
}"""
