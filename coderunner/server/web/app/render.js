// How the assistant's words are shown in a thread: chat messages built as DOM nodes from a small, safe subset
// of Markdown. Nothing a model or a page wrote is ever set as HTML: text goes in as text nodes, and a link's
// address is only ever an http(s) address.

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const siteOf = (url) => { try { return new URL(url).host.replace(/^www\./, ""); } catch { return ""; } };

// A link carries a small mark for its site: the site's first letter on a neutral tile. No icon is fetched (a
// favicon service would be told every site an answer mentions, and a site's own icon is one more request to it).
export function link(url, label, named) {
  const a = el("a", "link" + (named ? " named" : ""));
  const site = siteOf(url);
  if (site) a.appendChild(el("span", "site-mark", site.charAt(0).toUpperCase()));
  a.appendChild(document.createTextNode(label));
  a.href = url; a.title = url; a.target = "_blank"; a.rel = "noopener noreferrer";
  return a;
}

export function linkLabel(url) {
  try {
    const u = new URL(url), site = u.host.replace(/^www\./, "");
    const path = decodeURIComponent(u.pathname).replace(/\/$/, "");
    const tail = path.length > 28 ? path.slice(0, 27) + "…" : path + (u.search ? "…" : "");
    return site + (tail === "…" && !path ? "" : tail);
  } catch { return url.length > 48 ? url.slice(0, 47) + "…" : url; }
}

// Text with its web addresses as links (never HTML from the model or a page).
export function richText(node, text) {
  const parts = String(text).split(/(https?:\/\/[^\s<>()]+)/g);
  for (const part of parts) {
    if (/^https?:\/\//.test(part)) {
      // An address reads as where it leads: the site and the start of its path. The whole of it is the link.
      const url = part.replace(/[.,;:!?]+$/, ""), rest = part.slice(url.length);
      node.appendChild(link(url, linkLabel(url), false));
      if (rest) node.appendChild(document.createTextNode(rest));
    } else if (part) node.appendChild(document.createTextNode(part));
  }
  return node;
}

export function inline(node, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  for (const part of String(text).split(pattern)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) node.appendChild(el("strong", "", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) node.appendChild(el("code", "", part.slice(1, -1)));
    else {
      const m = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      if (m) node.appendChild(link(m[2], m[1].replace(/\*\*/g, ""), true));      // the page's name is the link
      else richText(node, part);                // also text that only begins with a bracket
    }
  }
  return node;
}

// A comparison of four columns or more does not fit the width of a thread as a table (a column squeezed to a
// word a line, or a table scrolled sideways with its last column out of sight). Each row becomes a card: what
// is compared as its title, its price beside it, the rest as labelled lines.
function rowCards(heads, rows) {
  const list = el("div", "md-cards");
  const priced = heads.findIndex((h, n) => n > 0 && /price|cost|fare|total|rate|amount|₹|\$/i.test(h));
  const bare = (s) => String(s).replace(/\*\*|`/g, "");
  for (const row of rows) {
    const card = el("div", "md-card"), top = el("div", "md-card-top");
    inline(top.appendChild(el("div", "md-card-title")), bare(row[0] || ""));
    const beside = priced > 0 && bare(row[priced] || "").trim().length <= 28 && bare(row[priced] || "").trim();
    if (beside) inline(top.appendChild(el("div", "md-card-price")), bare(row[priced]));
    card.appendChild(top);
    row.forEach((cell, n) => {
      if (n === 0 || !String(cell).trim() || (n === priced && beside)) return;
      const line = el("div", "md-card-line");
      line.appendChild(el("span", "md-card-label", bare(heads[n] || "")));
      inline(line.appendChild(el("span", "md-card-value")), cell);
      card.appendChild(line);
    });
    list.appendChild(card);
  }
  return list;
}

function table(heads, rows) {
  const made = el("table", "md-table"), head = made.appendChild(el("thead")).appendChild(el("tr"));
  for (const cell of heads) inline(head.appendChild(el("th")), cell);
  const body = made.appendChild(el("tbody"));
  for (const row of rows) {
    const line = body.appendChild(el("tr"));
    for (const cell of row) inline(line.appendChild(el("td")), cell);
  }
  const wrap = el("div", "md-table-wrap");
  wrap.appendChild(made);
  return wrap;
}

// The blocks of a text, in order: paragraphs, lists, tables (or their cards), headings.
export function blocks(text) {
  const out = [];
  let list = null, para = null;
  const lines = String(text ?? "").split("\n").map((raw) => raw.trimEnd());
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  const isRow = (line) => /^\s*\|.*\|\s*$/.test(line || "");
  const isRule = (line) => isRow(line) && cells(line).every((cell) => /^:?-{2,}:?$/.test(cell));
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (isRow(line) && isRule(lines[i + 1])) {             // a table: its head, a rule, then its rows
      const heads = cells(line), rows = [];
      for (i += 2; i < lines.length && isRow(lines[i]); i++) rows.push(cells(lines[i]));
      i -= 1;
      out.push(heads.length >= 4 && rows.length <= 12 ? rowCards(heads, rows) : table(heads, rows));
      list = null; para = null;
      continue;
    }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/), numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const kind = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== kind) {
        list = el(kind); if (numbered) list.start = Number(numbered[1]); out.push(list);
      }
      inline(list.appendChild(el("li")), bullet ? bullet[1] : numbered[2]);
      para = null;
    } else if (!line.trim()) { list = null; para = null; }
    else if (/^#{1,4}\s+/.test(line)) { list = null; para = null; out.push(inline(el("div", "md-h"), line.replace(/^#+\s+/, ""))); }
    else {
      list = null;
      if (para) para.appendChild(el("br")); else { para = el("p"); out.push(para); }
      inline(para, line);
    }
  }
  return out;
}

export function markdown(node, text) {
  node.classList.add("md");
  for (const block of blocks(text)) node.appendChild(block);
  return node;
}

// An answer as chat messages. Two or three short paragraphs are a message each, as a person would send them;
// a longer text stays together as one message. A table, its cards or a list is a block of its own between them.
export function messages(text, className = "say") {
  const made = blocks(text), out = [];
  const paragraphs = made.filter((b) => b.tagName.toLowerCase() === "p").length;
  const apart = paragraphs <= 3;
  let open = null;
  for (const block of made) {
    const tag = block.tagName.toLowerCase(), wide = block.className === "md-cards" || block.className === "md-table-wrap";
    if (wide) { open = null; out.push(block); block.classList.add("say-block"); continue; }
    if (!open || (apart && tag === "p")) { open = el("div", className + " md"); out.push(open); }
    open.appendChild(block);
    if (apart && tag !== "p") open = null;
  }
  return out;
}

// Things better seen than read (products, places), as a row of cards under the text. The items are a structure
// the assistant gave (show_items), never taken from its prose; every field is shown as text.
export function itemCards(items, picture) {
  const row = el("div", "items");
  for (const item of (items || []).slice(0, 8)) {
    const url = /^https?:\/\//.test(item.url || "") ? item.url : "";
    const card = el(url ? "a" : "div", "item");
    if (url) { card.href = url; card.title = url; card.target = "_blank"; card.rel = "noopener noreferrer"; }
    const frame = card.appendChild(el("div", "item-picture"));
    const name = String(item.name || "").trim();
    if (item.image && picture) {
      const image = frame.appendChild(el("img"));
      image.alt = ""; image.loading = "lazy"; image.src = picture(item.image);
    } else frame.appendChild(el("span", "item-letter", name.charAt(0).toUpperCase()));
    card.appendChild(el("div", "item-name", name));
    if (item.price) card.appendChild(el("div", "item-price", String(item.price)));
    const site = siteOf(url), where = [item.seller, site].filter(Boolean)[0];
    if (where) card.appendChild(el("div", "item-seller", String(where)));
    row.appendChild(card);
  }
  return row;
}
