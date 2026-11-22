// The thread's renderer (coderunner/server/web/app/render.js) against a minimal stand-in for the page's DOM.
import assert from "node:assert/strict";

class Node_ {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.childNodes = []; this.className = ""; this.attrs = {}; this._text = null; }
  appendChild(child) { this.childNodes.push(child); return child; }
  append(...kids) { for (const kid of kids) this.appendChild(kid); }
  get classList() { const self = this; return {add(name) { self.className = (self.className + " " + name).trim(); }}; }
  set textContent(value) { this.childNodes = [{text: String(value)}]; }
  get textContent() { return this.childNodes.map((c) => c.text !== undefined ? c.text : c.textContent).join(""); }
  set innerHTML(value) { throw new Error("the renderer never sets HTML"); }
  all(test, found = []) { for (const c of this.childNodes) if (c.tagName) { if (test(c)) found.push(c); c.all(test, found); } return found; }
}
globalThis.document = {createElement: (tag) => new Node_(tag), createTextNode: (text) => ({text: String(text), get textContent() { return this.text; }, set textContent(v) { this.text = String(v); }})};
const {messages, markdown, itemCards, inline, el, answer} = await import(process.argv[2]);
const tag = (name) => (n) => n.tagName === name.toUpperCase();

// Two short paragraphs are two messages; a long text stays one.
const two = messages("Amazon is cheaper at **₹50,999**.\n\nYour pincode may change the price.");
assert.equal(two.length, 2);
assert.ok(two.every((m) => m.className.startsWith("say")));
assert.equal(two[0].textContent, "Amazon is cheaper at ₹50,999.");
assert.equal(messages("a\n\nb\n\nc\n\nd\n\ne").length, 1);

// A table is a block of its own between the messages; four columns or more become cards, a row each.
const mixed = messages("Here they are.\n\n| Site | Price |\n|---|---|\n| A | ₹1 |\n| B | ₹2 |\n\nA is cheaper.");
assert.deepEqual(mixed.map((m) => m.className.split(" ")[0]), ["say", "md-table-wrap", "say"]);
const wide = messages("| Site | Price | Seller | Link |\n|---|---|---|---|\n| A | ₹1 | S | https://a.example/x |");
assert.equal(wide[0].className.split(" ")[0], "md-cards");
assert.equal(wide[0].all((n) => n.className === "md-card").length, 1);

// A named link is the name with a mark for its site; the address is only ever the link's own.
const linked = inline(el("p"), "the [Sony WH-CH520](https://www.amazon.in/dp/B0X) at ₹2,999");
const a = linked.all(tag("a"))[0];
assert.equal(a.href, "https://www.amazon.in/dp/B0X");
assert.equal(a.textContent, "ASony WH-CH520");                        // the mark's letter, then the name
assert.equal(a.all((n) => n.className === "site-mark")[0].textContent, "A");
assert.equal(inline(el("p"), "[x](javascript:alert(1))").all(tag("a")).length, 0);

// Text from a page stays text: tags are shown as written, never made into elements.
const hostile = messages('<img src=x onerror=alert(1)> <script>bad()</script> **<b>bold</b>**');
assert.equal(hostile[0].all(tag("img")).length + hostile[0].all(tag("script")).length + hostile[0].all(tag("b")).length, 0);
assert.ok(hostile[0].textContent.includes("<script>bad()</script>"));

// Item cards come from the structure given, each field as text; only an http(s) address makes a card a link.
const row = itemCards([{name: "Pixel 10a <b>", price: "₹50,999", seller: "Amazon", url: "https://www.amazon.in/dp/B0X", image: "abc-i0.jpg"},
                       {name: "Other", url: "javascript:alert(1)"}], (name) => "/api/shots/" + name);
const [first, second] = row.childNodes;
assert.equal(first.tagName, "A");
assert.equal(first.href, "https://www.amazon.in/dp/B0X");
assert.equal(first.all(tag("img"))[0].src, "/api/shots/abc-i0.jpg");
assert.equal(first.all((n) => n.className === "item-name")[0].textContent, "Pixel 10a <b>");
assert.equal(first.all((n) => n.className === "item-price")[0].textContent, "₹50,999");
assert.equal(second.tagName, "DIV");
assert.equal(second.all((n) => n.className === "item-letter")[0].textContent, "O");
assert.equal(markdown(el("div"), "one\n\ntwo").all(tag("p")).length, 2);

// A list whose every line opens with a bold name is marked to be shown in normal weight; other lists are not.
assert.ok(messages("- **A** one\n- **B** two")[0].all(tag("ul"))[0].className.includes("md-calm"));
assert.ok(!messages("- **A** one\n- plain two")[0].all(tag("ul"))[0].className.includes("md-calm"));

// With cards shown, a list that goes through the same items again is not said twice: what each line adds goes
// on its card, and the cards follow the words (or lead them when the words say "above").
const shown = [{name: "boAt Rockerz 430 Gen 2 (60H battery, ENC)", price: "₹1,499", url: "https://shop.example/a"},
               {name: "GOBOULT Mustang Q (70H battery)", price: "₹2,499", url: "https://shop.example/b"}];
const reply = "Two good ones under ₹3,000:\n\n- **boAt Rockerz 430 Gen 2** at ₹1,499: rated 4.1 from 44,000 ratings.\n" +
              "- **GOBOULT Mustang Q** at ₹2,499: rated 4.2, with noise cancelling.\n\nThe boAt is the better buy.";
const parts = answer(reply, shown, (n) => n);
assert.deepEqual(parts.map((p) => p.className.split(" ")[0]), ["say", "say", "items"]);
assert.equal(parts.flatMap((p) => p.all(tag("ul"))).length, 0);
const notes = parts[2].all((n) => n.className === "item-note").map((n) => n.textContent);
assert.deepEqual(notes, ["Rated 4.1 from 44,000 ratings.", "Rated 4.2, with noise cancelling."]);
assert.equal(parts[0].textContent, "Two good ones under ₹3,000.");        // no colon left pointing at nothing
assert.equal(shown[0].note, undefined);                                  // what was given is not changed
assert.equal(answer("The three are shown above.", shown, (n) => n)[0].className.split(" ")[0], "items");
// A list about other things stays, and an answer without cards is as before.
const other = answer("Shown below.\n\n- **Delivery** is free\n- **Returns** within 10 days", shown, (n) => n);
assert.equal(other.flatMap((p) => p.all ? p.all(tag("ul")) : []).length, 1);
assert.equal(answer("one\n\ntwo", [], (n) => n).length, 2);
console.log("ok");
