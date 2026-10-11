import json
import pathlib
import tempfile
import unittest

from coderunner.audit import AuditLog
from coderunner.memory import MAX_CHARS, Memory, fts_query
from coderunner.store import Store

# Fake embedding space: one axis per concept, plus a weak "other" axis so
# unrelated texts are near-orthogonal but never zero vectors.
CONCEPTS = {
    "dog": 0, "puppy": 0, "pet": 0, "rex": 0, "cat": 0,
    "food": 1, "ramen": 1, "pizza": 1, "dinner": 1, "eat": 1,
    "flight": 2, "lisbon": 2, "trip": 2, "travel": 2,
}


class FakeEmbedder:
    def __init__(self):
        self.calls = 0
        self.fail = False

    async def __call__(self, texts):
        self.calls += 1
        if self.fail:
            raise ConnectionError("ollama is not running")
        vectors = []
        for text in texts:
            vector = [0.0, 0.0, 0.0, 0.0]
            for word in text.lower().split():
                word = word.strip(".,!?'\"")
                if word in CONCEPTS:
                    vector[CONCEPTS[word]] += 1.0
                else:
                    vector[3] += 0.05
            vectors.append(vector)
        return vectors


class MemoryTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audit = AuditLog(pathlib.Path(self.tmp.name) / "audit.jsonl")
        self.store = Store(":memory:")
        self.embedder = FakeEmbedder()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def events(self, name):
        return [e for e in self.audit.tail(1000) if e["event"] == name]


class SanitiseTests(unittest.TestCase):
    def test_terms_are_quoted_and_operators_neutralised(self):
        self.assertEqual(fts_query('he said "hi" AND bye*'), '"he" OR "said" OR "hi" OR "and" OR "bye"')
        self.assertEqual(fts_query("NEAR(a b) NOT c", all_terms=True), '"near" AND "a" AND "b" AND "not" AND "c"')
        self.assertEqual(fts_query("tags:secret ^start -x"), '"tags" OR "secret" OR "start" OR "x"')

    def test_no_terms_gives_empty_expression(self):
        for text in ("", "   ", '"', '""', "*", "()", "😀🎉", "-- ;", None):
            self.assertEqual(fts_query(text), "")

    def test_duplicates_dropped_and_term_count_capped(self):
        self.assertEqual(fts_query("Dog dog DOG"), '"dog"')
        self.assertEqual(fts_query(" ".join(f"w{i}" for i in range(100))).count('"') // 2, 32)


class KeywordSearchTests(MemoryTestCase):
    async def asyncSetUp(self):
        self.memory = Memory(self.store, self.audit)
        self.pepper = await self.memory.save("Salt and pepper go in the left drawer")
        self.pizza = await self.memory.save("Favourite takeaway: 🍕 pizza from Luigi's", tags="food")
        self.wifi = await self.memory.save('Guest wifi password is "orange-cat-42"', tags="home")

    async def test_weird_queries_never_raise(self):
        weird = [
            '"', '""', "'", '"unterminated', "AND", "OR NOT", "NEAR(", "NEAR/2", "*", "foo*", "-bar",
            "a:b", "(", ")", "^start", "{text tags}", "😀", "text:hello", "tags:home", "\x00", "a" * 5000,
            "'; DROP TABLE memories; --", "\\", "%", "_", "col1 : col2", "+", "1.5e10", "ünïcödé",
        ]
        for query in weird:
            with self.subTest(query=query[:20]):
                self.assertIsInstance(await self.memory.search(query), list)
        self.assertEqual(len(self.memory.recent()), 3)

    async def test_operator_words_are_searched_literally(self):
        self.assertEqual([r["id"] for r in await self.memory.search("AND")], [self.pepper["id"]])
        self.assertEqual([r["id"] for r in await self.memory.search("pepper*")], [self.pepper["id"]])

    async def test_emoji_and_quotes_in_query(self):
        self.assertEqual([r["id"] for r in await self.memory.search("🍕 pizza?")], [self.pizza["id"]])
        self.assertEqual([r["id"] for r in await self.memory.search('"orange-cat-42"')], [self.wifi["id"]])
        self.assertEqual(await self.memory.search("😀"), [])

    async def test_tags_are_searchable_and_column_filters_are_not(self):
        self.assertEqual([r["id"] for r in await self.memory.search("home")], [self.wifi["id"]])
        results = await self.memory.search("tags:food")
        self.assertEqual(results[0]["id"], self.pizza["id"])

    async def test_result_shape_and_ranking(self):
        await self.memory.save("pepper pepper pepper grinder")
        results = await self.memory.search("pepper drawer", limit=5)
        self.assertEqual(results[0]["id"], self.pepper["id"])  # matches both terms
        self.assertEqual(set(results[0]), {"id", "text", "tags", "created", "source", "score"})
        self.assertEqual(results, sorted(results, key=lambda r: -r["score"]))
        self.assertEqual(len(await self.memory.search("pepper", limit=1)), 1)

    async def test_empty_query(self):
        self.assertEqual(await self.memory.search("   "), [])


class SaveTests(MemoryTestCase):
    async def asyncSetUp(self):
        self.memory = Memory(self.store, self.audit, self.embedder)

    async def test_strips_and_rejects_bad_input(self):
        saved = await self.memory.save("  Mum's birthday is 3 March \n", tags=" family ")
        self.assertEqual((saved["text"], saved["tags"], saved["duplicate"]), ("Mum's birthday is 3 March", "family", False))
        for bad in ("", "   \n\t", None):
            with self.assertRaises(ValueError):
                await self.memory.save(bad)
        with self.assertRaises(ValueError):
            await self.memory.save("x" * (MAX_CHARS + 1))
        self.assertEqual(len((await self.memory.save("y" * MAX_CHARS))["text"]), MAX_CHARS)

    async def test_exact_duplicate_returns_existing(self):
        first = await self.memory.save("Dentist is Dr Patel")
        again = await self.memory.save("  Dentist is Dr Patel  ", tags="health")
        self.assertEqual(again["id"], first["id"])
        self.assertTrue(again["duplicate"])
        self.assertEqual(self.store.one("SELECT COUNT(*) AS n FROM memories")["n"], 1)
        different = await self.memory.save("dentist is dr patel")
        self.assertNotEqual(different["id"], first["id"])

    async def test_list_tags_are_joined(self):
        saved = await self.memory.save("Gate code 1234", tags=["home", " access ", ""])
        self.assertEqual(saved["tags"], "home, access")

    async def test_embedding_stored_as_json(self):
        saved = await self.memory.save("My dog is called Rex")
        stored = json.loads(self.store.one("SELECT embedding FROM memories WHERE id = ?", (saved["id"],))["embedding"])
        self.assertAlmostEqual(sum(v * v for v in stored), 1.0, places=4)
        self.assertEqual(self.events("memory.save")[0]["embedded"], True)

    async def test_recent_is_newest_first(self):
        ids = [(await self.memory.save(f"note {i}"))["id"] for i in range(5)]
        self.assertEqual([r["id"] for r in self.memory.recent(3)], ids[::-1][:3])


class HybridSearchTests(MemoryTestCase):
    async def asyncSetUp(self):
        self.memory = Memory(self.store, self.audit, self.embedder)
        self.dog = await self.memory.save("My dog is called Rex")
        self.flight = await self.memory.save("Flight to Lisbon on Friday at 7am")
        self.ramen = await self.memory.save("Favourite food is ramen")

    async def test_paraphrase_found_through_embeddings(self):
        results = await self.memory.search("puppy name")
        self.assertEqual(results[0]["id"], self.dog["id"])
        keyword_only = Memory(self.store, self.audit)
        self.assertEqual(await keyword_only.search("puppy name"), [])

    async def test_fusion_prefers_documents_both_rankers_agree_on(self):
        walk = await self.memory.save("dog walking schedule for the weekend")
        meeting = await self.memory.save("team schedule review")
        puppy = await self.memory.save("puppy training")
        results = await self.memory.search("dog schedule", limit=10)
        ids = [r["id"] for r in results]
        self.assertEqual(ids[0], walk["id"])
        self.assertIn(meeting["id"], ids)  # keyword-only hit
        self.assertIn(puppy["id"], ids)    # meaning-only hit
        self.assertNotIn(self.flight["id"], ids[:3])
        self.assertEqual([r["score"] for r in results], sorted((r["score"] for r in results), reverse=True))

    async def test_forgotten_memories_leave_vector_search(self):
        self.assertEqual(self.memory.forget(id=self.dog["id"]), 1)
        self.assertNotIn(self.dog["id"], [r["id"] for r in await self.memory.search("puppy")])

    async def test_vectors_written_by_another_instance_are_picked_up(self):
        other = Memory(self.store, self.audit, self.embedder)
        trip = await other.save("Trip to Lisbon booked")
        ids = [r["id"] for r in await self.memory.search("travel")]
        self.assertIn(trip["id"], ids)

    async def test_recall_returns_only_clearly_relevant_memories(self):
        # search() ranks a shared word ("Friday"); recall() wants the meaning to match.
        self.assertIn(self.flight["id"], [r["id"] for r in await self.memory.search("dentist on Friday")])
        self.assertEqual(await self.memory.recall("dentist on Friday"), [])
        self.assertEqual([r["id"] for r in await self.memory.recall("what is my puppy called")], [self.dog["id"]])
        # Without embeddings nothing is recalled automatically; the assistant can call memory_search.
        self.assertEqual(await Memory(self.store, self.audit).recall("Lisbon Friday"), [])

    async def test_corrupt_or_mismatched_vectors_are_skipped(self):
        self.store.execute("UPDATE memories SET embedding = 'not json' WHERE id = ?", (self.flight["id"],))
        self.store.execute("UPDATE memories SET embedding = '[1.0, 0.0]' WHERE id = ?", (self.ramen["id"],))
        fresh = Memory(self.store, self.audit, self.embedder)
        self.assertEqual((await fresh.search("pet"))[0]["id"], self.dog["id"])


class EmbedderFailureTests(MemoryTestCase):
    async def asyncSetUp(self):
        self.memory = Memory(self.store, self.audit, self.embedder)

    async def test_falls_back_to_keywords_and_audits_once(self):
        self.embedder.fail = True
        saved = await self.memory.save("My dog is called Rex")
        await self.memory.save("Flight to Lisbon")
        self.assertIsNone(self.store.one("SELECT embedding FROM memories WHERE id = ?", (saved["id"],))["embedding"])
        self.assertEqual([r["id"] for r in await self.memory.search("dog")], [saved["id"]])
        self.assertEqual(await self.memory.search("puppy"), [])
        self.assertEqual(len(self.events("memory.embedding_unavailable")), 1)

        # Recovery: backfill embeds what was missed, then paraphrase search works.
        self.embedder.fail = False
        self.assertEqual(await self.memory.backfill(), 2)
        self.assertEqual(await self.memory.backfill(), 0)
        self.assertEqual((await self.memory.search("puppy"))[0]["id"], saved["id"])

        # A later outage is reported again, once.
        self.embedder.fail = True
        await self.memory.search("puppy")
        await self.memory.search("puppy")
        self.assertEqual(len(self.events("memory.embedding_unavailable")), 2)

    async def test_malformed_vectors_count_as_failure(self):
        for bad in ([], [[0.0, 0.0]], [["a", "b"]], [[1.0], [2.0]], None, [[float("nan"), 1.0]]):
            async def embed(texts, bad=bad):
                return bad
            memory = Memory(self.store, self.audit, embed)
            saved = await memory.save(f"note about {len(self.events('memory.save'))}")
            self.assertIsNone(self.store.one("SELECT embedding FROM memories WHERE id = ?", (saved["id"],))["embedding"])
        self.assertEqual(len(self.events("memory.embedding_unavailable")), 6)

    async def test_no_embedding_call_when_nothing_is_embedded(self):
        memory = Memory(self.store, self.audit)
        await memory.save("plain note")
        self.memory.embedder = self.embedder
        await self.memory.search("plain")
        self.assertEqual(self.embedder.calls, 0)


class ForgetTests(MemoryTestCase):
    async def asyncSetUp(self):
        self.memory = Memory(self.store, self.audit)
        self.old = await self.memory.save("My old address is 12 Elm Street")
        self.new = await self.memory.save("My new address is 4 Oak Avenue")
        self.other = await self.memory.save("My favourite colour is green")

    async def test_forget_by_id(self):
        self.assertEqual(self.memory.forget(id=self.old["id"]), 1)
        self.assertEqual(self.memory.forget(id=self.old["id"]), 0)
        self.assertEqual(await self.memory.search("Elm"), [])
        self.assertEqual(len(self.memory.recent()), 2)

    async def test_forget_by_query_requires_every_term(self):
        self.assertEqual(self.memory.forget(query="old address"), 1)
        remaining = {r["id"] for r in self.memory.recent()}
        self.assertEqual(remaining, {self.new["id"], self.other["id"]})
        self.assertEqual(await self.memory.search("Elm"), [])
        self.assertEqual(self.memory.forget(query="address"), 1)

    async def test_forget_with_no_usable_terms_deletes_nothing(self):
        for query in ("", "   ", "*", '"', "😀", "AND OR"):
            self.memory.forget(query=query)
        self.assertEqual(len(self.memory.recent()), 3)

    async def test_requires_exactly_one_selector(self):
        with self.assertRaises(ValueError):
            self.memory.forget()
        with self.assertRaises(ValueError):
            self.memory.forget(id=1, query="x")

    async def test_forget_is_audited(self):
        self.memory.forget(query="favourite colour")
        event = self.events("memory.forget")[-1]
        self.assertEqual((event["by"], event["ids"], event["count"]), ("query", [self.other["id"]], 1))


if __name__ == "__main__":
    unittest.main()
