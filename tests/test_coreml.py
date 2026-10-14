import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import httpx

from coderunner.core.llm import LLM
from coderunner.models import catalog, coreml, coreml_server as srv, manager
from coderunner.server.gateway import sandbox_app
from fakes import Svc
from test_models import TempHome, apple

SANDBOX = "sandbox-token-0123456789abcdefghijklmnop"


class FakeEncoder:
    """Vectors that say which text they came from: [length, 1, 0...]."""

    def __init__(self):
        self.loaded, self.released, self.seen = False, 0, []

    def embed(self, texts):
        self.loaded = True
        self.seen.append(list(texts))
        return [[float(len(t)), 1.0] + [0.0] * 766 for t in texts]

    def release(self):
        self.loaded, self.released = False, self.released + 1


class FakeOllama:
    """Ollama's embed, show and tags endpoints, for the pictures and sound the server passes on."""

    def __init__(self):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"models": [{"name": "embeddinggemma-2:latest", "digest": "abcdef0123456789"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if self.path == "/api/show":
                    self._send({"capabilities": ["embedding", "vision", "audio"]})
                else:
                    self._send({"embeddings": [[-1.0] * 768 for _ in body["input"]]})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class Served:
    def __init__(self, service):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), srv.handler(service))
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, body):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as response:
            return json.load(response)


class ServerTests(unittest.TestCase):
    def test_long_text_is_read_in_windows_the_encoder_can_take(self):
        self.assertEqual(srv.windows([]), [[2, 1]])
        parts = srv.windows(list(range(10, 610)))
        self.assertEqual([len(p) for p in parts], [512, 92])
        self.assertTrue(all(p[0] == 2 and p[-1] == 1 for p in parts))
        self.assertEqual(parts[1][1], 10 + srv.WINDOW)
        mixed = srv.combine([[1.0, 0.0], [0.0, 1.0]], [3, 1])
        self.assertAlmostEqual(mixed[0], 3 / 10 ** 0.5)
        self.assertAlmostEqual(sum(v * v for v in mixed), 1.0)

    def test_texts_here_pictures_from_ollama_in_order(self):
        encoder = FakeEncoder()
        with FakeOllama() as ollama, Served(srv.Service(encoder, "coderunner-embeddinggemma-2", "70fcf18",
                                                        ollama.url, "embeddinggemma-2")) as server:
            out = server.post("/api/embed", {"model": "x", "input": ["ab", {"image": "AAAA"}, "abcd"], "keep_alive": "1h"})
            self.assertEqual([v[0] for v in out["embeddings"]], [2.0, -1.0, 4.0])
            self.assertEqual(encoder.seen, [["ab", "abcd"]])
            self.assertEqual(ollama.requests[-1], ("/api/embed", {"model": "embeddinggemma-2",
                                                                  "input": [{"image": "AAAA"}], "keep_alive": "1h"}))
            one = server.post("/api/embed", {"model": "x", "input": "abc"})
            self.assertEqual(one["embeddings"][0][0], 3.0)
            openai = server.post("/v1/embeddings", {"model": "x", "input": ["a", "bb"]})
            self.assertEqual([d["index"] for d in openai["data"]], [0, 1])
            self.assertEqual(openai["data"][1]["embedding"][0], 2.0)
            self.assertEqual(server.post("/api/show", {"model": "x"})["capabilities"], ["audio", "embedding", "vision"])
            # A change of either the text weights or Ollama's build changes the digest the file index keeps.
            self.assertEqual(server.get("/api/tags")["models"][0]["digest"], "70fcf18+abcdef012345")
            self.assertTrue(server.get("/health")["loaded"])

    def test_pictures_without_ollama_are_refused_and_text_still_works(self):
        with Served(srv.Service(FakeEncoder(), "m")) as server:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                server.post("/api/embed", {"model": "m", "input": [{"image": "AAAA"}]})
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(server.post("/api/show", {})["capabilities"], ["embedding"])
            self.assertEqual(len(server.post("/api/embed", {"model": "m", "input": ["hi"]})["embeddings"]), 1)

    def test_the_model_is_released_when_idle(self):
        encoder = FakeEncoder()
        service = srv.Service(encoder, "m", idle=0.1)
        service.embed(["hi"])
        threading.Thread(target=service.watch, kwargs={"every": 0.02}, daemon=True).start()
        deadline = time.monotonic() + 3
        while encoder.loaded and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(encoder.loaded)
        self.assertEqual(encoder.released, 1)


class ChoiceTests(unittest.TestCase):
    def test_core_ml_unless_measurably_slower_or_different(self):
        self.assertEqual(coreml.choice({"coreml_per_s": 20.4, "gpu_per_s": 15.4, "cosine": 0.985})[0], "coreml")
        self.assertEqual(coreml.choice({"coreml_per_s": 19.0, "gpu_per_s": 20.0, "cosine": 0.985})[0], "coreml")
        pick, why = coreml.choice({"coreml_per_s": 20.0, "gpu_per_s": 40.0, "cosine": 0.985})
        self.assertEqual(pick, "gpu")
        self.assertIn("GPU is faster", why)
        self.assertEqual(coreml.choice({"coreml_per_s": 30.0, "gpu_per_s": 10.0, "cosine": 0.6})[0], "gpu")

    def test_the_sample_is_the_same_everywhere(self):
        sample = coreml.passages(8)
        self.assertEqual(len(sample), 8)
        self.assertTrue(all(p.startswith("title: ") and " | text: " in p for p in sample))
        self.assertEqual(sample, coreml.passages(8))


class FakeCoreML(coreml.CoreMLRuntime):
    def __init__(self, config=None, fail=None):
        super().__init__(config)
        self.base_url = "http://127.0.0.1:7/v1"
        self.calls, self.fail, self.up = [], fail, False

    def pull(self, entry, progress=None):
        self.calls.append("pull")
        if self.fail:
            raise RuntimeError(self.fail)

    def installed(self, entry):
        return "pull" in self.calls

    def available(self):
        return True

    def start(self, timeout=120):
        self.calls.append("start")
        self.up = True

    def stop(self):
        self.up = False

    def running(self):
        return self.up

    def remove(self, entry):
        self.calls.append("remove")


class RuntimeTests(TempHome):
    def test_server_command_carries_idle_release_and_the_media_model(self):
        cfg = self.config()
        cfg.set("models.idle_unload_minutes", 5)
        cfg.set("models.embedding_media", "embeddinggemma-2")
        cfg.set("models.embedding_media_url", "http://127.0.0.1:11434/v1")
        rt = coreml.CoreMLRuntime(cfg)
        cmd = rt.command(catalog.get("embeddinggemma-2"))
        self.assertEqual(cmd[1:3], ["-I", str(coreml.SERVER)])
        for flag, value in (("--idle", "300"), ("--media-model", "embeddinggemma-2"), ("--name", "coderunner-embeddinggemma-2"),
                            ("--media-url", "http://127.0.0.1:11434/v1"), ("--port", str(coreml.PORT))):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertFalse(rt.installed(catalog.get("embeddinggemma-2")))
        self.assertIsNone(rt.python())                  # no environment holding the pinned packages yet
        (rt.env / "bin").mkdir(parents=True)
        (rt.env / "bin" / "python").write_text("")
        (rt.env / "installed.txt").write_text("coremltools==1.0")
        self.assertIsNone(rt.python())                  # other packages than pinned: installed again
        (rt.env / "installed.txt").write_text("\n".join(coreml.PACKAGES))
        self.assertEqual(rt.python(), str(rt.env / "bin" / "python"))

    def test_remove_deletes_only_the_downloaded_build(self):
        rt = coreml.CoreMLRuntime(self.config())
        entry = catalog.get("embeddinggemma-2")
        folder = rt.folder(entry)
        (folder / "EmbeddingGemma2Text.mlpackage").mkdir(parents=True)
        for name in entry["artifacts"]["coreml"]["files"]:
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("x")
        self.assertTrue(rt.installed(entry))
        rt.remove(entry)
        self.assertFalse(folder.exists())
        self.assertTrue(rt.root.exists())


class SetupTests(TempHome):
    def setUp(self):
        super().setUp()
        supported = mock.patch.object(coreml, "supported", return_value=True)
        supported.start()
        self.addCleanup(supported.stop)
        self.hw = apple(18, "Apple M3 Pro", sandbox=1)
        self.cfg = self.config()
        self.cfg.set("models.embedding", "embeddinggemma-2")             # what Ollama's setup leaves
        self.cfg.set("models.embedding_url", "http://127.0.0.1:11434/v1")

    def measure(self, coreml_rate, gpu_rate, cosine=0.985):
        calls = []

        async def compare(coreml_url, coreml_model, gpu_url, gpu_model):
            calls.append((coreml_url, coreml_model, gpu_url, gpu_model))
            return {"coreml_per_s": coreml_rate, "gpu_per_s": gpu_rate, "cosine": cosine, "passages": 96}
        compare.calls = calls
        return compare

    def test_text_moves_to_the_neural_engine_and_pictures_stay_with_ollama(self):
        rt, measure = FakeCoreML(self.cfg), self.measure(20.4, 15.4)
        used = manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=rt, measure=measure)
        self.assertEqual(used, "coreml")
        self.assertEqual(measure.calls, [(rt.base_url, "coderunner-embeddinggemma-2", "http://127.0.0.1:11434/v1",
                                          "embeddinggemma-2")])
        self.assertEqual((self.cfg.get("models.embedding"), self.cfg.get("models.embedding_url")),
                         ("coderunner-embeddinggemma-2", rt.base_url))
        self.assertEqual((self.cfg.get("models.embedding_media"), self.cfg.get("models.embedding_media_url")),
                         ("embeddinggemma-2", "http://127.0.0.1:11434/v1"))
        self.assertTrue(rt.up)
        self.assertEqual(coreml.load(self.hw)["coreml_per_s"], 20.4)
        # Setting up again keeps the measurement and the media model.
        again = manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=FakeCoreML(self.cfg),
                                               measure=measure)
        self.assertEqual((again, len(measure.calls)), ("coreml", 1))
        self.assertEqual(self.cfg.get("models.embedding_media"), "embeddinggemma-2")

    def test_a_faster_gpu_keeps_the_gpu_build_and_frees_the_disk(self):
        rt = FakeCoreML(self.cfg)
        used = manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=rt,
                                              measure=self.measure(20.0, 45.0))
        self.assertEqual(used, "gpu")
        self.assertIn("remove", rt.calls)
        self.assertFalse(rt.up)
        self.assertEqual((self.cfg.get("models.embedding"), self.cfg.get("models.embedding_url")),
                         ("embeddinggemma-2", "http://127.0.0.1:11434/v1"))
        self.assertEqual(self.cfg.get("models.embedding_media"), "")
        # The measurement stands: a later setup does not download the Core ML build again.
        later = FakeCoreML(self.cfg)
        self.assertEqual(manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=later), "gpu")
        self.assertEqual(later.calls, [])

    def test_owner_choice_and_failures_keep_the_gpu_build(self):
        self.cfg.set("models.embedding_runtime", "gpu")
        rt = FakeCoreML(self.cfg)
        self.assertEqual(manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=rt), "gpu")
        self.assertEqual(rt.calls, [])
        self.cfg.set("models.embedding_runtime", "auto")
        broken = FakeCoreML(self.cfg, fail="download cut")
        notes = []
        self.assertEqual(manager.setup_coreml_embedding(self.cfg, self.hw, progress=notes.append, rt=broken,
                                                        measure=self.measure(20.0, 10.0)), "gpu")
        self.assertIn("download cut", notes[-1])
        self.assertEqual(self.cfg.get("models.embedding"), "embeddinggemma-2")
        with mock.patch.object(coreml, "supported", return_value=False):
            self.assertEqual(manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None,
                                                            rt=FakeCoreML(self.cfg)), "gpu")

    def test_without_ollama_text_still_runs_on_the_neural_engine(self):
        self.cfg.set("models.embedding", "")
        self.cfg.set("models.embedding_url", "")
        rt = FakeCoreML(self.cfg)
        measure = self.measure(1.0, 1.0)
        self.assertEqual(manager.setup_coreml_embedding(self.cfg, self.hw, progress=lambda m: None, rt=rt,
                                                        measure=measure), "coreml")
        self.assertEqual(measure.calls, [])                                # nothing to compare with
        self.assertEqual(self.cfg.get("models.embedding"), "coderunner-embeddinggemma-2")


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sandbox_embeddings_reach_the_embedding_server(self):
        with Served(srv.Service(FakeEncoder(), "coderunner-embeddinggemma-2")) as server:
            svc = Svc()
            svc.llm = LLM("http://127.0.0.1:9/v1", "chat-model")
            svc.embedding_model = "coderunner-embeddinggemma-2"
            svc.embedding_url = server.url + "/v1"
            app = sandbox_app(svc)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
                response = await client.post("/v1/embeddings", headers={"Authorization": "Bearer " + SANDBOX},
                                             json={"input": ["abc"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"][0]["embedding"][0], 3.0)
            await svc.llm.aclose()


if __name__ == "__main__":
    unittest.main()
