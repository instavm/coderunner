import json
import pathlib
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

    def embed(self, texts, urgent=False):
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
            self.assertEqual(server.get("/api/tags")["models"][0]["digest"], "text:70fcf18|media:abcdef012345")
            self.assertTrue(server.get("/health")["loaded"])

    def test_pictures_without_ollama_are_refused_and_text_still_works(self):
        with Served(srv.Service(FakeEncoder(), "m")) as server:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                server.post("/api/embed", {"model": "m", "input": [{"image": "AAAA"}]})
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(server.post("/api/show", {})["capabilities"], ["embedding"])
            self.assertEqual(len(server.post("/api/embed", {"model": "m", "input": ["hi"]})["embeddings"]), 1)

    def test_first_use_compiles_the_package_where_coremltools_accepts_it(self):
        import sys
        import tempfile
        import types
        compiled = []

        def compile_model(source, destination):
            if not destination.endswith(".mlmodelc"):        # coremltools' own rule
                raise Exception('"destination_path" parameter must have ".mlmodelc" file extension.')
            pathlib.Path(destination).mkdir(parents=True)
            compiled.append(destination)
        fake = types.SimpleNamespace(utils=types.SimpleNamespace(compile_model=compile_model))
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"coremltools": fake}):
            (pathlib.Path(folder) / "config.json").write_text("{}")
            encoder = srv.Encoder(pathlib.Path(folder))
            path = encoder._compiled()
            self.assertEqual(path, pathlib.Path(folder) / srv.COMPILED)
            self.assertTrue(path.is_dir())
            self.assertFalse((pathlib.Path(folder) / "compiling").exists())
            self.assertEqual(encoder._compiled(), path)
            self.assertEqual(len(compiled), 1)                     # compiled once, then reused

    def test_a_search_goes_before_the_rest_of_an_indexing_batch(self):
        import tempfile
        import types
        with tempfile.TemporaryDirectory() as folder:
            (pathlib.Path(folder) / "config.json").write_text('{"bos_token_id": 2, "eos_token_id": 1}')
            encoder = srv.Encoder(pathlib.Path(folder))

            def load():
                encoder.tokenizer = types.SimpleNamespace(encode=lambda text, add_special_tokens=False:
                                                          types.SimpleNamespace(ids=[5, 6]))

            def window(ids):
                time.sleep(0.05)                 # one passage on the model
                return [1.0, 0.0]
            encoder._load, encoder._window = load, window
            batch = threading.Thread(target=encoder.embed, args=(["passage"] * 40,))
            began = time.monotonic()
            batch.start()
            time.sleep(0.2)
            asked = time.monotonic()
            encoder.embed(["task: search result | query: rent"], urgent=True)
            answered = time.monotonic()
            batch.join()
            self.assertLess(answered - asked, 0.3)                    # not the ~1.8 s left of the batch
            self.assertGreater(time.monotonic() - began, 1.9)         # the batch still did all its work

    def media_encoder(self, folder):
        """An Encoder whose Core ML parts are fakes that record what they are given."""
        import numpy as np
        config = {"bos_token_id": 2, "eos_token_id": 1, "embed_scale": 1.0, "token_table": {"file": "t", "shape": [300000, 8]},
                  "vision": {"package": "V.mlpackage", "functions": {"vision_70": {"tokens": 70, "patches": 630}},
                             "patch_size": 16, "pooling_kernel": 3, "boi_token_id": 255999, "eoi_token_id": 258882,
                             "position_table": {"file": "p", "shape": [2, 1024, 768]}},
                  "audio": {"package": "A.mlpackage", "sample_rate": 16000, "window_seconds": 10, "waveform_samples": 160160,
                            "left_pad_samples": 160, "frames": 1000, "tokens_per_window": 250,
                            "boa_token_id": 256000, "eoa_token_id": 258883}}
        (pathlib.Path(folder) / "config.json").write_text(json.dumps(config))
        for package in ("V.mlpackage", "A.mlpackage"):
            (pathlib.Path(folder) / package).mkdir()
        encoder = srv.Encoder(pathlib.Path(folder))
        encoder.positions = np.zeros((2, 1024, 768), dtype=np.float16)
        encoder.calls, encoder.sequences = [], []

        class Fake:
            def __init__(self, name):
                self.name = name

            def predict(self, inputs):
                encoder.calls.append((self.name, inputs))
                if self.name == "vision":
                    return {"image_tokens": np.ones((1, 70, 8), dtype=np.float16)}
                return {"audio_tokens": np.ones((1, 250, 8), dtype=np.float16)}
        encoder._encoder = lambda name, function=None: Fake(name)
        encoder._load = lambda: setattr(encoder, "tokenizer", object())
        encoder._rows = lambda ids: np.zeros((len(ids), 8), dtype=np.float32)

        def sequence(x):
            encoder.sequences.append(len(x))
            return [1.0] + [0.0] * 7
        encoder._sequence = sequence
        return encoder

    def test_pictures_are_cut_into_patches_and_pooled_as_the_card_says(self):
        import io as bytes_io
        import tempfile
        from PIL import Image
        self.assertEqual(srv.fit(1280, 720, 630 * 256, 48), (528, 288))       # 16:9 within 630 patches of 16 px
        for width, height in ((1280, 720), (300, 2000), (50, 50)):
            w, h = srv.fit(width, height, 630 * 256, 48)
            self.assertTrue(w % 48 == 0 and h % 48 == 0 and (w // 16) * (h // 16) <= 630)
        with tempfile.TemporaryDirectory() as folder:
            encoder = self.media_encoder(folder)
            picture = bytes_io.BytesIO()
            Image.new("RGB", (1280, 720), (255, 0, 0)).save(picture, "JPEG")
            vector = encoder.image(picture.getvalue(), 70)
            self.assertEqual(len(vector), 8)
            name, inputs = encoder.calls[-1]
            count = (528 // 16) * (288 // 16)                                  # 33 x 18 patches
            self.assertEqual(name, "vision")
            self.assertEqual(int(inputs["valid"].sum()), count)
            self.assertEqual(inputs["patches"].shape, (1, 630, 768))
            self.assertAlmostEqual(float(inputs["patches"][0, 0, 0]), 1.0, places=2)     # red, scaled to 0..1
            self.assertEqual(float(inputs["positions"][0, count, 0]), -1)                # padding
            self.assertEqual(tuple(inputs["positions"][0, 33, :]), (0, 1))               # second row starts at x 0
            pool = inputs["pool"].astype(float)
            self.assertTrue(all(abs(pool[:, i].sum() - 1 / 9) < 1e-3 for i in range(count)))
            self.assertEqual(encoder.sequences[-1], (33 // 3) * (18 // 3) + 4)          # 66 tokens + 4 special

    def test_a_search_does_not_wait_for_a_picture_in_the_vision_encoder(self):
        import io as bytes_io
        import tempfile
        import types
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            encoder = self.media_encoder(folder)
            inside, finish = threading.Event(), threading.Event()
            tokens = encoder._image_tokens

            def slow(data, count):
                inside.set()
                finish.wait(5)                   # the picture is still in its encoder
                return tokens(data, count)
            encoder._image_tokens = slow
            encoder._window = lambda ids: [1.0] + [0.0] * 7
            encoder._load = lambda: setattr(encoder, "tokenizer", types.SimpleNamespace(
                encode=lambda text, add_special_tokens=False: types.SimpleNamespace(ids=[5, 6])))
            picture = bytes_io.BytesIO()
            Image.new("RGB", (96, 96)).save(picture, "PNG")
            indexing = threading.Thread(target=encoder.image, args=(picture.getvalue(), 70))
            indexing.start()
            self.assertTrue(inside.wait(5))
            asked = time.monotonic()
            encoder.embed(["task: search result | query: rent"], urgent=True)
            self.assertLess(time.monotonic() - asked, 1.0)
            self.assertFalse(encoder.release())  # not while the picture uses the model
            finish.set()
            indexing.join(5)
            self.assertEqual(len(encoder.sequences), 1)                        # then its tokens were read
            self.assertTrue(encoder.release())

    def test_sound_goes_through_the_audio_encoder_ten_seconds_at_a_time(self):
        import io as bytes_io
        import tempfile
        import wave
        with tempfile.TemporaryDirectory() as folder:
            encoder = self.media_encoder(folder)
            sound = bytes_io.BytesIO()
            with wave.open(sound, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\x00\x01" * 16000 * 25)                     # 25 s
            encoder.audio(sound.getvalue())
            windows = [inputs for name, inputs in encoder.calls if name == "audio"]
            self.assertEqual(len(windows), 3)                                   # 10 + 10 + 5 s
            self.assertEqual(int(windows[-1]["frame_mask"].sum()), (160 + 80000 - 321) // 160 + 1)
            self.assertEqual(encoder.sequences, [504, 129])                     # 500 tokens, then the last 5 s
            with self.assertRaises(ValueError):                                # 44.1 kHz stereo is refused
                stereo = bytes_io.BytesIO()
                with wave.open(stereo, "wb") as out:
                    out.setnchannels(2)
                    out.setsampwidth(2)
                    out.setframerate(44100)
                    out.writeframes(b"\x00" * 400)
                encoder.audio(stereo.getvalue())

    def test_the_server_embeds_media_itself_when_the_build_has_encoders(self):
        import base64 as b64
        encoder = FakeEncoder()
        encoder.media = lambda: {"vision", "audio"}
        encoder.image = lambda data: [7.0] + [0.0] * 767
        encoder.audio = lambda data: [8.0] + [0.0] * 767
        with Served(srv.Service(encoder, "m", "rev", "http://127.0.0.1:9", "embeddinggemma-2")) as server:
            out = server.post("/api/embed", {"model": "m", "input": ["ab", {"image": b64.b64encode(b"x").decode()},
                                                                      {"audio": b64.b64encode(b"y").decode()}]})
            self.assertEqual([v[0] for v in out["embeddings"]], [2.0, 7.0, 8.0])   # nothing went to Ollama (port 9)
            self.assertEqual(server.post("/api/show", {})["capabilities"], ["audio", "embedding", "vision"])
            self.assertEqual(server.get("/api/tags")["models"][0]["digest"], "text:rev|media:coreml-audio-vision")

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
        self.warmed = []

        async def warm(url, model):
            self.warmed.append((url, model))
        warming = mock.patch.object(coreml, "warm", warm)
        warming.start()
        self.addCleanup(warming.stop)
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
        self.assertEqual(self.warmed, [(rt.base_url, "coderunner-embeddinggemma-2")])   # compiled during setup
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
        self.assertEqual((self.cfg.get("models.embedding_media"), self.cfg.get("models.embedding_url")),
                         ("", "http://127.0.0.1:11434/v1"))
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
