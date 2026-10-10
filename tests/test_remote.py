import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from coderunner import remote

TS = "/fake/bin/tailscale"
HOST = "studio.tail1234.ts.net"

STATUS = {"BackendState": "Running", "Self": {"DNSName": HOST + ".", "TailscaleIPs": ["100.101.102.103"]}}
SERVING = {"TCP": {"443": {"HTTPS": True}},
           "Web": {f"{HOST}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8224"}}}}}
FUNNEL = {**SERVING, "AllowFunnel": {f"{HOST}:443": True}}
OTHER = {"TCP": {"443": {"HTTPS": True}},
         "Web": {f"{HOST}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}}}}
WHOIS = {"Node": {"Name": "phone"}, "UserProfile": {"LoginName": "owner@example.com", "DisplayName": "Owner"}}


class FakeTailscale:
    """Answers by subcommand with canned JSON and records every argv."""

    def __init__(self, status=STATUS, serve=None, whois=WHOIS, fail=()):
        self.calls = []
        self.status, self.serve, self.whois, self.fail = status, serve or {}, whois, set(fail)

    def __call__(self, argv):
        self.calls.append(list(argv))
        args = argv[1:]
        key = " ".join(args[:2]) if args[:2] == ["serve", "status"] else args[0]
        if key in self.fail:
            return subprocess.CompletedProcess(argv, 1, "", f"{key} failed: HTTPS is not enabled")
        out = {"status": self.status, "serve status": self.serve, "whois": self.whois}.get(key, "")
        return subprocess.CompletedProcess(argv, 0, json.dumps(out) if out != "" else "", "")


class StatusTests(unittest.TestCase):
    def test_status_and_dns_name(self):
        ts = FakeTailscale()
        status = remote.tailscale_status(runner=ts, binary=TS)
        self.assertEqual(ts.calls, [[TS, "status", "--json"]])
        self.assertEqual(remote.dns_name(status), HOST)
        self.assertEqual(remote.dns_name(runner=ts, binary=TS), HOST)
        self.assertIsNone(remote.dns_name({"Self": {}}))
        self.assertIsNone(remote.tailscale_status(runner=FakeTailscale(fail={"status"}), binary=TS))

    def test_binary_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = os.path.join(tmp, "Tailscale")
            pathlib.Path(app).touch()
            with mock.patch.object(remote.shutil, "which", return_value=None):
                self.assertEqual(remote.find_binary(app), app)
                self.assertIsNone(remote.find_binary(os.path.join(tmp, "missing")))
                with mock.patch.object(remote, "APP_BINARY", os.path.join(tmp, "missing")):
                    ts = FakeTailscale()
                    self.assertFalse(remote.status(runner=ts)["installed"])
                    self.assertIsNone(remote.tailscale_status(runner=ts))
                    with self.assertRaises(remote.RemoteError) as caught:
                        remote.enable(8224, runner=ts)
                    self.assertIn("not installed", str(caught.exception))
                    self.assertEqual(ts.calls, [])
            with mock.patch.object(remote.shutil, "which", return_value="/usr/local/bin/tailscale"):
                self.assertEqual(remote.find_binary(app), "/usr/local/bin/tailscale")

    def test_serve_status(self):
        info = remote.status(8224, runner=FakeTailscale(serve=SERVING), binary=TS)
        self.assertTrue(info["running"] and info["serving"])
        self.assertEqual(info["url"], f"https://{HOST}/")
        self.assertEqual(info["target"], "http://127.0.0.1:8224")
        self.assertFalse(info["funnel"])
        self.assertTrue(remote.status(8224, runner=FakeTailscale(serve=FUNNEL), binary=TS)["funnel"])
        info = remote.status(8224, runner=FakeTailscale(serve=OTHER), binary=TS)
        self.assertFalse(info["serving"])
        self.assertIsNone(info["url"])
        info = remote.status(runner=FakeTailscale(serve={}), binary=TS)
        self.assertFalse(info["serving"])


class EnableTests(unittest.TestCase):
    def test_enable_serves_loopback_admin_port(self):
        ts = FakeTailscale()
        self.assertEqual(remote.enable(8224, runner=ts, binary=TS), f"https://{HOST}/")
        self.assertEqual(ts.calls[-1], [TS, "serve", "--bg", "--yes", "--https=443", "http://127.0.0.1:8224"])
        self.assertFalse(any("funnel" in part for call in ts.calls for part in call))

    def test_refuses_when_funnel_is_on(self):
        for serve in (FUNNEL, {"Web": {f"{HOST}:8443": {"Handlers": {"/": {"Proxy": "127.0.0.1:8224"}}}},
                               "AllowFunnel": {f"{HOST}:8443": True}},
                      {"Foreground": {"abc": FUNNEL}}):
            ts = FakeTailscale(serve=serve)
            with self.assertRaises(remote.RemoteError) as caught:
                remote.enable(8224, runner=ts, binary=TS)
            self.assertIn("Funnel", str(caught.exception))
            self.assertFalse(any(call[1:3] == ["serve", "--bg"] for call in ts.calls))

    def test_funnel_on_an_unrelated_port_is_not_ours(self):
        serve = {"Web": {f"{HOST}:8443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:9000"}}}},
                 "AllowFunnel": {f"{HOST}:8443": True}}
        self.assertEqual(remote.funnel_ports(serve, 8224), [])

    def test_refuses_to_replace_another_service_unless_forced(self):
        ts = FakeTailscale(serve=OTHER)
        with self.assertRaises(remote.RemoteError) as caught:
            remote.enable(8224, runner=ts, binary=TS)
        self.assertIn("127.0.0.1:3000", str(caught.exception))
        self.assertEqual(remote.enable(8224, force=True, runner=ts, binary=TS), f"https://{HOST}/")

    def test_refuses_when_not_connected_or_unreadable(self):
        for status in ({"BackendState": "NeedsLogin", "Self": {}}, {"BackendState": "Running", "Self": {}}):
            with self.assertRaises(remote.RemoteError):
                remote.enable(8224, runner=FakeTailscale(status=status), binary=TS)
        with self.assertRaises(remote.RemoteError):
            remote.enable(8224, runner=FakeTailscale(fail={"serve status"}), binary=TS)
        with self.assertRaises(remote.RemoteError) as caught:
            remote.enable(8224, runner=FakeTailscale(fail={"serve"}), binary=TS)
        self.assertIn("HTTPS is not enabled", str(caught.exception))
        with self.assertRaises(remote.RemoteError):
            remote.enable(0, runner=FakeTailscale(), binary=TS)

    def test_disable(self):
        ts = FakeTailscale(serve=SERVING)
        self.assertEqual(remote.disable(runner=ts, binary=TS), "Remote access disabled.")
        self.assertEqual(ts.calls[-1], [TS, "serve", "--https=443", "off"])
        ts = FakeTailscale(serve={})
        self.assertIn("not enabled", remote.disable(runner=ts, binary=TS))
        self.assertNotIn([TS, "serve", "--https=443", "off"], ts.calls)
        ts = FakeTailscale(serve=OTHER)
        self.assertIn("Left", remote.disable(admin_port=8224, runner=ts, binary=TS))
        self.assertNotIn([TS, "serve", "--https=443", "off"], ts.calls)


class IdentityTests(unittest.TestCase):
    def test_decode_identity(self):
        self.assertEqual(remote.decode_identity({"Tailscale-User-Login": "owner@example.com"}), "owner@example.com")
        self.assertEqual(remote.decode_identity({"tailscale-user-login": "=?utf-8?q?owner=40example.com?="}),
                         "owner@example.com")
        self.assertEqual(remote.decode_identity({"Tailscale-User-Login": "=?utf-8?b?am9zw6lAZXhhbXBsZS5jb20=?="}),
                         "josé@example.com")
        self.assertEqual(remote.decode_identity(
            {"Tailscale-User-Name": "=?utf-8?q?Jos=C3=A9?="}, "Tailscale-User-Name"), "José")
        self.assertIsNone(remote.decode_identity({}))
        self.assertEqual(remote.decode_identity([("Tailscale-User-Login", "a@x.com"),
                                                 ("tailscale-user-login", "a@x.com")]), "")

    def test_whois(self):
        ts = FakeTailscale()
        self.assertEqual(remote.whois("100.64.0.7", runner=ts, binary=TS), "owner@example.com")
        self.assertEqual(ts.calls, [[TS, "whois", "--json", "100.64.0.7"]])
        self.assertIsNone(remote.whois("--help", runner=ts, binary=TS))
        self.assertIsNone(remote.whois("100.64.0.7:443", runner=ts, binary=TS))
        self.assertEqual(len(ts.calls), 1)
        self.assertIsNone(remote.whois("100.64.0.8", runner=FakeTailscale(fail={"whois"}), binary=TS))
        self.assertIsNone(remote.whois("100.64.0.8", runner=FakeTailscale(whois={"Node": {}}), binary=TS))

    def test_check_owner(self):
        owners = ["owner@example.com"]
        lookup = {"100.64.0.7": "owner@example.com", "100.64.0.9": "guest@example.com"}.get
        def check(headers, owners=owners):
            return remote.check_owner(headers, owners, lookup=lookup)
        self.assertTrue(check({}))
        self.assertTrue(check({"Tailscale-User-Login": "anyone@example.com"}, []))
        self.assertTrue(check({"Tailscale-User-Login": "owner@example.com", "X-Forwarded-For": "100.64.0.7"}))
        self.assertTrue(check({"Tailscale-User-Login": "=?utf-8?q?owner=40example.com?="}))
        self.assertFalse(check({"Tailscale-User-Login": "guest@example.com"}))
        self.assertFalse(check({"Tailscale-User-Login": "owner@example.com", "X-Forwarded-For": "100.64.0.9"}))
        self.assertFalse(check({"X-Forwarded-For": "100.64.0.9"}))
        self.assertFalse(check({"X-Forwarded-For": "100.64.0.99"}))
        self.assertTrue(check({"X-Forwarded-For": "100.64.0.7, 10.0.0.1"}))


if __name__ == "__main__":
    unittest.main()
