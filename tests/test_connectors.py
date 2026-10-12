import asyncio
import json
import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime

from coderunner.core.config import Config
from coderunner.connectors import ConnectorError, all_tools, check, load_connectors, redact
from coderunner.connectors import macos

HOSTILE = '"); do shell script "rm -rf ~"'


class FakeRunner:
    def __init__(self, stdout="{}", code=0, stderr="", raises=None):
        self.calls = []
        self.stdout, self.code, self.stderr, self.raises = stdout, code, stderr, raises

    async def __call__(self, argv, timeout):
        self.calls.append((list(argv), timeout))
        if self.raises:
            raise self.raises
        return self.code, self.stdout if isinstance(self.stdout, str) else json.dumps(self.stdout), self.stderr


def tools(runner):
    return {tool.name: tool for tool in all_tools(runner)}


class ArgvTests(unittest.IsolatedAsyncioTestCase):
    def assert_only_argv(self, argv, *values):
        self.assertEqual(argv[:4], ["/usr/bin/osascript", "-l", "JavaScript", "-e"])
        self.assertEqual(argv[5], "--")
        script, args = argv[4], argv[6:]
        for value in values:
            self.assertNotIn(value, script)
            self.assertIn(value, args)

    async def test_hostile_strings_reach_scripts_only_as_argv(self):
        runner = FakeRunner({"created": True})
        t = tools(runner)
        await t["calendar_create"].call({"title": HOSTILE, "start": "2026-10-12T15:00", "end": "2026-10-12T16:00",
                                         "calendar": HOSTILE + "cal", "notes": HOSTILE + "n", "location": "-x"})
        self.assert_only_argv(runner.calls[-1][0], HOSTILE, HOSTILE + "cal", HOSTILE + "n", "-x")
        await t["reminders_create"].call({"title": HOSTILE, "notes": HOSTILE + "n", "list": HOSTILE + "l"})
        self.assert_only_argv(runner.calls[-1][0], HOSTILE, HOSTILE + "n", HOSTILE + "l")
        await t["mail_draft"].call({"to": "a@example.com, b@example.com", "subject": HOSTILE, "body": HOSTILE + "b",
                                    "cc": "c@example.com"})
        argv = runner.calls[-1][0]
        self.assert_only_argv(argv, HOSTILE, HOSTILE + "b")
        self.assertEqual(argv[6:8], ["a@example.com\nb@example.com", "c@example.com"])
        for name, args in (("notes_search", {"query": HOSTILE}), ("notes_read", {"id": HOSTILE}),
                           ("contacts_find", {"name": HOSTILE}), ("reminders_list", {"list": HOSTILE}),
                           ("mail_recent", {"mailbox": HOSTILE}), ("mail_read", {"id": 42, "mailbox": HOSTILE})):
            runner.stdout = {}
            await t[name].call(args)
            self.assert_only_argv(runner.calls[-1][0], HOSTILE)

    async def test_note_body_is_escaped_html(self):
        runner = FakeRunner({"created": True})
        await tools(runner)["notes_create"].call({"title": "<b>Plan</b>", "body": "line & one\n\n<script>x</script>"})
        argv = runner.calls[-1][0]
        self.assertEqual(argv[6], "<h1>&lt;b&gt;Plan&lt;/b&gt;</h1><div>line &amp; one</div><div><br></div>"
                                  "<div>&lt;script&gt;x&lt;/script&gt;</div>")

    async def test_dates_become_epoch_milliseconds_and_ranges_are_checked(self):
        runner = FakeRunner({"events": []})
        events = tools(runner)["calendar_events"]
        await events.call({"start": "2026-10-12", "end": "2026-10-13T00:00:00Z", "max": 5000})
        argv = runner.calls[-1][0][6:]
        self.assertEqual(argv[0], str(int(datetime(2026, 10, 12).astimezone().timestamp() * 1000)))
        self.assertEqual(argv[1], str(int(datetime.fromisoformat("2026-10-13T00:00:00+00:00").timestamp() * 1000)))
        self.assertEqual(argv[2], "200")
        self.assertEqual(runner.calls[-1][1], macos.CALENDAR_TIMEOUT)
        for bad in ({"start": "tomorrow", "end": "2026-10-13"}, {"start": "2026-10-13", "end": "2026-10-12"},
                    {"start": "2026-01-01", "end": "2027-06-01"}):
            with self.assertRaises(ValueError):
                await events.call(bad)
        self.assertEqual(len(runner.calls), 1)

    async def test_limits_and_flags(self):
        runner = FakeRunner({"messages": []})
        t = tools(runner)
        await t["mail_recent"].call({"limit": 500, "unread_only": True})
        self.assertEqual(runner.calls[-1][0][6:], ["inbox", "50", "1"])
        self.assertEqual(runner.calls[-1][1], macos.DEFAULT_TIMEOUT)
        await t["mail_recent"].call({})
        self.assertEqual(runner.calls[-1][0][6:], ["inbox", "20", ""])
        for bad in ("12; drop", -3, "abc"):
            with self.assertRaises(ValueError):
                await t["mail_read"].call({"id": bad})

    async def test_mail_draft_rejects_non_addresses(self):
        draft = tools(FakeRunner({}))["mail_draft"]
        for to in ("", "not-an-address", "Eve <eve@example.com>", "a@example.com\x00"):
            with self.assertRaises(ValueError):
                await draft.call({"to": to, "subject": "s", "body": "b"})


class SafetyTests(unittest.TestCase):
    def test_no_tool_can_send_mail(self):
        names = [tool.name for tool in all_tools()]
        self.assertFalse([n for n in names if "send" in n])
        self.assertNotIn(".send(", macos.MAIL_DRAFT)
        self.assertIn("msg.save()", macos.MAIL_DRAFT)

    def test_write_and_untrusted_flags(self):
        t = tools(None)
        self.assertEqual({n for n, tool in t.items() if tool.write},
                         {"calendar_create", "reminders_create", "notes_create", "mail_draft"})
        for name in ("mail_recent", "mail_read", "notes_search", "notes_read", "calendar_events", "contacts_find"):
            self.assertTrue(t[name].untrusted, name)
        for tool in t.values():
            self.assertTrue(set(tool.parameters["required"]) <= set(tool.parameters["properties"]))
            self.assertIn(tool.connector, macos.CONNECTORS)

    def test_describe_shows_every_written_field_in_full(self):
        t = tools(None)
        body = "Hi Sam,\n\n" + "All the details. " * 600 + "\nBest"
        summary = t["mail_draft"].describe({"to": "sam@example.com", "cc": "kim@example.com", "subject": "Plans",
                                            "body": body})
        for part in ("sam@example.com", "kim@example.com", "Plans", body, "NOT be sent"):
            self.assertIn(part, summary)
        summary = t["calendar_create"].describe({"title": "Dentist", "start": "2026-10-12T15:00",
                                                 "end": "2026-10-12T16:00", "notes": "Bring forms\nand card",
                                                 "location": "12 High St"})
        for part in ("Dentist", "2026-10-12T15:00", "2026-10-12T16:00", "Bring forms\nand card", "12 High St"):
            self.assertIn(part, summary)
        summary = t["notes_create"].describe({"title": "T", "body": "x" * 9000})
        self.assertIn("x" * 9000, summary)
        summary = t["reminders_create"].describe({"title": "Call", "due": "nonsense", "notes": "n", "list": "L"})
        for part in ("Call", "nonsense", "n", "L"):
            self.assertIn(part, summary)


class ErrorTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, runner, name="calendar_events", args=None):
        return await tools(runner)[name].call(args or {"start": "2026-10-12", "end": "2026-10-13"})

    async def test_not_authorized(self):
        runner = FakeRunner("", 1, "execution error: Error: Error: Not authorized to send Apple events to Calendar."
                                   " (-1743)")
        with self.assertRaises(ConnectorError) as caught:
            await self.call(runner)
        self.assertIn("Privacy & Security → Automation", str(caught.exception))
        self.assertIn("Calendar", str(caught.exception))

    async def test_timeouts(self):
        with self.assertRaises(ConnectorError) as caught:
            await self.call(FakeRunner("", 1, "execution error: Error: Error: AppleEvent timed out. (-1712)"))
        self.assertIn("did not respond", str(caught.exception))
        with self.assertRaises(ConnectorError) as caught:
            await self.call(FakeRunner(raises=asyncio.TimeoutError()))
        self.assertIn("within 60 seconds", str(caught.exception))

    async def test_script_errors_and_bad_output(self):
        with self.assertRaises(ConnectorError) as caught:
            await self.call(FakeRunner("", 1, 'execution error: Error: Error: No calendar named "Work" (-2700)'))
        self.assertEqual(str(caught.exception), 'Calendar: No calendar named "Work"')
        with self.assertRaises(ConnectorError) as caught:
            await self.call(FakeRunner("", 1, "execution error: Error: Error: Can't get object. (-1728)"),
                            "notes_read", {"id": "x"})
        self.assertIn("Not found in Notes", str(caught.exception))
        with self.assertRaises(ConnectorError):
            await self.call(FakeRunner("not json"))
        with self.assertRaises(ConnectorError) as caught:
            await self.call(FakeRunner(raises=FileNotFoundError()))
        self.assertIn("macOS", str(caught.exception))

    async def test_real_process_timeout_kills_child(self):
        with self.assertRaises(asyncio.TimeoutError):
            await macos.run_process(["/bin/sleep", "5"], 0.2)


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_mail_is_redacted(self):
        runner = FakeRunner({"id": 7, "subject": "Your verification code is 482913", "sender": "x@example.com",
                             "content": "Use code 913377 or open https://example.com/reset?token=abc123 today. "
                                        "Room 1234 costs $1234."})
        result = await tools(runner)["mail_read"].call({"id": 7})
        self.assertNotIn("482913", result["subject"])
        self.assertNotIn("913377", result["content"])
        self.assertNotIn("token=abc123", result["content"])
        self.assertIn("Room 1234 costs $1234.", result["content"])
        runner.stdout = {"messages": [{"id": 1, "subject": "123456 is your sign-in code"}], "total": 1}
        result = await tools(runner)["mail_recent"].call({})
        self.assertEqual(result["messages"][0]["subject"], "[redacted code] is your sign-in code")

    async def test_contacts_are_flattened(self):
        runner = FakeRunner({"contacts": [{"name": "Sam", "organization": "", "phones": ["+1 555 0100"],
                                           "phone_labels": ["_$!<Mobile>!$_"], "emails": ["sam@example.com"],
                                           "email_labels": ["work"]}], "total": 1})
        result = await tools(runner)["contacts_find"].call({"name": "Sam"})
        self.assertEqual(result["contacts"][0]["phones"], [{"label": "Mobile", "value": "+1 555 0100"}])
        self.assertEqual(result["contacts"][0]["emails"], [{"label": "work", "value": "sam@example.com"}])

    async def test_check_runs_a_probe(self):
        runner = FakeRunner(["Home", "Work"])
        self.assertIn("2 found", await check("calendar", runner=runner))
        self.assertNotIn("Home", runner.calls[0][0][4])
        with self.assertRaises(ValueError):
            await check("photos", runner=runner)


class LoadTests(unittest.TestCase):
    def test_only_enabled_connectors_on_macos(self):
        config = Config({"connectors": {"enabled": ["calendar", "mail", "nonsense"]}})
        loaded = load_connectors(config, system="Darwin")
        self.assertEqual({t.connector for t in loaded}, {"calendar", "mail"})
        self.assertIn("mail_draft", {t.name for t in loaded})
        self.assertEqual(load_connectors(config, system="Linux"), [])
        self.assertEqual(load_connectors(Config({}), system="Darwin"), [])
        self.assertEqual(load_connectors(Config({"connectors": {"enabled": None}}), system="Darwin"), [])

    def test_runner_is_passed_through(self):
        runner = FakeRunner()
        loaded = load_connectors(Config({"connectors": {"enabled": ["notes"]}}), runner=runner, system="Darwin")
        self.assertTrue(loaded and all(t.runner is runner for t in loaded))


@unittest.skipUnless(platform.system() == "Darwin" and shutil.which("osacompile"), "needs macOS")
class ScriptSyntaxTests(unittest.TestCase):
    """Compiles every script without running it, so no app is contacted."""

    def test_scripts_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = [(t.name, t.script) for t in all_tools()] + list(macos.PROBES.items())
            for name, script in scripts:
                result = subprocess.run(["osacompile", "-l", "JavaScript", "-o", os.path.join(tmp, name + ".scpt"),
                                         "-e", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_argv_round_trip_through_osascript(self):
        values = [HOSTILE, "-e", "--", "", "é ✓\nsecond line"]
        script = "function run(argv) { return JSON.stringify(argv) }"
        result = asyncio.run(macos.run_jxa(script, values, app="Test"))
        self.assertEqual(result, values)


class RedactTests(unittest.TestCase):
    def test_masks_codes_links_and_tokens(self):
        cases = {
            "Your verification code is 482913.": "Your verification code is [redacted code].",
            "482913 is your Instagram code": "[redacted code] is your Instagram code",
            "Your OTP is 123 456": "Your OTP is [redacted code]",
            "G-551234 is your Google verification code.": "[redacted code] is your Google verification code.",
            "Passcode: 0042": "Passcode: [redacted code]",
            "Your security code:\n\n    7391\n": "Your security code:\n\n    [redacted code]\n",
            "Reset it: https://example.com/account/reset?t=abc. Thanks": "Reset it: [redacted link]. Thanks",
            "Sign in: https://app.example.com/login?next=/home": "Sign in: [redacted link]",
            "Magic: https://example.com/auth/magic/abc": "Magic: [redacted link]",
            "Go to https://example.com/x?token=abc&u=1 now": "Go to [redacted link] now",
            "Confirm (https://example.com/verify-email/abc)": "Confirm ([redacted link])",
            "key Zx7Qp2Lm9Vb4Nc8Rt1Wy6Hs3Jd5Kf0GaAbCdEfGh end": "key [redacted token] end",
        }
        for text, expected in cases.items():
            self.assertEqual(redact(text), expected, text)

    def test_keeps_ordinary_text(self):
        for text in ("Room 1234 on floor 3", "Price: $1234 or 1,234.56 EUR", "Order #12345678 has shipped",
                     "Call 555-1234 before 2026-10-10 at 09:00", "See https://example.com/blog/2026/plans",
                     "File q3-2026-annual-report-final-version-two.pdf", "Meeting at 1400 in building 1234",
                     "Donaudampfschifffahrtsgesellschaftskapitaen", ""):
            self.assertEqual(redact(text), text)


if __name__ == "__main__":
    unittest.main()
