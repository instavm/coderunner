"""Acting for the owner outside the browser: mail sent, a setting changed, a call dialled, each approved first;
and a model kept to names it did not make up (recipients, file paths)."""

import pathlib
import tempfile
import unittest

from coderunner.agent import RunContext, build_registry
from coderunner.agent.tools import handle, handles
from coderunner.connectors import load_connectors
from coderunner.files.discover import FileAccessError
from fakes import Svc
from test_connectors import FakeRunner


class NativeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runner = FakeRunner(stdout={"sent": True})
        self.svc = Svc({"connectors": {"enabled": ["mail", "contacts", "settings", "phone"], "send": ["mail"]}})
        self.svc.connectors = load_connectors(self.svc.config, runner=self.runner, system="Darwin")
        self.registry = build_registry(self.svc)
        self.asked, self.answer = [], True

        def decide(approval):
            self.asked.append(approval)
            self.svc.policy.decide(approval.id, self.answer, decided_by="test")
        self.svc.policy.add_listener(decide)

    def ctx(self, said=""):
        ctx = RunContext(conversation="app:one", origin="owner")
        ctx.known |= handles(said)
        return ctx

    async def test_mail_goes_only_after_the_owner_saw_it_and_said_yes(self):
        args = {"to": "asha@example.com", "subject": "The flat", "body": "It is free from 1 March."}
        reply = await self.registry.invoke("mail_send", args, self.ctx("write to asha@example.com about the flat"))
        self.assertIn("sent", reply)
        asked = self.asked[0]
        self.assertEqual((asked.kind, asked.target, asked.risk), ("connector.send", "mail:asha@example.com", "high"))
        self.assertEqual(asked.summary, "Send this email from your Mail account now\nTo: asha@example.com\n"
                                        "Subject: The flat\nBody:\nIt is free from 1 March.")
        argv = self.runner.calls[0][0]
        self.assertEqual(argv[-4:], ["asha@example.com", "", "The flat", "It is free from 1 March."])
        # Declined: nothing runs, and the model is told so.
        self.answer, before = False, len(self.runner.calls)
        reply = await self.registry.invoke("mail_send", args, self.ctx("write to asha@example.com"))
        self.assertEqual((len(self.runner.calls), "did not approve" in reply), (before, True))
        # An address that cannot be one never reaches an approval.
        reply = await self.registry.invoke("mail_send", {**args, "to": "asha at example"}, self.ctx())
        self.assertEqual((reply.startswith("Error:"), len(self.asked)), (True, 2))

    async def test_a_recipient_the_owner_never_named_is_pointed_out(self):
        args = {"to": "billing@elsewhere.example", "subject": "Invoice", "body": "Attached."}
        await self.registry.invoke("mail_send", args, self.ctx("email the invoice to asha@example.com"))
        self.assertTrue(self.asked[-1].summary.endswith(
            "Check billing@elsewhere.example: not in what you asked, nor in your contacts."))
        # One the owner wrote, in any case, or that their contacts returned, is not.
        await self.registry.invoke("mail_send", {**args, "to": "Asha@Example.com"}, self.ctx("email asha@example.com"))
        self.assertNotIn("Check ", self.asked[-1].summary)
        self.runner.stdout = {"contacts": [{"name": "Ravi Kumar", "emails": ["ravi@example.org"], "email_labels": ["work"],
                                            "phones": ["+1 415 555 0142"], "phone_labels": ["mobile"]}], "total": 1}
        ctx = self.ctx("send Ravi the invoice")
        await self.registry.invoke("contacts_find", {"name": "Ravi"}, ctx)
        await self.registry.invoke("mail_send", {**args, "to": "ravi@example.org"}, ctx)
        self.assertNotIn("Check ", self.asked[-1].summary)
        await self.registry.invoke("phone_call", {"number": "(415) 555-0142", "name": "Ravi"}, ctx)
        self.assertEqual(self.asked[-1].summary, "Call 4155550142 (Ravi) from this Mac. FaceTime asks you to confirm, "
                                                 "and you do the talking.")
        await self.registry.invoke("phone_call", {"number": "+1 415 555 0100"}, ctx)
        self.assertIn("Check +14155550100: not in what you asked", self.asked[-1].summary)

    async def test_a_setting_is_changed_one_at_a_time_with_a_yes_each_time(self):
        self.runner.stdout = {"changed": "appearance", "now": {"appearance": "dark", "volume": 40, "muted": False}}
        reply = await self.registry.invoke("settings_set", {"setting": "appearance", "value": "dark"}, self.ctx())
        asked = self.asked[-1]
        self.assertEqual((asked.kind, asked.target, asked.summary), ("system.setting", "appearance",
                                                                    "Change this Mac's appearance to dark"))
        self.assertIn('"appearance": "dark"', reply)                    # what the Mac says afterwards
        self.assertEqual(self.runner.calls[-1][0][-2:], ["appearance", "dark"])
        calls, asks = len(self.runner.calls), len(self.asked)
        for refused in ({"setting": "firewall", "value": "off"}, {"setting": "volume", "value": "200"}):
            reply = await self.registry.invoke("settings_set", refused, self.ctx())
            self.assertTrue(reply.startswith("Error:"), reply)
        self.assertEqual((len(self.runner.calls), len(self.asked)), (calls, asks))   # refused before anyone is asked
        await self.registry.invoke("settings_get", {}, self.ctx())
        self.assertEqual(len(self.asked), asks)                         # reading them needs no approval

    def test_addresses_and_numbers_are_compared_in_one_form(self):
        self.assertEqual(handles("Mail Asha <Asha@Example.com>, or call +1 415-555-0142 / (212) 555 0187 before 5."),
                         {"asha@example.com", "4155550142", "2125550187"})
        self.assertEqual(handle("+1 415 555 0142"), handle("(415) 555-0142"))
        self.assertEqual(handles("Flat 12B, rent 42000, invoice 4471, on 14 Nov 2026"), set())


class PathTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_path_the_model_made_up_is_answered_with_the_ones_its_search_found(self):
        tmp = pathlib.Path(tempfile.mkdtemp()).resolve()
        real = tmp / "leases" / "Lease Agreement - 14 Alder St.pdf"
        real.parent.mkdir()
        real.write_text("lease")

        class Files:
            sources = None
            media = None

            def resolve(self, path):
                found = pathlib.Path(path)
                if not found.is_file():
                    raise FileAccessError(f"{path} is not a file")
                return found

            async def search(self, query, kind, limit):
                return [{"path": str(real), "kind": "doc"}, {"path": str(tmp / "notes.txt"), "kind": "text"}]

        svc = Svc()
        svc.files = Files()
        registry = build_registry(svc)
        ctx = RunContext(conversation="app:one", origin="owner")
        guessed = str(tmp / "Lease Agreement - 14 Alder St Apt 3B.pdf")
        cold = await registry.invoke("file_read", {"path": guessed}, ctx)
        self.assertNotIn("searches found", cold)                         # nothing searched yet: nothing to offer
        await registry.invoke("files_search", {"query": "lease"}, ctx)
        self.assertEqual(ctx.found_files, [str(real), str(tmp / "notes.txt")])
        reply = await registry.invoke("file_read", {"path": guessed}, ctx)
        self.assertIn(f"is not a file. This task's searches found: {real}. Use one of these paths exactly", reply)
        self.assertNotIn("notes.txt", reply)                             # only names close to the one asked for


if __name__ == "__main__":
    unittest.main()
