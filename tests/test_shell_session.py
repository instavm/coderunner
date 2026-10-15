import asyncio
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import shell_session  # noqa: E402


class ShellTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp()
        self.shells = shell_session.Shells(home=self.tmp, jobs=str(pathlib.Path(self.tmp) / "jobs"))

    async def asyncTearDown(self):
        for session in list(self.shells.shells):
            await self.shells.reset(session)

    async def test_directory_variables_and_functions_carry_over(self):
        work = pathlib.Path(self.tmp) / "project"
        await self.shells.run("s", f"mkdir -p {work} && cd {work} && export GREETING=hello && hi() {{ echo hi $1; }}")
        result = await self.shells.run("s", "echo $GREETING; hi there; pwd")
        self.assertEqual(result.output.split("\n"), ["hello", "hi there", str(work)])
        self.assertEqual((result.exit_code, result.cwd), (0, str(work)))
        other = await self.shells.run("t", "echo ${GREETING:-unset}")      # sessions are separate shells
        self.assertEqual(other.output, "unset")

    async def test_exit_codes_and_an_exit_that_ends_the_shell(self):
        self.assertEqual((await self.shells.run("s", "false")).exit_code, 1)
        ended = await self.shells.run("s", "echo bye; exit 3")
        self.assertTrue(ended.restarted)
        self.assertEqual((ended.output.strip(), ended.exit_code), ("bye", 3))
        self.assertEqual((await self.shells.run("s", "echo again")).output, "again")

    async def test_a_command_past_its_timeout_is_stopped_and_the_shell_lives_on(self):
        await self.shells.run("s", "export KEEP=1")
        began = time.monotonic()
        result = await self.shells.run("s", "echo started; sleep 30", timeout=1)
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - began, 8)
        self.assertIn("started", result.output)
        self.assertEqual((await self.shells.run("s", "echo $KEEP")).output, "1")

    async def test_stdin_is_closed_and_heredocs_work(self):
        self.assertEqual((await asyncio.wait_for(self.shells.run("s", "cat; echo eof"), 10)).output, "eof")
        self.assertEqual((await self.shells.run("s", "cat <<'EOF'\nline one\n$NOT_EXPANDED\nEOF")).output,
                         "line one\n$NOT_EXPANDED")

    async def test_long_output_keeps_its_start_and_end(self):
        result = await self.shells.run("s", "seq 1 400000; echo the-end")
        self.assertGreater(result.omitted, 0)
        self.assertTrue(result.output.startswith("1\n2\n3\n"))
        self.assertTrue(result.output.endswith("400000\nthe-end"))

    async def test_jobs_run_detached_in_the_shells_directory_with_its_variables(self):
        await self.shells.run("s", f"cd {self.tmp} && export TOKEN=7")
        job = await self.shells.start_job("s", "echo $TOKEN; pwd; sleep 1; echo done")
        self.assertTrue(job["running"])
        self.assertEqual((await self.shells.run("s", "echo free")).output, "free")    # the shell is not held
        for _ in range(60):
            if not self.shells.job(job["job"])["running"]:
                break
            await asyncio.sleep(0.1)
        found = self.shells.job(job["job"])
        self.assertEqual((found["running"], found["exit_code"]), (False, 0))
        self.assertEqual(found["output"].split(), ["7", str(pathlib.Path(self.tmp)), "done"])
        self.assertEqual([j["job"] for j in self.shells.list_jobs()], [job["job"]])

    async def test_a_job_can_be_stopped_and_unknown_jobs_are_refused(self):
        job = await self.shells.start_job("s", "sleep 30")
        stopped = self.shells.stop_job(job["job"])
        self.assertFalse(stopped["running"])
        with self.assertRaises(KeyError):
            self.shells.job("../etc")


if __name__ == "__main__":
    unittest.main()
