import asyncio
import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from coderunner.audit import AuditLog
from coderunner.scheduler import Scheduler, is_one_shot, parse_spec
from coderunner.store import Store

NY = ZoneInfo("America/New_York")
FRI = datetime(2026, 10, 9, 10, 0, tzinfo=NY)  # Friday


def ny(*args, fold=0):
    return datetime(*args, tzinfo=NY, fold=fold)


class ParseSpecTests(unittest.TestCase):
    def test_every_and_in_are_elapsed_time(self):
        self.assertEqual(parse_spec("every 30m", FRI), FRI + timedelta(minutes=30))
        self.assertEqual(parse_spec("EVERY 2 Hours", FRI), FRI + timedelta(hours=2))
        self.assertEqual(parse_spec("every 1d", FRI), FRI + timedelta(days=1))
        self.assertEqual(parse_spec("every 5m", FRI), FRI + timedelta(minutes=5))
        self.assertEqual(parse_spec("in 90m", FRI), FRI + timedelta(minutes=90))
        self.assertEqual(parse_spec("In 1m", FRI), FRI + timedelta(minutes=1))
        self.assertEqual(parse_spec("in 3 days", FRI), FRI + timedelta(days=3))

    def test_daily_is_strictly_after_now(self):
        self.assertEqual(parse_spec("daily 11:30", FRI), ny(2026, 10, 9, 11, 30))
        self.assertEqual(parse_spec("daily 08:00", FRI), ny(2026, 10, 10, 8, 0))
        self.assertEqual(parse_spec("daily 10:00", FRI), ny(2026, 10, 10, 10, 0))
        self.assertEqual(parse_spec("Daily at 7:05", FRI), ny(2026, 10, 10, 7, 5))
        self.assertEqual(parse_spec("daily 00:00", FRI), ny(2026, 10, 10, 0, 0))

    def test_weekdays_skip_weekends(self):
        self.assertEqual(parse_spec("weekdays 09:00", FRI), ny(2026, 10, 12, 9, 0))
        self.assertEqual(parse_spec("weekdays 11:00", FRI), ny(2026, 10, 9, 11, 0))
        self.assertEqual(parse_spec("weekdays 09:00", ny(2026, 10, 10, 12, 0)), ny(2026, 10, 12, 9, 0))
        self.assertEqual(parse_spec("weekdays 09:00", ny(2026, 10, 11, 23, 59)), ny(2026, 10, 12, 9, 0))
        self.assertEqual(parse_spec("weekdays 09:00", ny(2026, 10, 12, 9, 0)), ny(2026, 10, 13, 9, 0))

    def test_weekly(self):
        self.assertEqual(parse_spec("weekly mon 09:00", FRI), ny(2026, 10, 12, 9, 0))
        self.assertEqual(parse_spec("weekly fri 10:30", FRI), ny(2026, 10, 9, 10, 30))
        self.assertEqual(parse_spec("weekly friday 10:00", FRI), ny(2026, 10, 16, 10, 0))
        self.assertEqual(parse_spec("Weekly Sunday 18:00", FRI), ny(2026, 10, 11, 18, 0))
        self.assertEqual(parse_spec("weekly tues at 08:15", FRI), ny(2026, 10, 13, 8, 15))
        self.assertEqual(parse_spec("weekly thurs 08:00", FRI), ny(2026, 10, 15, 8, 0))

    def test_at(self):
        self.assertEqual(parse_spec("at 2026-12-24 18:00", FRI), ny(2026, 12, 24, 18, 0))
        self.assertEqual(parse_spec("at 2026-10-09T10:01", FRI), ny(2026, 10, 9, 10, 1))
        self.assertIsNone(parse_spec("at 2026-10-09 10:00", FRI))
        self.assertIsNone(parse_spec("at 2020-01-01 00:00", FRI))

    def test_result_is_in_nows_timezone(self):
        tokyo = FRI.astimezone(ZoneInfo("Asia/Tokyo"))
        result = parse_spec("daily 08:00", tokyo)
        self.assertEqual((result.tzinfo, result.hour), (tokyo.tzinfo, 8))
        utc = parse_spec("daily 08:00", FRI.astimezone(timezone.utc))
        self.assertEqual(utc, datetime(2026, 10, 10, 8, 0, tzinfo=timezone.utc))

    def test_invalid_specs(self):
        bad = [
            "", "   ", "sometimes", "daily", "daily 8", "daily 25:00", "daily 08:60", "daily 8am",
            "weekly 09:00", "weekly funday 09:00", "weekly mo 09:00", "every 4m", "every 0m", "every 30",
            "every 5x", "every -5m", "in 0m", "in 2 weeks", "at tomorrow", "at 2026-02-30 10:00",
            "at 2026-13-01 10:00", "at 2026-12-24 24:00", "at 2026-12-24", "cron 0 9 * * *",
            "every 99999999d", "daily 08:00 extra", "every 5m; rm -rf /",
        ]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_spec(spec, FRI)
        with self.assertRaisesRegex(ValueError, "every 5m"):
            parse_spec("every 1m", FRI)
        with self.assertRaisesRegex(ValueError, "use one of"):
            parse_spec("whenever", FRI)

    def test_naive_now_rejected(self):
        with self.assertRaises(ValueError):
            parse_spec("daily 08:00", datetime(2026, 10, 9, 10, 0))

    def test_is_one_shot(self):
        for spec in ("at 2026-12-24 18:00", "in 2h", "IN 5m"):
            self.assertTrue(is_one_shot(spec))
        for spec in ("every 30m", "daily 08:00", "weekdays 09:00", "weekly mon 09:00"):
            self.assertFalse(is_one_shot(spec))
        with self.assertRaises(ValueError):
            is_one_shot("nonsense")


class DstTests(unittest.TestCase):
    # America/New_York: clocks go back 02:00 EDT -> 01:00 EST on 2026-11-01,
    # and forward 02:00 EST -> 03:00 EDT on 2026-03-08.

    def test_daily_keeps_wall_clock_across_fall_back(self):
        result = parse_spec("daily 09:00", ny(2026, 10, 31, 10, 0))
        self.assertEqual((result.day, result.hour, result.utcoffset()), (1, 9, timedelta(hours=-5)))
        self.assertEqual(result.timestamp() - ny(2026, 10, 31, 9, 0).timestamp(), 25 * 3600)

    def test_repeated_hour_runs_once(self):
        first = parse_spec("daily 01:30", ny(2026, 11, 1, 0, 30))
        self.assertEqual((first.hour, first.minute, first.utcoffset()), (1, 30, timedelta(hours=-4)))
        after_run = datetime.fromtimestamp(first.timestamp() + 1, NY)
        second = parse_spec("daily 01:30", after_run)
        self.assertEqual((second.day, second.hour, second.utcoffset()), (2, 1, timedelta(hours=-5)))

    def test_second_pass_through_repeated_hour_is_not_in_the_past(self):
        # 01:10 EST (fold=1) is after 01:30 EDT in real time; a naive wall-clock
        # comparison would return the already-passed 01:30.
        now = ny(2026, 11, 1, 1, 10, fold=1)
        result = parse_spec("daily 01:30", now)
        self.assertGreater(result.timestamp(), now.timestamp())
        self.assertEqual((result.month, result.day, result.hour, result.minute), (11, 2, 1, 30))

    def test_interval_counts_real_time_through_fall_back(self):
        now = ny(2026, 11, 1, 1, 30)
        result = parse_spec("every 1h", now)
        self.assertEqual(result.timestamp() - now.timestamp(), 3600)
        self.assertEqual((result.hour, result.fold, result.utcoffset()), (1, 1, timedelta(hours=-5)))

    def test_skipped_time_shifts_forward_on_spring_forward(self):
        result = parse_spec("daily 02:30", ny(2026, 3, 8, 0, 0))
        self.assertEqual((result.day, result.hour, result.minute), (8, 3, 30))
        self.assertEqual(result.utcoffset(), timedelta(hours=-4))
        following = parse_spec("daily 02:30", result)
        self.assertEqual((following.day, following.hour, following.minute), (9, 2, 30))

    def test_weekly_across_transition(self):
        result = parse_spec("weekly mon 08:00", ny(2026, 10, 30, 12, 0))
        self.assertEqual((result.day, result.hour, result.utcoffset()), (2, 8, timedelta(hours=-5)))


class Harness:
    def __init__(self):
        self.runs = []
        self.delivered = []
        self.fail = set()
        self.gate = None
        self.reply = "done"

    async def run_task(self, schedule):
        self.runs.append(schedule["name"])
        if self.gate is not None:
            await self.gate.wait()
        if schedule["name"] in self.fail:
            raise RuntimeError(f"{schedule['name']} broke")
        return self.reply

    async def deliver(self, schedule, text):
        if schedule["name"] == "undeliverable":
            raise ConnectionError("channel offline")
        self.delivered.append((schedule["name"], schedule["chat_id"], text))


class SchedulerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audit = AuditLog(pathlib.Path(self.tmp.name) / "audit.jsonl")
        self.store = Store(":memory:")
        self.h = Harness()
        self.s = Scheduler(self.store, self.audit, self.h.run_task, self.h.deliver, tz="America/New_York")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def events(self, name):
        return [e for e in self.audit.tail(1000) if e["event"] == name]


class ManageTests(SchedulerTestCase):
    def test_add_list_remove(self):
        row = self.s.add(" Briefing ", "daily  08:00", "Summarise my day", "telegram", 42, now=FRI)
        self.assertEqual((row["name"], row["spec"], row["chat_id"], row["enabled"]), ("Briefing", "daily 08:00", "42", True))
        self.assertEqual(row["next_run"], ny(2026, 10, 10, 8, 0).timestamp())
        self.assertEqual([r["id"] for r in self.s.list()], [row["id"]])
        self.assertTrue(self.s.remove(row["id"]))
        self.assertFalse(self.s.remove(row["id"]))
        self.assertEqual(self.s.list(), [])
        self.assertEqual([e["event"] for e in self.audit.tail()], ["schedule.add", "schedule.remove"])

    def test_add_validates(self):
        with self.assertRaises(ValueError):
            self.s.add("x", "every 1m", "p", now=FRI)
        with self.assertRaisesRegex(ValueError, "past"):
            self.s.add("x", "at 2026-10-01 09:00", "p", now=FRI)
        with self.assertRaises(ValueError):
            self.s.add("", "daily 08:00", "p", now=FRI)
        with self.assertRaises(ValueError):
            self.s.add("x", "daily 08:00", "  ", now=FRI)
        self.assertEqual(self.s.list(), [])

    def test_bad_timezone(self):
        with self.assertRaises(ValueError):
            Scheduler(self.store, self.audit, self.h.run_task, self.h.deliver, tz="Mars/Olympus")
        local = Scheduler(self.store, self.audit, self.h.run_task, self.h.deliver)
        self.assertIsNotNone(datetime.now(local.tz).utcoffset())

    def test_set_enabled_recomputes_next_run(self):
        row = self.s.add("b", "daily 08:00", "p", now=FRI)
        self.assertTrue(self.s.set_enabled(row["id"], False))
        self.assertFalse(self.s.get(row["id"])["enabled"])
        later = ny(2026, 10, 20, 12, 0)
        self.assertTrue(self.s.set_enabled(row["id"], True, now=later))
        self.assertEqual(self.s.get(row["id"])["next_run"], ny(2026, 10, 21, 8, 0).timestamp())
        self.assertFalse(self.s.set_enabled(999, True))

    async def test_reenabling_a_past_one_shot_fails(self):
        row = self.s.add("once", "at 2026-10-09 11:00", "p", now=FRI)
        await self.s.tick(ny(2026, 10, 9, 11, 0))
        with self.assertRaises(ValueError):
            self.s.set_enabled(row["id"], True, now=ny(2026, 10, 9, 12, 0))


class TickTests(SchedulerTestCase):
    async def test_runs_only_when_due_and_delivers(self):
        row = self.s.add("brief", "daily 08:00", "Summarise", "telegram", "chat-1", now=ny(2026, 10, 9, 7, 0))
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 7, 59)), 0)
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 8, 0)), 1)
        self.assertEqual(self.h.delivered, [("brief", "chat-1", "done")])
        stored = self.s.get(row["id"])
        self.assertEqual((stored["last_status"], stored["last_run"]), ("ok", ny(2026, 10, 9, 8, 0).timestamp()))
        self.assertEqual(stored["next_run"], ny(2026, 10, 10, 8, 0).timestamp())
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 8, 0, 30)), 0)
        self.assertEqual(self.events("schedule.run")[0]["status"], "ok")

    async def test_empty_result_is_not_delivered(self):
        self.h.reply = "  "
        self.s.add("quiet", "in 5m", "p", now=FRI)
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=5)), 1)
        self.assertEqual(self.h.delivered, [])

    async def test_one_shots_disable_after_running(self):
        rel = self.s.add("rel", "in 10m", "p", now=FRI)
        at = self.s.add("abs", "at 2026-10-09 10:20", "p", now=FRI)
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=10)), 1)
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=30)), 1)
        for row in (rel, at):
            stored = self.s.get(row["id"])
            self.assertEqual((stored["enabled"], stored["next_run"], stored["last_status"]), (False, None, "ok"))
        self.assertEqual(await self.s.tick(FRI + timedelta(days=2)), 0)
        self.assertEqual(self.h.runs, ["rel", "abs"])

    async def test_failed_one_shot_still_disabled(self):
        self.h.fail.add("once")
        row = self.s.add("once", "in 5m", "p", now=FRI)
        await self.s.tick(FRI + timedelta(minutes=5))
        stored = self.s.get(row["id"])
        self.assertFalse(stored["enabled"])
        self.assertTrue(stored["last_status"].startswith("error:"))

    async def test_disabled_schedules_do_not_run(self):
        row = self.s.add("off", "every 5m", "p", now=FRI)
        self.s.set_enabled(row["id"], False)
        self.assertEqual(await self.s.tick(FRI + timedelta(hours=1)), 0)

    async def test_missed_daily_runs_once(self):
        row = self.s.add("brief", "daily 08:00", "p", now=ny(2026, 10, 5, 7, 0))
        late = ny(2026, 10, 9, 12, 0)  # four 08:00s missed while the daemon was off
        self.assertEqual(await self.s.tick(late), 1)
        self.assertEqual(await self.s.tick(late), 0)
        self.assertEqual(self.h.runs, ["brief"])
        self.assertEqual(self.s.get(row["id"])["next_run"], ny(2026, 10, 10, 8, 0).timestamp())

    async def test_missed_interval_runs_once_and_keeps_phase(self):
        row = self.s.add("poll", "every 1h", "p", now=FRI)
        self.assertEqual(await self.s.tick(FRI + timedelta(hours=5, minutes=20)), 1)
        self.assertEqual(self.h.runs, ["poll"])
        self.assertEqual(self.s.get(row["id"])["next_run"], (FRI + timedelta(hours=6)).timestamp())
        self.assertEqual(await self.s.tick(FRI + timedelta(hours=5, minutes=59)), 0)
        self.assertEqual(await self.s.tick(FRI + timedelta(hours=6, seconds=15)), 1)
        self.assertEqual(self.s.get(row["id"])["next_run"], (FRI + timedelta(hours=7)).timestamp())

    async def test_missed_one_shot_runs_once(self):
        self.s.add("reminder", "at 2026-10-09 11:00", "p", now=FRI)
        self.assertEqual(await self.s.tick(ny(2026, 10, 12, 9, 0)), 1)
        self.assertEqual(await self.s.tick(ny(2026, 10, 12, 9, 1)), 0)

    async def test_failure_does_not_stop_other_schedules(self):
        names = ["first", "broken", "undeliverable", "last"]
        rows = [self.s.add(name, "in 5m", "p", now=FRI) for name in names]
        self.h.fail.add("broken")
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=5)), 4)
        statuses = [self.s.get(r["id"])["last_status"] for r in rows]
        self.assertEqual(statuses[0], "ok")
        self.assertEqual(statuses[1], "error: broken broke")
        self.assertEqual(statuses[2], "error: delivery failed: channel offline")
        self.assertEqual(statuses[3], "ok")
        self.assertEqual([d[0] for d in self.h.delivered], ["first", "last"])
        self.assertEqual(len(self.events("schedule.run")), 4)

    async def test_hung_task_times_out(self):
        async def run_task(row):
            if row["name"] == "hung":
                await asyncio.sleep(10)
            return "fine"

        s = Scheduler(self.store, self.audit, run_task, self.h.deliver, tz="America/New_York", run_timeout=0.05)
        hung = s.add("hung", "in 5m", "p", now=FRI)
        fine = s.add("fine", "in 5m", "p", now=FRI)
        self.assertEqual(await s.tick(FRI + timedelta(minutes=5)), 2)
        self.assertEqual(s.get(hung["id"])["last_status"], "error: timed out after 0.05s")
        self.assertEqual(s.get(fine["id"])["last_status"], "ok")

    async def test_invalid_stored_spec_is_disabled_without_breaking_tick(self):
        self.store.execute(
            "INSERT INTO schedules(name, spec, prompt, next_run, enabled) VALUES ('legacy', 'cron 0 9 * * *', 'p', ?, 1)",
            (FRI.timestamp(),),
        )
        good = self.s.add("good", "in 5m", "p", now=FRI)
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=5)), 1)
        legacy = self.store.one("SELECT * FROM schedules WHERE name = 'legacy'")
        self.assertEqual(legacy["enabled"], 0)
        self.assertTrue(legacy["last_status"].startswith("error:"))
        self.assertEqual(self.s.get(good["id"])["last_status"], "ok")
        self.assertEqual(self.h.runs, ["good"])

    async def test_run_task_sees_schedule_details(self):
        seen = []

        async def run_task(schedule):
            seen.append(schedule)
            return ""

        self.s.run_task = run_task
        self.s.add("brief", "daily 08:00", "Summarise", "telegram", "c1", now=ny(2026, 10, 9, 7, 0))
        await self.s.tick(ny(2026, 10, 9, 8, 5))
        self.assertEqual((seen[0]["prompt"], seen[0]["channel"], seen[0]["chat_id"]), ("Summarise", "telegram", "c1"))
        self.assertEqual(seen[0]["scheduled_for"], ny(2026, 10, 9, 8, 0).timestamp())


class ConcurrencyTests(SchedulerTestCase):
    async def test_concurrent_ticks_run_each_schedule_once(self):
        self.h.gate = asyncio.Event()
        for name in ("a", "b", "c"):
            self.s.add(name, "every 5m", "p", now=FRI)
        due = FRI + timedelta(minutes=5)
        ticks = [asyncio.create_task(self.s.tick(due)) for _ in range(4)]
        await asyncio.sleep(0.01)
        self.h.gate.set()
        counts = await asyncio.gather(*ticks)
        self.assertEqual(sorted(self.h.runs), ["a", "b", "c"])
        self.assertEqual(sum(counts), 3)

    async def test_long_run_is_not_started_again_while_in_progress(self):
        self.h.gate = asyncio.Event()
        row = self.s.add("slow", "every 5m", "p", now=FRI)
        first = asyncio.create_task(self.s.tick(FRI + timedelta(minutes=5)))
        await asyncio.sleep(0.01)
        # Next slot has come and gone while the first run is still going.
        self.assertEqual(await asyncio.wait_for(self.s.tick(FRI + timedelta(minutes=12)), 1), 0)
        self.h.gate.set()
        self.assertEqual(await first, 1)
        self.assertEqual(self.h.runs, ["slow"])
        self.assertEqual(await self.s.tick(FRI + timedelta(minutes=12)), 1)
        self.assertEqual(self.s.get(row["id"])["next_run"], (FRI + timedelta(minutes=15)).timestamp())

    async def test_second_scheduler_on_same_store_cannot_double_run(self):
        # Simulates another process: separate in-memory guard, shared table.
        self.h.gate = asyncio.Event()
        other = Scheduler(self.store, self.audit, self.h.run_task, self.h.deliver, tz="America/New_York")
        self.s.add("shared", "daily 08:00", "p", now=ny(2026, 10, 9, 7, 0))
        due = ny(2026, 10, 9, 8, 0)
        ticks = [asyncio.create_task(self.s.tick(due)), asyncio.create_task(other.tick(due))]
        await asyncio.sleep(0.01)
        self.h.gate.set()
        self.assertEqual(sum(await asyncio.gather(*ticks)), 1)
        self.assertEqual(self.h.runs, ["shared"])

    async def test_run_forever_ticks_and_cancels(self):
        self.s.add("soon", "every 5m", "p", now=datetime.now(NY) - timedelta(minutes=10))
        task = asyncio.create_task(self.s.run_forever(interval=0.01))
        for _ in range(100):
            if self.h.runs:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.h.runs, ["soon"])

    async def test_run_forever_survives_tick_errors(self):
        calls = []

        async def flaky_tick(now=None):
            calls.append(now)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            return 0

        self.s.tick = flaky_tick
        task = asyncio.create_task(self.s.run_forever(interval=0.005))
        while len(calls) < 3:
            await asyncio.sleep(0.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.events("schedule.error")), 1)


if __name__ == "__main__":
    unittest.main()
