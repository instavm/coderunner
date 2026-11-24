import asyncio
import json
import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from coderunner.core.audit import AuditLog
from coderunner.agent.scheduler import Scheduler, canonical, is_one_shot, parse_spec
from coderunner.core.store import Store

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

    def test_several_times_a_day_run_at_the_earliest_still_to_come(self):
        self.assertEqual(parse_spec("daily 09:00,14:00,20:00", FRI), ny(2026, 10, 9, 14, 0))
        self.assertEqual(parse_spec("daily 09:00, 14:00 and 20:00", ny(2026, 10, 9, 14, 0)), ny(2026, 10, 9, 20, 0))
        self.assertEqual(parse_spec("daily 20:00,9:00,14:00", ny(2026, 10, 9, 20, 30)), ny(2026, 10, 10, 9, 0))
        self.assertEqual(parse_spec("Daily at 09:00, 14:00", ny(2026, 10, 9, 8, 59)), ny(2026, 10, 9, 9, 0))
        self.assertEqual(parse_spec("weekdays 09:00,17:30", FRI), ny(2026, 10, 9, 17, 30))
        self.assertEqual(parse_spec("weekdays 09:00,17:30", ny(2026, 10, 9, 18, 0)), ny(2026, 10, 12, 9, 0))
        self.assertEqual(parse_spec("weekly mon 09:00,17:00", ny(2026, 10, 12, 10, 0)), ny(2026, 10, 12, 17, 0))
        self.assertEqual(parse_spec("daily 09:00,09:00", FRI), ny(2026, 10, 10, 9, 0))

    def test_canonical_spec(self):
        self.assertEqual(canonical("Daily at 20:00,9:00 and 14:00"), "daily 09:00, 14:00, 20:00")
        self.assertEqual(canonical("weekdays  9:00"), "weekdays 09:00")
        self.assertEqual(canonical("Weekly Monday 17:00, 09:00"), "weekly mon 09:00, 17:00")
        self.assertEqual(canonical("every  2h"), "every 2h")
        self.assertEqual(canonical("daily 09:00, 14:00, 20:00"), "daily 09:00, 14:00, 20:00")
        with self.assertRaises(ValueError):
            canonical("twice a day")

    def test_invalid_time_lists(self):
        for spec in ("daily 09:00,", "daily ,09:00", "daily 09:00,,10:00", "daily 09:00 14:00", "daily 09:00,25:00",
                     "weekdays 09:00,9am", "daily " + ",".join(f"{h:02d}:30" for h in range(24)) + ",00:00"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_spec(spec, FRI)
        self.assertEqual(parse_spec("daily " + ",".join(f"{h:02d}:00" for h in range(24)), FRI), ny(2026, 10, 9, 11, 0))

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

    def test_several_times_through_fall_back_each_run_once(self):
        runs, now = [], ny(2026, 11, 1, 0, 0)
        for _ in range(4):
            now = parse_spec("daily 01:15,01:45,09:00", now)
            runs.append((now.day, now.hour, now.minute, now.utcoffset()))
        edt, est = timedelta(hours=-4), timedelta(hours=-5)
        self.assertEqual(runs, [(1, 1, 15, edt), (1, 1, 45, edt), (1, 9, 0, est), (2, 1, 15, est)])

    def test_a_skipped_time_comes_after_a_later_time_of_the_same_day(self):
        # 02:30 does not exist on spring-forward day and runs at 03:30 EDT, after 03:00 EDT.
        first = parse_spec("daily 02:30,03:00", ny(2026, 3, 8, 0, 0))
        self.assertEqual((first.day, first.hour, first.minute), (8, 3, 0))
        second = parse_spec("daily 02:30,03:00", first)
        self.assertEqual((second.day, second.hour, second.minute), (8, 3, 30))
        third = parse_spec("daily 02:30,03:00", second)
        self.assertEqual((third.day, third.hour, third.minute), (9, 2, 30))


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

    def test_spec_is_stored_in_one_form(self):
        row = self.s.add("deals", "Daily 20:00,9:00 and 14:00", "Find deals", "app", "c1", now=FRI)
        self.assertEqual(row["spec"], "daily 09:00, 14:00, 20:00")
        self.assertEqual(row["next_run"], ny(2026, 10, 9, 14, 0).timestamp())

    def test_update_changes_what_is_given(self):
        row = self.s.add("deals", "daily 09:00", "Find deals", "app", "c1", now=FRI)
        updated = self.s.update(row["id"], spec="daily 09:00,14:00,20:00", now=FRI)
        self.assertEqual((updated["spec"], updated["prompt"], updated["name"]), ("daily 09:00, 14:00, 20:00", "Find deals", "deals"))
        self.assertEqual(updated["next_run"], ny(2026, 10, 9, 14, 0).timestamp())
        renamed = self.s.update(row["id"], name=" Deal hunt ", prompt=" Find deals on two sites ")
        self.assertEqual((renamed["name"], renamed["prompt"]), ("Deal hunt", "Find deals on two sites"))
        self.assertEqual(renamed["next_run"], updated["next_run"])       # the time is kept when the spec is
        self.assertEqual(self.s.update(row["id"]), renamed)              # nothing to change
        self.assertIsNone(self.s.update(999, name="x"))
        for bad in ({"spec": "twice a day"}, {"spec": "at 2026-10-01 09:00"}, {"name": " "}, {"prompt": ""}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.s.update(row["id"], now=FRI, **bad)
        self.assertEqual(self.s.get(row["id"]), renamed)                 # a refused change changes nothing
        updates = self.events("schedule.update")
        self.assertEqual([e["fields"] for e in updates], [["spec"], ["name", "prompt"]])

    async def test_update_sets_a_finished_one_shot_again(self):
        row = self.s.add("once", "in 10m", "p", now=FRI)
        await self.s.tick(FRI + timedelta(minutes=10))
        self.assertFalse(self.s.get(row["id"])["enabled"])
        again = self.s.update(row["id"], spec="in 1h", now=FRI + timedelta(minutes=20))
        self.assertTrue(again["enabled"])
        self.assertEqual(again["next_run"], (FRI + timedelta(minutes=80)).timestamp())

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

    async def test_several_times_a_day_each_run_once_and_missed_ones_collapse(self):
        row = self.s.add("deals", "daily 09:00,14:00,20:00", "p", now=ny(2026, 10, 9, 8, 0))
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 9, 0)), 1)
        self.assertEqual(self.s.get(row["id"])["next_run"], ny(2026, 10, 9, 14, 0).timestamp())
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 13, 59)), 0)
        self.assertEqual(await self.s.tick(ny(2026, 10, 9, 14, 1)), 1)
        self.assertEqual(self.s.get(row["id"])["next_run"], ny(2026, 10, 9, 20, 0).timestamp())
        # Off from before 20:00 until the next afternoon: one catch-up run, then the next time still to come.
        self.assertEqual(await self.s.tick(ny(2026, 10, 10, 15, 0)), 1)
        self.assertEqual(await self.s.tick(ny(2026, 10, 10, 15, 0)), 0)
        self.assertEqual(self.s.get(row["id"])["next_run"], ny(2026, 10, 10, 20, 0).timestamp())
        self.assertEqual(self.h.runs, ["deals"] * 3)

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


class RoutineToolTests(unittest.IsolatedAsyncioTestCase):
    """Routines set up, changed and removed from a conversation, as the model does it."""

    def setUp(self):
        from fakes import Svc
        from coderunner.agent import ToolRegistry
        from coderunner.agent.tools import schedule_tools
        self.svc = Svc()
        h = Harness()
        self.svc.scheduler = Scheduler(self.svc.store, self.svc.audit, h.run_task, h.deliver, tz="America/New_York")
        self.registry = ToolRegistry(self.svc, schedule_tools(self.svc))
        self.shown = []

    def ctx(self, chat="c1", owner=True, tainted=False):
        from coderunner.agent import RunContext
        ctx = RunContext(conversation=f"app:{chat}", origin="owner" if owner else "telegram:someone", channel="app",
                         chat_id=chat, origin_is_owner=owner, events=self.shown.append)
        if tainted:
            ctx.taint()
        return ctx

    async def call(self, name, args, ctx):
        return await self.registry.invoke(name, args, ctx)

    async def asked(self, name, args, ctx):
        """Starts the call and returns it with the approval it waits for."""
        call = asyncio.create_task(self.call(name, args, ctx))
        for _ in range(300):
            pending = self.svc.policy.pending()
            if pending:
                return call, pending[0]
            await asyncio.sleep(0.01)
        self.fail(f"{name} did not ask: {await call}")

    def lines(self):
        return [(e["action"], e["name"], e["spec"]) for e in self.shown if e["type"] == "routine"]

    def approvals(self):
        return self.svc.store.query("SELECT kind, status FROM approvals ORDER BY created")

    async def test_the_owner_sets_up_and_changes_a_routine_in_their_conversation_without_being_asked(self):
        made = await self.call("schedule_create", {"name": "Deal hunt", "spec": "daily 09:00",
                                                   "prompt": "Find the best deals on amazon.in"}, self.ctx())
        self.assertTrue(made.startswith("Created routine 1 'Deal hunt' (daily 09:00)"), made)
        row = self.svc.scheduler.get(1)
        self.assertEqual((row["channel"], row["chat_id"]), ("app", "c1"))
        self.assertFalse(self.svc.store.get_kv("schedule:tainted:1"))
        # "Do it 3 times a day", in the next message: no id, this conversation's routine.
        changed = await self.call("schedule_update", {"spec": "daily 9:00, 14:00, 20:00"}, self.ctx())
        self.assertTrue(changed.startswith("Updated routine 1 'Deal hunt' (daily 09:00, 14:00, 20:00)"), changed)
        self.assertEqual(self.svc.scheduler.get(1)["spec"], "daily 09:00, 14:00, 20:00")
        self.assertEqual(self.approvals(), [])                          # never asked
        listed = json.loads(await self.call("schedule_list", {}, self.ctx()))
        self.assertEqual([(r["id"], r["this_conversation"]) for r in listed], [(1, True)])
        # From another conversation it is not "this conversation's".
        self.assertIn("no routine", await self.call("schedule_update", {"spec": "daily 10:00"}, self.ctx("c2")))
        self.assertFalse(json.loads(await self.call("schedule_list", {}, self.ctx("c2")))[0]["this_conversation"])
        removed = json.loads(await self.call("schedule_delete", {}, self.ctx()))
        self.assertEqual((removed["removed"], removed["name"]), (True, "Deal hunt"))
        self.assertEqual(self.svc.scheduler.list(), [])
        self.assertEqual(self.lines(), [("created", "Deal hunt", "daily 09:00"),
                                        ("updated", "Deal hunt", "daily 09:00, 14:00, 20:00"),
                                        ("removed", "Deal hunt", "daily 09:00, 14:00, 20:00")])

    async def test_after_untrusted_content_or_for_someone_else_the_owner_is_asked(self):
        args = {"name": "Deal hunt", "spec": "daily 09:00", "prompt": "Find deals"}
        call, approval = await self.asked("schedule_create", args, self.ctx(tainted=True))
        self.assertEqual((approval.kind, approval.target), ("schedule.create", "daily 09:00"))
        self.svc.policy.decide(approval.id, True)
        self.assertTrue((await call).startswith("Created routine"))
        self.assertTrue(self.svc.store.get_kv("schedule:tainted:1"))   # its runs start untrusted
        # Changed after untrusted content: asked again.
        call, approval = await self.asked("schedule_update", {"spec": "daily 09:00,20:00"}, self.ctx(tainted=True))
        self.assertEqual(approval.kind, "schedule.update")
        self.svc.policy.decide(approval.id, False)
        self.assertIn("did not approve", await call)
        self.assertEqual(self.svc.scheduler.get(1)["spec"], "daily 09:00")
        # The owner rewriting its prompt in their own words: not asked, and its runs are trusted again.
        await self.call("schedule_update", {"prompt": "Find deals on amazon.in"}, self.ctx())
        self.assertFalse(self.svc.store.get_kv("schedule:tainted:1"))
        # Someone else asking in their own chat is asked about, and nothing is made when declined.
        call, approval = await self.asked("schedule_create", args, self.ctx("c9", owner=False))
        self.assertEqual(approval.kind, "schedule.create")
        self.svc.policy.decide(approval.id, False)
        self.assertIn("did not approve", await call)
        self.assertEqual(len(self.svc.scheduler.list()), 1)
        # A spec that is not understood is said at once, before anyone is asked.
        self.assertIn("unrecognised schedule", await self.call(
            "schedule_create", {**args, "spec": "twice a day"}, self.ctx(tainted=True)))
        self.assertEqual([a["kind"] for a in self.approvals()], ["schedule.create", "schedule.update", "schedule.create"])

    async def test_several_routines_in_a_conversation_are_named_by_id(self):
        for name in ("Deals", "News"):
            await self.call("schedule_create", {"name": name, "spec": "daily 09:00", "prompt": name}, self.ctx())
        await self.call("schedule_create", {"name": "Theirs", "spec": "daily 09:00", "prompt": "x"}, self.ctx("c9"))
        said = await self.call("schedule_update", {"spec": "daily 10:00"}, self.ctx())
        self.assertIn("several routines: 1 'Deals' (daily 09:00), 2 'News' (daily 09:00)", said)
        self.assertIn("Updated routine 2", await self.call("schedule_update", {"id": 2, "name": "Headlines"}, self.ctx()))
        # Someone else sees and changes only the routines of their own chat.
        other = self.ctx("c9", owner=False)
        self.assertEqual([r["name"] for r in json.loads(await self.call("schedule_list", {}, other))], ["Theirs"])
        self.assertIn("no routine 1", await self.call("schedule_delete", {"id": 1}, other))
        self.assertEqual(len(self.svc.scheduler.list()), 3)


if __name__ == "__main__":
    unittest.main()
