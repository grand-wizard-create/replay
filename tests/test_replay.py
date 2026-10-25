"""Replay: recordings, the motion rules on fake arms, the simulated pair, monorobot start, and the built package."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import pytest
from conftest import (
    AL5D_HOME_DEG,
    APP_DIR,
    MID_DEG,
    REST_DEG,
    FakeAL5D,
    FakeArm,
    FakeLeader,
    LineReader,
    al5d_envelope_deg,
    assert_al5d_motion_ok,
    assert_motion_ok,
    built_manifest,
    envelope_deg,
    make_app,
    more_ticks,
    run_paced,
    run_ticks,
    run_under_sdk,
    sim_joints_deg,
    sim_pair_copy,
    start_on_sim_pair,
    urdf_limits_deg,
    wait_until_still,
)

VERSION = "0.1.0"
WHATS_NEW = "First version."
RATE_HZ = 60
FRAMES = 450                                     # 15 s at 30 frames a second
FOLLOW_DEG_PER_S = 3.0 * RATE_HZ                 # following the leader: at most 3 degrees a tick, a hand in the loop
PLAY_DEG_PER_S = 45.0                            # D37: playing back, no hand in the loop, at most 45 degrees a second
RECORD_PHASES = {"glide": 30.0, "record": FOLLOW_DEG_PER_S, "to_start": 45.0, "play": PLAY_DEG_PER_S}
DEMO_PHASES = {"to_start": 45.0, "play": 45.0}
GRIPPER_STEPS = {"record": 0.1, "play": 0.1}
SO101_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
AL5D_NAMES = ["base", "shoulder", "elbow", "wrist"]
UNKNOWN = "Replay doesn't know this robot."
RECORDING = re.compile(r"so101-\d{8}-\d{6}(-\d+)?\.jsonl")
STARTS = {"rest": REST_DEG, "mid": MID_DEG}
LONG = 60 * RATE_HZ                              # more ticks than any run here needs


def slow_path(t: float):
    """A hand moving the leader around a pose in front of the base, as the simulated leader moves."""
    mid = (0.0, -40.0, 20.0, 20.0, 0.0)
    joints = [m + 20.0 * math.sin(2 * math.pi * t / 8.0 + i * math.pi / 3) for i, m in enumerate(mid)]
    return joints, 0.5 + 0.3 * math.sin(2 * math.pi * t / 8.0 + math.pi)


def hand(path):
    """A ``before_tick`` that puts the leader where ``path(seconds)`` says, each tick."""

    def move(app) -> None:
        leader = app.peers.get("leader")
        if leader is not None:
            joints, opening = path(app.tick / RATE_HZ)
            leader.joints_deg, leader.opening = list(joints), opening

    return move


def record_and_play(main, arm=None, leader=None, path=slow_path):
    """Replay with a leader on fakes, from setup to finish(): it records, then plays it back."""
    arm = arm or FakeArm(MID_DEG, gripper=0.2)
    leader = leader or FakeLeader()
    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, LONG, before_tick=hand(path))
    return app, arm, leader


def play(main, arm=None, robot_type="hf_so101_follower", peers=None):
    """Replay without a leader on a fake arm, from setup to finish()."""
    arm = arm or FakeArm(MID_DEG)
    app = make_app(main, arm, robot_type, peers=peers)
    run_ticks(app, LONG)
    return app, arm


def write_recording(folder, name, frames, *, joints=SO101_NAMES, rate_hz=30, schema="replay/1", tail=""):
    folder.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"schema": schema, "robot_type": "hf_so101_follower", "rate_hz": rate_hz, "joints": joints})]
    lines += [json.dumps({"t": round(i / 30, 3), "joints_deg": list(j), "gripper": g})
              for i, (j, g) in enumerate(frames)]
    (folder / name).write_text("\n".join(lines) + "\n" + tail)
    return folder / name


def short_recording(n=60, joints=5):
    """``n`` frames moving the first joint 1 degree a frame from a mid pose."""
    base = list(MID_DEG[:joints]) if joints == 5 else list(AL5D_HOME_DEG)
    return [([base[0] + i, *base[1:]], 0.5) for i in range(n)]


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def so101_envelope(main):
    return [main.SO101_ENVELOPE_DEG[name] for name in main.JOINTS["so101"]]


# --------------------------------------------------------------------------- setup


def test_needs_a_robot_it_knows(main, app_data):
    app = make_app(main, FakeArm(), robot_type="unitree_go2", peers={"leader": FakeLeader()})
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == UNKNOWN
    app.on_shutdown()
    assert not app_data.exists()


@pytest.mark.parametrize("with_leader", [True, False])
def test_nothing_moves_in_setup(main, app_data, with_leader):
    arm, leader = FakeArm(REST_DEG), FakeLeader()
    app = make_app(main, arm, peers={"leader": leader} if with_leader else {})
    app.setup()
    assert arm.targets == []
    assert leader.reads == 0
    assert app.mode == ("record" if with_leader else "play")
    assert not app_data.exists()                     # created on the first write only


def test_works_on_an_engine_that_sets_no_peers(main, app_data):
    app = make_app(main, FakeArm())
    del app.peers
    app.setup()
    assert app.mode == "play"


def test_the_leader_counts_only_on_the_so101(main, app_data):
    leader = FakeLeader()
    app, _ = play(main, FakeAL5D(), robot_type="lynxmotion_al5d", peers={"leader": leader})
    assert app.mode == "play" and app.kind == "al5d"
    assert leader.reads == 0
    assert app.finish_calls


def test_the_simulated_follower_is_an_so101(main, app_data):
    app = make_app(main, FakeArm(), robot_type="sim_so101_follower", peers={"leader": FakeLeader()})
    app.setup()
    assert app.kind == "so101" and app.mode == "record"


# --------------------------------------------------------------------------- recordings


def test_header_and_frames_round_trip(main, tmp_path):
    folder = tmp_path / "recordings"
    recorder = main.Recorder(folder, "so101", "hf_so101_follower", 30.0)
    frames = [([1.23456, -40.0, 20.5, 19.0, -0.0004], 0.4321), ([2.0, -39.0, 21.0, 18.0, 1.0], 1.0)]
    for joints, gripper in frames:
        recorder.add(joints, gripper)
    recorder.close()
    assert RECORDING.fullmatch(recorder.path.name), recorder.path.name
    lines = read_lines(recorder.path)
    assert lines[0] == {"schema": "replay/1", "robot_type": "hf_so101_follower", "rate_hz": 30, "joints": SO101_NAMES}
    assert isinstance(lines[0]["rate_hz"], int)
    assert lines[1:] == [
        {"t": 0.0, "joints_deg": [1.235, -40.0, 20.5, 19.0, -0.0], "gripper": 0.432},
        {"t": 0.033, "joints_deg": [2.0, -39.0, 21.0, 18.0, 1.0], "gripper": 1.0},
    ]
    recording = main.load_recording(recorder.path, "so101", so101_envelope(main))
    assert recording.name == recorder.path.name and recording.rate_hz == 30.0
    assert recording.frames == recorder.frames
    assert recording.frames == [([1.235, -40.0, 20.5, 19.0, -0.0], 0.432), ([2.0, -39.0, 21.0, 18.0, 1.0], 1.0)]


def test_a_recording_is_flushed_every_second(main, tmp_path):
    recorder = main.Recorder(tmp_path, "so101", "hf_so101_follower", 30.0)
    assert len(recorder.path.read_text().splitlines()) == 1          # the header at once
    for i in range(60):
        recorder.add([float(i), 0.0, 0.0, 0.0, 0.0], 0.5)
        if i + 1 in (30, 60):
            assert len(recorder.path.read_text().splitlines()) == 1 + i + 1
    recorder.close()


def test_two_recordings_in_the_same_second(main, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "stamp", lambda: "20261016-101500")
    first = main.Recorder(tmp_path, "so101", "hf_so101_follower", 30.0)
    second = main.Recorder(tmp_path, "so101", "hf_so101_follower", 30.0)
    first.close()
    second.close()
    assert (first.path.name, second.path.name) == ("so101-20261016-101500.jsonl", "so101-20261016-101500-2.jsonl")
    assert main.recordings("so101", tmp_path) == [first.path, second.path]


def test_newest_file_choice(main, tmp_path):
    frames = short_recording(3)
    for name in ("so101-20261015-235959.jsonl", "so101-20261016-101500.jsonl", "so101-20261016-101500-2.jsonl",
                 "so101-20261016-101500-10.jsonl", "so101-20261016-090000.jsonl"):
        write_recording(tmp_path, name, frames)
    write_recording(tmp_path, "al5d-20261017-000000.jsonl", short_recording(3, 4), joints=AL5D_NAMES)
    write_recording(tmp_path, "so101-latest.jsonl", frames)
    (tmp_path / "notes.txt").write_text("not a recording")
    (tmp_path / "so101-20261018-000000.jsonl").mkdir()
    names = [p.name for p in main.recordings("so101", tmp_path)]
    assert names == ["so101-20261015-235959.jsonl", "so101-20261016-090000.jsonl", "so101-20261016-101500.jsonl",
                     "so101-20261016-101500-2.jsonl", "so101-20261016-101500-10.jsonl"]
    assert main.newest_recording("so101", so101_envelope(main), tmp_path).name == "so101-20261016-101500-10.jsonl"
    assert main.newest_recording("al5d", al5d_envelope_deg(), tmp_path).name == "al5d-20261017-000000.jsonl"


def test_a_newest_file_that_is_not_a_recording_is_skipped(main, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="replay")
    write_recording(tmp_path, "so101-20261016-101500.jsonl", short_recording(3))
    write_recording(tmp_path, "so101-20261016-101600.jsonl", short_recording(3), schema="other/1")
    assert main.newest_recording("so101", so101_envelope(main), tmp_path).name == "so101-20261016-101500.jsonl"
    assert [r.getMessage() for r in caplog.records] == [
        "Skipped so101-20261016-101600.jsonl: it isn't a recording Replay can play."
    ]


@pytest.mark.parametrize("bad", ["schema", "joints", "rate", "no frames", "nan", "empty", "not json"])
def test_files_that_are_not_recordings(main, tmp_path, bad):
    path = tmp_path / "so101-20261016-101500.jsonl"
    frames = short_recording(3)
    if bad == "schema":
        write_recording(tmp_path, path.name, frames, schema="replay/2")
    elif bad == "joints":
        write_recording(tmp_path, path.name, frames, joints=AL5D_NAMES)
    elif bad == "rate":
        write_recording(tmp_path, path.name, frames, rate_hz=0)
    elif bad == "no frames":
        write_recording(tmp_path, path.name, [])
    elif bad == "nan":
        write_recording(tmp_path, path.name, [([float("nan"), 0.0, 0.0, 0.0, 0.0], 0.5)])
    elif bad == "empty":
        path.write_text("")
    else:
        path.write_text("hello\n")
    assert main.load_recording(path, "so101", so101_envelope(main)) is None


def test_a_recording_cut_short_keeps_the_frames_before_the_break(main, tmp_path):
    path = write_recording(tmp_path, "so101-20261016-101500.jsonl", short_recording(3), tail='{"t": 0.1, "joi')
    recording = main.load_recording(path, "so101", so101_envelope(main))
    assert len(recording.frames) == 3


def test_frames_are_held_inside_the_envelope(main, tmp_path):
    path = write_recording(tmp_path, "so101-20261016-101500.jsonl", [([90.0, -120.0, 120.0, -120.0, 200.0], 1.7)])
    (joints, gripper), = main.load_recording(path, "so101", so101_envelope(main)).frames
    assert joints == [hi if v > 0 else lo for v, (lo, hi) in zip([90.0, -120.0, 120.0, -120.0, 200.0],
                                                                    so101_envelope(main))]
    assert gripper == 1.0
    assert not any(v for v in [max(lo - j, j - hi, 0.0) for j, (lo, hi) in zip(joints, envelope_deg())])


def test_trim_keeps_the_20_newest(main, app_data):
    folder = app_data / "recordings"
    for minute in range(25):
        write_recording(folder, f"so101-20261016-10{minute:02d}00.jsonl", short_recording(3))
    write_recording(folder, "al5d-20261016-090000.jsonl", short_recording(3, 4), joints=AL5D_NAMES)
    record_and_play(main)
    so101 = main.recordings("so101", folder)
    assert len(so101) == 20
    assert so101[0].name == "so101-20261016-100600.jsonl"     # the 6 oldest are gone, the new one is kept
    assert RECORDING.fullmatch(so101[-1].name) and so101[-1].name > "so101-20261016-102400.jsonl"
    assert (folder / "al5d-20261016-090000.jsonl").exists()   # another arm's recordings stay


def test_trim_leaves_other_files_alone(main, tmp_path):
    for minute in range(22):
        write_recording(tmp_path, f"so101-20261016-10{minute:02d}00.jsonl", short_recording(1))
    (tmp_path / "notes.txt").write_text("mine")
    removed = main.trim("so101", tmp_path)
    assert [p.name for p in removed] == ["so101-20261016-100000.jsonl", "so101-20261016-100100.jsonl"]
    assert (tmp_path / "notes.txt").read_text() == "mine"


# --------------------------------------------------------------------------- the data folder


def test_data_dir_follows_MONOROBOT_APP_DATA(main, tmp_path, monkeypatch):
    monkeypatch.setenv("MONOROBOT_APP_DATA", str(tmp_path / "given"))
    assert main.data_dir() == tmp_path / "given"
    assert main.recordings_dir() == tmp_path / "given" / "recordings"
    monkeypatch.delenv("MONOROBOT_APP_DATA")
    assert main.data_dir() == tmp_path / "monorobot-home" / "data" / "replay"
    monkeypatch.delenv("MONOROBOT_HOME")
    assert main.data_dir() == tmp_path / "home" / ".monorobot" / "data" / "replay"   # the sandboxed home's


def test_the_tests_never_see_the_real_home(tmp_path):
    """CONTRACTS 0.5: HOME and MONOROBOT_HOME both point under the test's temp folder."""
    assert Path.home().resolve().is_relative_to(tmp_path.resolve())
    assert Path(os.environ["MONOROBOT_HOME"]).resolve().is_relative_to(tmp_path.resolve())


def test_records_into_the_data_folder_and_nowhere_else(main, tmp_path, app_data):
    def files(root):
        return {p for p in root.rglob("*") if p.is_file() and not {"__pycache__", ".pytest_cache", ".ruff_cache"}
                & set(p.relative_to(root).parts)}

    app_files = files(APP_DIR)
    record_and_play(main)
    written = files(tmp_path)
    assert written and all(p.is_relative_to(app_data / "recordings") for p in written), written
    assert len(written) == 1
    assert files(APP_DIR) == app_files


# --------------------------------------------------------------------------- record, then play


def test_record_then_play(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    app, arm, leader = record_and_play(main)
    assert app.finish_calls == [app.tick - 1] and app.phase == "done"
    path, = (app_data / "recordings").iterdir()
    assert RECORDING.fullmatch(path.name)
    lines = read_lines(path)
    assert lines[0] == {"schema": "replay/1", "robot_type": "hf_so101_follower", "rate_hz": 30, "joints": SO101_NAMES}
    assert len(lines) == 1 + FRAMES
    assert [line["t"] for line in lines[1:4]] == [0.0, 0.033, 0.067] and lines[-1]["t"] == round(449 / 30, 3)
    # every second tick of the recording, as the follower was sent it
    record = arm.sent("record")
    assert len(record) == 15 * RATE_HZ
    for line, joints in zip(lines[1:], record[::2]):
        assert line["joints_deg"] == pytest.approx(joints, abs=1e-3)
    # then the glide back to the first frame, and every frame played, one tick each between them
    assert arm.sent("to_start")[-1] == pytest.approx(lines[1]["joints_deg"])
    played = arm.sent("play")
    assert len(played) == 2 * (FRAMES - 1)
    for line, joints in zip(lines[2:], played[1::2]):
        assert joints == pytest.approx(line["joints_deg"])
    assert arm.grippers("play")[-1] == pytest.approx(lines[-1]["gripper"])
    assert [name for name, _ in phase_runs(arm)] == ["glide", "record", "to_start", "play"]
    assert leader.reads == len(arm.sent("glide")) + len(record)     # the leader isn't read while playing
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


@pytest.mark.parametrize("start", sorted(STARTS))
def test_record_and_play_follow_the_rules(main, app_data, start):
    arm = FakeArm(STARTS[start], gripper=0.1)
    record_and_play(main, arm)
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)
    assert {method for method, *_ in arm.targets} == {"send_teleop_action"}


def test_a_leader_moved_past_the_envelope_is_recorded_at_its_edge(main, app_data):
    far = [90.0, -100.0, 96.0, -95.0, 160.0]
    _, arm, _ = record_and_play(main, leader=FakeLeader(far, opening=1.0), path=lambda t: (far, 1.0))
    path, = (app_data / "recordings").iterdir()
    edge = main.clip(far, so101_envelope(main))
    assert read_lines(path)[-1]["joints_deg"] == pytest.approx(edge, abs=1e-3)
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_the_glide_onto_the_leader_is_speed_based(main, app_data):
    """As Teleop's: a 150 degree gap takes 5 s, at most 30 degrees a second, and ends on the leader."""
    start = [0.0, -40.0, 45.0, -15.0, -75.0]
    goal = [0.0, -40.0, 45.0, -15.0, 75.0]
    arm = FakeArm(start)
    app = make_app(main, arm, peers={"leader": FakeLeader(goal)})
    run_ticks(app, 6 * RATE_HZ)
    glide = arm.sent("glide")
    assert len(glide) / RATE_HZ == pytest.approx(5.0, abs=0.2)
    assert max(abs(b[4] - a[4]) for a, b in zip([start, *glide], glide)) == pytest.approx(30.0 / RATE_HZ)
    assert glide[-1] == pytest.approx(goal, abs=1e-9)
    # The leader is still: no tick, the first recorded one included, moves faster than the glide.
    assert_motion_ok(arm, RATE_HZ, {"glide": 30.0, "record": 30.0})


@pytest.mark.parametrize("gap_deg", [0.0, 2.5, 10.0])
def test_a_small_gap_glides_for_at_least_a_second(main, app_data, gap_deg):
    """D37, as Teleop's: a small gap glides for 1 s, and the first recorded frame is the leader's pose."""
    start = list(MID_DEG)
    goal = [MID_DEG[0] + gap_deg, *MID_DEG[1:]]
    arm = FakeArm(start)
    app = make_app(main, arm, peers={"leader": FakeLeader(goal)})
    run_ticks(app, 2 * RATE_HZ)
    glide = arm.sent("glide")
    assert len(glide) == RATE_HZ
    assert glide[-1] == pytest.approx(goal, abs=1e-9)
    assert arm.sent("record")[0] == pytest.approx(goal, abs=1e-9)
    assert_motion_ok(arm, RATE_HZ, {"glide": 30.0, "record": 30.0})


def test_play_back_after_recording_glides_to_the_first_frame(main, app_data):
    """The recording ends far from where it began: the arm glides back at 45 degrees a second at most."""

    def sweep(t):                          # the base swings 50 degrees over the 15 s
        return [min(50.0, 50.0 * t / 15.0) - 25.0, -40.0, 20.0, 20.0, 0.0], 0.5

    _, arm, _ = record_and_play(main, FakeArm([-25.0, -40.0, 20.0, 20.0, 0.0]), path=sweep)
    to_start = arm.sent("to_start")
    assert to_start[-1][0] == pytest.approx(read_lines(next((app_data / "recordings").iterdir()))[1]["joints_deg"][0])
    assert len(to_start) >= 1.0 * RATE_HZ
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_an_unusable_leader_reading_is_skipped(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    arm, leader = FakeArm(MID_DEG), FakeLeader(MID_DEG)
    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, 10)
    leader.joints_deg[1] = float("nan")
    more_ticks(app, 3)
    assert len(arm.targets) == 10
    leader.joints_deg[1] = MID_DEG[1]
    more_ticks(app, 3)
    assert len(arm.targets) == 13
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["The leader sent a reading that isn't usable, so the follower waits for the next one."]


def test_a_stop_while_recording_keeps_what_was_recorded(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    app = make_app(main, FakeArm(MID_DEG), peers={"leader": FakeLeader(MID_DEG)})
    run_ticks(app, 4 * RATE_HZ)                      # 1 s gliding, 3 s recording
    assert app.phase == "record"
    app.on_shutdown()
    path, = (app_data / "recordings").iterdir()
    frames = len(read_lines(path)) - 1
    assert 80 <= frames <= 90
    assert caplog.records[-1].getMessage() == f"Stopped. Kept the {frames} frames recorded so far."
    assert main.load_recording(path, "so101", so101_envelope(main)) is not None


def test_a_stop_before_recording_writes_nothing(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    app = make_app(main, FakeArm(REST_DEG), peers={"leader": FakeLeader(MID_DEG)})
    run_ticks(app, 30)
    assert app.phase == "glide"
    app.on_shutdown()
    assert not app_data.exists()
    assert caplog.records[-1].getMessage() == "Stopped."


def test_a_folder_it_cannot_write_is_one_plain_sentence(main, tmp_path, monkeypatch):
    blocker = tmp_path / "data"
    blocker.write_text("a file where the folder should be")
    monkeypatch.setenv("MONOROBOT_APP_DATA", str(blocker / "replay"))
    app = make_app(main, FakeArm(MID_DEG), peers={"leader": FakeLeader(MID_DEG)})
    with pytest.raises(RuntimeError, match=r"^Replay can't save a recording in .*\.$"):
        run_ticks(app, 5 * RATE_HZ)


def test_logs_are_short_and_plain(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    app, _, _ = record_and_play(main)
    app.on_shutdown()
    lines = [r.getMessage() for r in caplog.records if r.name == "replay"]
    assert lines == ["Reading the leader.", "Recording 15 s: move the leader.", "Recording, 10 s left.",
                     "Recording, 5 s left.", "Recorded 450 frames, playing it back.", "Done."]
    for line in lines:
        assert line[0].isupper() and line.endswith(".") and "\u2014" not in line
        assert "hf_so101" not in line and "lynxmotion" not in line and "monorobot" not in line.lower()


def phase_runs(arm):
    runs: list[tuple[str, int]] = []
    for label in arm.labels():
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


# --------------------------------------------------------------------------- play without the leader


@pytest.mark.parametrize("start", sorted(STARTS))
def test_without_the_leader_it_plays_the_demo(main, app_data, caplog, start):
    caplog.set_level(logging.INFO, logger="replay")
    app, arm = play(main, FakeArm(STARTS[start], gripper=0.0))
    demo = main.demo_frames("so101", so101_envelope(main))
    assert [r.getMessage() for r in caplog.records if r.name == "replay"] == ["Playing the demo.", "Done."]
    assert app.finish_calls and len(app.finish_calls) == 1
    assert arm.sent("to_start")[-1] == pytest.approx(demo[0][0])
    assert arm.sent("play")[-1] == pytest.approx(demo[-1][0])
    assert len(arm.sent("play")) == pytest.approx(8.0 * RATE_HZ, abs=2)
    assert_motion_ok(arm, RATE_HZ, DEMO_PHASES, gripper_steps=GRIPPER_STEPS)
    assert not app_data.exists()


def test_without_the_leader_it_plays_the_newest_recording(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    older = write_recording(app_data / "recordings", "so101-20261016-101500.jsonl", short_recording(30))
    newest = [([MID_DEG[0], MID_DEG[1] - i, *MID_DEG[2:]], 0.3) for i in range(60)]
    write_recording(app_data / "recordings", "so101-20261016-101600.jsonl", newest)
    _, arm = play(main, FakeArm(MID_DEG))
    assert caplog.records[0].getMessage() == "Playing so101-20261016-101600.jsonl."
    assert arm.sent("play")[-1] == pytest.approx(newest[-1][0])
    assert len(arm.sent("play")) == 2 * 59
    assert older.exists()                               # playing deletes nothing
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_a_recording_with_jumps_plays_at_45_degrees_a_second(main, app_data):
    """D37: a 40 then an 80 degree jump between frames plays at 45 degrees a second, along the
    recording's own path (out to the first jump's end, then back), so it lasts longer than the file."""
    jumpy = [(list(MID_DEG), 0.5), ([MID_DEG[0] + 40.0, *MID_DEG[1:]], 0.5), ([MID_DEG[0] - 40.0, *MID_DEG[1:]], 1.0)]
    write_recording(app_data / "recordings", "so101-20261016-101500.jsonl", jumpy)
    _, arm = play(main, FakeArm(MID_DEG))
    played = arm.sent("play")
    assert played[-1] == pytest.approx(jumpy[-1][0])
    assert max(joints[0] for joints in played) == pytest.approx(jumpy[1][0][0])
    assert len(played) >= (40.0 + 80.0) / (PLAY_DEG_PER_S / RATE_HZ)
    steps = [abs(b[0] - a[0]) for a, b in itertools.pairwise(arm.sent())]
    assert max(steps) == pytest.approx(PLAY_DEG_PER_S / RATE_HZ)
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_a_header_with_too_high_a_rate_still_plays_at_45_degrees_a_second(main, app_data):
    """A recording that moves 15 degrees a second, with a header that says 1000 frames a second
    instead of 30: its frames come 33 times too fast, and the playback holds them to 45 degrees a second."""
    smooth = [([MID_DEG[0] + 0.5 * i, *MID_DEG[1:]], 0.5) for i in range(60)]
    write_recording(app_data / "recordings", "so101-20261016-101500.jsonl", smooth, rate_hz=1000)
    _, arm = play(main, FakeArm(MID_DEG))
    played = arm.sent("play")
    assert played[-1] == pytest.approx(smooth[-1][0])
    assert len(played) >= 29.5 / (PLAY_DEG_PER_S / RATE_HZ)
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_a_recording_under_45_degrees_a_second_plays_at_its_own_pace(main, app_data):
    """30 degrees a second (1 degree a frame at 30 frames a second): two ticks a frame, on every frame."""
    steady = [([MID_DEG[0] + i, *MID_DEG[1:]], 0.5) for i in range(30)]
    write_recording(app_data / "recordings", "so101-20261016-101500.jsonl", steady)
    _, arm = play(main, FakeArm(MID_DEG))
    played = arm.sent("play")
    assert len(played) == 2 * 29
    for (joints, _), sent in zip(steady[1:], played[1::2]):
        assert sent == pytest.approx(joints)


def test_a_recording_of_one_frame(main, app_data):
    write_recording(app_data / "recordings", "so101-20261016-101500.jsonl", [([5.0, -30.0, 30.0, 0.0, 0.0], 0.5)])
    app, arm = play(main, FakeArm(MID_DEG))
    assert arm.sent()[-1] == pytest.approx([5.0, -30.0, 30.0, 0.0, 0.0])
    assert app.finish_calls
    assert_motion_ok(arm, RATE_HZ, RECORD_PHASES, gripper_steps=GRIPPER_STEPS)


def test_the_so101_demo_stays_in_its_envelope(main):
    demo = main.demo_frames("so101", so101_envelope(main))
    assert (len(demo) - 1) / main.FRAME_HZ == pytest.approx(8.0)
    for joints, gripper in demo:
        assert not any(max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(joints, envelope_deg())), joints
        assert 0.0 <= gripper <= 1.0
    steps = [max(abs(b - a) for a, b in zip(x, y)) for (x, _), (y, _) in itertools.pairwise(demo)]
    assert max(steps) * main.FRAME_HZ <= 45.0 + 1e-9


def test_the_app_envelope_sits_inside_the_urdf_one(main):
    for name, (lo, hi), (ulo, uhi) in zip(SO101_NAMES, so101_envelope(main), envelope_deg()):
        assert ulo - 1e-9 <= lo <= ulo + 1.0, name
        assert uhi - 1.0 <= hi <= uhi + 1e-9, name


def test_the_motion_check_catches_a_target_past_the_urdf_limits():
    """The SDK's safety check has no SO-101 limits, so ``assert_motion_ok`` checks the URDF ones
    itself: a small step inward that is still past the shoulder's limit fails."""
    lowest = urdf_limits_deg()[1][0]
    arm = FakeArm([0.0, lowest - 6.0, 30.0, 0.0, 0.0])
    arm.send_teleop_action([math.radians(v) for v in (0.0, lowest - 5.5, 30.0, 0.0, 0.0)], 0.0)
    with pytest.raises(AssertionError, match="URDF"):
        assert_motion_ok(arm, RATE_HZ, RECORD_PHASES)


@pytest.mark.parametrize("start", sorted(STARTS))
def test_the_demo_keeps_the_gripper_10_cm_above_the_base(main, app_data, start):
    """Forward kinematics on the vendored URDF (ikpy): the gripper frame never comes within 10 cm of the base plane."""
    pytest.importorskip("ikpy")
    from monorobot.hal.adapters.hf_so101.follower import _URDF_PATH, _URDF_TARGET_FRAME, ARM_MOTOR_NAMES
    from monorobot.hal.kinematics import URDFChainKinematics

    kin = URDFChainKinematics(_URDF_PATH, ARM_MOTOR_NAMES, target_frame=_URDF_TARGET_FRAME)
    _, arm = play(main, FakeArm(STARTS[start]))
    heights = [kin.fk([math.radians(v) for v in pose])[2, 3] for pose in [arm.start_deg, *arm.sent()]]
    assert min(heights) >= 0.10, min(heights)


# --------------------------------------------------------------------------- the AL5D


def test_the_al5d_demo_stays_15_percent_inside_its_limits(main):
    envelope = al5d_envelope_deg()
    assert main.al5d_envelope(main.joint_limits_deg(FakeAL5D(), "lynxmotion_al5d")) == pytest.approx(envelope)
    demo = main.demo_frames("al5d", envelope)
    assert (len(demo) - 1) / main.FRAME_HZ == pytest.approx(8.0)
    for joints, _ in demo:
        assert not any(max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(joints, envelope)), joints


def test_the_al5d_demo_keeps_the_gripper_10_cm_above_the_table(main):
    """The AL5D's own forward kinematics (z = 0 at the table): the demo's lowest gripper point is 10 cm up."""
    from monorobot.hal.adapters.lynxmotion_al5d.adapter import AL5DAdapter

    demo = main.demo_frames("al5d", al5d_envelope_deg())
    heights = [AL5DAdapter._forward_kinematics(None, [math.radians(v) for v in joints])[2] for joints, _ in demo]
    assert min(heights) >= 0.10, min(heights)


@pytest.mark.parametrize("start", ["home", "far"])
def test_the_al5d_plays_the_demo_with_timed_moves(main, app_data, caplog, start):
    caplog.set_level(logging.INFO, logger="replay")
    start_deg = AL5D_HOME_DEG if start == "home" else (-80.0, 10.0, -170.0, 80.0)
    arm = FakeAL5D(start_deg)
    app, _ = play(main, arm, robot_type="lynxmotion_al5d")
    assert [r.getMessage() for r in caplog.records if r.name == "replay"] == ["Playing the demo.", "Done."]
    assert app.finish_calls
    assert_al5d_motion_ok(arm, RATE_HZ, al5d_envelope_deg())
    moves = arm.moves()
    assert moves[0][1] == (1500 if start == "home" else math.ceil(80.0 / 45.0 * 1000))
    demo = main.demo_frames("al5d", al5d_envelope_deg())
    assert moves[0][0] == pytest.approx(demo[0][0]) and moves[-1][0] == pytest.approx(demo[-1][0])
    assert arm.grips() == ["gripper_open", "gripper_close", "gripper_open"]
    assert (moves[-1][2] - moves[1][2]) / RATE_HZ == pytest.approx(8.0, abs=0.3)


def test_the_al5d_reads_its_limits_at_setup(main, app_data):
    """Limits narrower than the adapter's: the demo is held 15 percent inside them."""
    narrow = {"joint_limits_lo": [math.radians(-30), math.radians(60), math.radians(-120), math.radians(-30)],
              "joint_limits_hi": [math.radians(30), math.radians(120), math.radians(-60), math.radians(30)]}
    arm = FakeAL5D(capabilities=narrow)
    play(main, arm, robot_type="lynxmotion_al5d")
    envelope = [(-21.0, 21.0), (69.0, 111.0), (-111.0, -69.0), (-21.0, 21.0)]
    assert_al5d_motion_ok(arm, RATE_HZ, envelope)
    assert max(abs(joints[0]) for joints, *_ in arm.moves()) == pytest.approx(21.0)


def test_the_al5d_plays_its_newest_recording(main, app_data, caplog):
    caplog.set_level(logging.INFO, logger="replay")
    frames = [([float(i), 90.0 + i / 2, -90.0, 0.0], 1.0 if i < 20 else 0.0) for i in range(40)]
    write_recording(app_data / "recordings", "al5d-20261016-101500.jsonl", frames, joints=AL5D_NAMES)
    write_recording(app_data / "recordings", "so101-20261016-111500.jsonl", short_recording(10))
    arm = FakeAL5D()
    play(main, arm, robot_type="lynxmotion_al5d")
    assert caplog.records[0].getMessage() == "Playing al5d-20261016-101500.jsonl."
    assert arm.moves()[-1][0] == pytest.approx(frames[-1][0])
    assert arm.grips() == ["gripper_open", "gripper_close"]
    assert_al5d_motion_ok(arm, RATE_HZ, al5d_envelope_deg())


# --------------------------------------------------------------------------- the simulated pair


def folded(state: dict) -> bool:
    """The simulated follower's state file says it folded to rest and let go."""
    return state["torque"] == "off" and state["joints_deg"][1:4] == pytest.approx(list(REST_DEG[1:4]), abs=0.5)


def test_on_the_simulated_follower_the_demo_plays(main, app_data, sim_follower):
    assert sim_joints_deg(sim_follower) == pytest.approx(list(REST_DEG), abs=0.01)
    app = make_app(main, sim_follower, "sim_so101_follower")
    run_paced(app, LONG)
    assert app.finish_calls and app.phase == "done"
    last = main.demo_frames("so101", so101_envelope(main))[-1][0]
    assert wait_until_still(sim_follower) == pytest.approx(last, abs=2.0)


def test_on_the_simulated_pair_it_records_then_plays_back(main, tmp_path, app_data, monkeypatch, caplog):
    """The SDK's own run(): the leader from robot.yaml's also_uses, a recording, the playback, the fold to rest.

    Then a run with the leader unplugged plays that recording.
    """
    caplog.set_level(logging.INFO, logger="replay")
    monkeypatch.setattr(main, "RECORD_S", 3.0)
    app, code, state = run_under_sdk(main, tmp_path, leader="on")
    assert code == 0 and app.error is None
    assert type(app.peers["leader"]).__name__ == "SimSO101LeaderAdapter"
    assert not app.peers["leader"].is_connected()                # the framework closed it
    assert folded(state)
    path, = (app_data / "recordings").iterdir()
    assert RECORDING.fullmatch(path.name)
    lines = read_lines(path)
    assert lines[0]["robot_type"] == "sim_so101_follower" and len(lines) == 1 + 90
    moved = max(abs(a - b) for a, b in zip(lines[1]["joints_deg"], lines[-1]["joints_deg"]))
    assert moved > 1.0                                             # the scripted hand moved the leader
    said = [r.getMessage() for r in caplog.records if r.name == "replay"]
    assert said == ["Reading the leader.", "Recording 3 s: move the leader.", "Recorded 90 frames, playing it back.",
                    "Done."]

    caplog.clear()
    app, code, state = run_under_sdk(main, tmp_path, leader="unplugged")
    assert code == 0 and app.error is None and app.peers == {}
    assert folded(state)
    said = [r.getMessage() for r in caplog.records if r.name == "replay"]
    assert said == [f"Playing {path.name}.", "Done."]


@pytest.mark.slow
def test_monorobot_start_on_the_simulated_pair(tmp_path):
    """``monorobot start`` with the simulated pair: it records (4 s here), plays it back, and ends by
    itself with exit code 0, the follower folded to rest. With no app data folder given, the
    recording goes under the sandbox's monorobot home."""
    copy, binding, sim = sim_pair_copy(tmp_path, record_s=4.0)
    proc = start_on_sim_pair(copy, binding, tmp_path)
    out = LineReader(proc)
    try:
        code = proc.wait(timeout=120)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        raise
    text = out.text()
    assert code == 0, text
    assert "Using the sim_so101_follower settings in" in text, text   # the binding reached the follower
    for line in ("Recording 4 s: move the leader.", "Recorded 120 frames, playing it back.", "Done."):
        assert line in text, text
    assert "single goal" not in text and "was cut to" not in text
    assert folded(json.loads((sim / "sim-follower-1.json").read_text()))
    recordings = list((tmp_path / "proc-home" / ".monorobot" / "data" / "replay" / "recordings").iterdir())
    assert len(recordings) == 1 and RECORDING.fullmatch(recordings[0].name)


def test_start_never_runs_on_a_real_robot_here(tmp_path):
    """The copy ``monorobot start`` runs names only simulated robots, and no serial port."""
    copy, binding, _ = sim_pair_copy(tmp_path, record_s=4.0)
    for text in ((copy / "robot.yaml").read_text(), binding.read_text()):
        assert "hf_so101" not in text and "lynxmotion" not in text and "/dev/" not in text


# --------------------------------------------------------------------------- the package


def test_built_manifest(tmp_path):
    manifest, names = built_manifest(tmp_path)
    assert manifest["schema"] == "monorobot/1.2"
    assert manifest["name"] == "replay"
    assert manifest["version"] == VERSION
    assert manifest["author"] == {"handle": "grand-wizard-create"}
    assert manifest["compat"]["robots"] == ["hf_so101_follower", "lynxmotion_al5d"]
    assert manifest["also_uses"] == [{"name": "leader", "type": "hf_so101_leader", "required": False}]
    assert manifest["devices"] == {"required": [], "optional": []}
    assert manifest["permissions"]["network"] is False
    assert manifest["permissions"]["filesystem_write"] == ["app_data"]
    assert manifest["license"] == "Apache-2.0"
    assert manifest["whats_new"] == WHATS_NEW
    assert manifest["description"] == "Record a motion with the leader arm, then play it back on the follower."
    assert sorted(names) == [".gitignore", "README.md", "main.py", "manifest.json", "pyproject.toml", "robot.yaml"]


def test_the_readme_puts_the_leader_port_in_robot_yaml():
    """A robot binding reaches only the main robot for now, so the terminal steps never bind the
    leader: its port goes in robot.yaml, next to the leader's own kwargs."""
    import yaml

    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    assert "robot bind hf_so101_leader" not in text
    leader, = yaml.safe_load((APP_DIR / "robot.yaml").read_text())["also_uses"]
    snippet = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    assert snippet is not None
    assert yaml.safe_load(snippet.group(1))["kwargs"] == {**leader["kwargs"], "port": "<leader port>"}


def test_no_uv_lock_and_no_em_dashes():
    """T2: no uv.lock to be packed by mistake; the copy rules: no em dash in any file of the app."""

    def kept(path) -> bool:
        """A file of the app: not in a hidden folder (git's, a tool's cache), a build or a bytecode folder."""
        folders = path.relative_to(APP_DIR).parts[:-1]
        return not any(part.startswith(".") or part in {"__pycache__", "dist"} for part in folders)

    assert not (APP_DIR / "uv.lock").exists()
    files = [p for p in APP_DIR.rglob("*") if p.is_file() and kept(p)]
    assert APP_DIR / "main.py" in files
    for path in files:
        if path.name != "LICENSE":
            assert "\u2014" not in path.read_text(encoding="utf-8"), path


def test_timing_needs_no_sleeping(main, app_data):
    """A whole record and play on fakes runs in well under a second: the app's clock is its ticks."""
    began = time.monotonic()
    record_and_play(main)
    assert time.monotonic() - began < 5.0
