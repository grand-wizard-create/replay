"""Replay: record a motion with the leader arm, then play it back on the follower.

Runs on the SO-101 follower arm, with the SO-101 leader when it is there (``also_uses``, optional
in ``robot.yaml``), and on the Lynxmotion AL5D arm.

With the leader (SO-101 only), Replay records. The follower glides onto the leader, at most 30
degrees a second on the joint with the farthest to go and for at least 1 s, then copies it for
15 s, each joint target at most 3 degrees from the one before, as Teleop does. Every second tick
(30 frames a second) goes into a new recording in the app's data folder. Then it plays that
recording back.

Without the leader, it plays the newest recording for this arm, or a built-in 8 s demo when there
is none. On the SO-101 it glides to the first frame at most 45 degrees a second, then streams the
frames at their own pace, slowed down along the same path wherever a joint would move faster than
45 degrees a second: no hand is on the leader while it plays. The AL5D has no streaming call: its
first move lasts as long as the gap needs at 45 degrees a second (at least 1.5 s), then it gets
one target every 100 ms, each at most 3 degrees from the one before.

Targets stay inside the arm's envelope: on the SO-101, 10 degrees inside its limits with the base
within 60 degrees either way; on the AL5D, 15 percent inside the limits the arm reports. The work
runs as phases in ``loop()``, so a Stop lands on any tick. When the playback ends Replay finishes,
and the SO-101 folds back to rest on its own.

A recording is JSON lines: a header ``{"schema": "replay/1", "robot_type", "rate_hz", "joints"}``,
then one ``{"t", "joints_deg", "gripper"}`` per frame. The 20 newest per arm are kept.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from monorobot.sdk.app import App

APP_NAME = "replay"
log = logging.getLogger(APP_NAME)

SO101 = frozenset({"hf_so101_follower", "sim_so101_follower"})
AL5D = frozenset({"lynxmotion_al5d"})
JOINTS: dict[str, tuple[str, ...]] = {
    "so101": ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
    "al5d": ("base", "shoulder", "elbow", "wrist"),
}

RECORD_S = 15.0
FRAME_HZ = 30.0              # frames a second in a recording, and in the demos
FLUSH_EVERY_S = 1.0
KEEP = 20                    # recordings kept per arm
SCHEMA = "replay/1"
MAX_FILE_BYTES = 20_000_000  # bigger than this, a file isn't a recording Replay made
REPORT_EVERY_S = 5.0

LEADER_DEG_PER_S = 30.0      # onto the leader, on the joint with the farthest to go
GLIDE_DEG_PER_S = 45.0       # to a recording's first frame
GLIDE_MIN_S = 1.0            # a glide lasts at least this long
STEP_DEG = 3.0               # following the leader: no joint target moves more than this in one tick
PLAY_DEG_PER_S = 45.0        # playing back on the SO-101: no joint target moves faster than this (D37)
GRIPPER_GLIDE_PER_S = 1.0    # gripper opening per second while gliding onto the leader
GRIPPER_STEP = 0.1           # following and playing back: the most the gripper target moves in one tick

AL5D_EVERY_S = 0.1           # the AL5D gets one target every 100 ms, each lasting as long
AL5D_STEP_DEG = 3.0          # each at most this far from the one before on any joint (30 deg/s)
AL5D_FIRST_MIN_MS = 1500     # its first move lasts at least this long
AL5D_MARGIN = 0.15           # its envelope: 15 percent of each joint's range inside its limits

# Every SO-101 target stays 10 degrees inside the follower's URDF limits, and the base turns at
# most 60 degrees either way.
SO101_ENVELOPE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-86.8, 86.8),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-147.0, 152.0),
}

# The demos, about 8 s each: key poses in degrees with the gripper opening (0 closed, 1 open) and
# the seconds to get there from the pose before, joined by straight glides of at most
# DEMO_DEG_PER_S on any joint. The AL5D's are slower: one target every 100 ms, 3 degrees at most.
DEMO_DEG_PER_S: dict[str, float] = {"so101": 45.0, "al5d": 30.0}
DEMOS: dict[str, tuple[tuple[tuple[float, ...], float, float], ...]] = {
    "so101": (
        ((0.0, -40.0, 45.0, -15.0, 0.0), 0.3, 0.0),
        ((30.0, -30.0, 30.0, 0.0, 30.0), 0.8, 2.0),
        ((30.0, -30.0, 30.0, 0.0, 30.0), 0.8, 0.5),
        ((-30.0, -30.0, 30.0, 0.0, -30.0), 0.2, 3.0),
        ((-30.0, -30.0, 30.0, 0.0, -30.0), 0.2, 0.5),
        ((0.0, -40.0, 45.0, -15.0, 0.0), 0.3, 2.0),
    ),
    "al5d": (
        ((0.0, 90.0, -90.0, 0.0), 1.0, 0.0),
        ((30.0, 100.0, -100.0, -15.0), 1.0, 2.0),
        ((30.0, 100.0, -100.0, -15.0), 0.0, 0.5),
        ((-30.0, 100.0, -100.0, -15.0), 0.0, 3.0),
        ((-30.0, 100.0, -100.0, -15.0), 1.0, 0.5),
        ((0.0, 90.0, -90.0, 0.0), 1.0, 2.0),
    ),
}

# A recording's file name: <kind>-<YYYYmmdd-HHMMSS>.jsonl, with -2, -3, ... for more in the same second.
RECORDING_NAME = re.compile(r"(so101|al5d)-(\d{8}-\d{6})(?:-(\d{1,4}))?\.jsonl")

Frame = tuple[list[float], float]           # (joints in degrees, gripper opening)
Envelope = list[tuple[float, float]]        # (lowest, highest) per joint, in degrees


# --------------------------------------------------------------------------- small helpers


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def clip(joints_deg: Sequence[float], envelope: Envelope) -> list[float]:
    """The pose held inside ``envelope``, joint by joint."""
    return [min(max(float(v), lo), hi) for v, (lo, hi) in zip(joints_deg, envelope)]


def step_toward(last: float, aim: float, most: float) -> float:
    """``aim``, or ``last`` moved toward it by ``most`` when it is further than that."""
    return aim if abs(aim - last) <= most else last + math.copysign(most, aim - last)


def frame_at(frames: Sequence[Frame], position: float) -> Frame:
    """The frame at a fractional ``position``, on a straight line between the two around it."""
    end = len(frames) - 1
    i = min(end, max(0, math.floor(position)))
    if i >= end:
        joints, gripper = frames[end]
        return list(joints), gripper
    f = position - i
    (a, ga), (b, gb) = frames[i], frames[i + 1]
    return [x + (y - x) * f for x, y in zip(a, b)], ga + (gb - ga) * f


def data_dir() -> Path:
    """Replay's own data folder: ``MONOROBOT_APP_DATA`` when set (Robot App Store sets it), else
    ``<MONOROBOT_HOME or ~/.monorobot>/data/replay``. Created on the first write."""
    given = os.environ.get("MONOROBOT_APP_DATA")
    if given:
        return Path(given).expanduser()
    home = os.environ.get("MONOROBOT_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".monorobot") / "data" / APP_NAME


def recordings_dir() -> Path:
    return data_dir() / "recordings"


def stamp() -> str:
    """This second, as recording names spell it."""
    return time.strftime("%Y%m%d-%H%M%S")


def first_line(exc: BaseException) -> str:
    text = str(getattr(exc, "strerror", None) or exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


# --------------------------------------------------------------------------- recordings


class Recording:
    """A recording read back: its file name, its frames a second and its frames."""

    def __init__(self, name: str, rate_hz: float, frames: list[Frame]) -> None:
        self.name = name
        self.rate_hz = float(rate_hz)
        self.frames = frames


def recordings(kind: str, folder: Path) -> list[Path]:
    """The recordings for arm ``kind`` in ``folder``, oldest first (by the time in their name)."""
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    found = []
    for path in entries:
        match = RECORDING_NAME.fullmatch(path.name)
        if match and match.group(1) == kind and path.is_file():
            found.append(((match.group(2), int(match.group(3) or 1)), path))
    return [path for _, path in sorted(found)]


def trim(kind: str, folder: Path, keep: int = KEEP) -> list[Path]:
    """Delete all but the ``keep`` newest recordings for arm ``kind``; return the ones deleted."""
    removed = []
    for path in recordings(kind, folder)[:-keep]:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Couldn't delete the old recording %s (%s).", path.name, first_line(exc))
            continue
        removed.append(path)
    return removed


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _frame(line: str, joints: int) -> Frame | None:
    """One frame line, checked: ``joints`` finite angles and a finite gripper opening."""
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    angles, gripper = data.get("joints_deg"), _number(data.get("gripper"))
    if not isinstance(angles, list) or len(angles) != joints or gripper is None:
        return None
    values = [_number(v) for v in angles]
    if any(v is None for v in values):
        return None
    return [float(v) for v in values if v is not None], gripper


def load_recording(path: Path, kind: str, envelope: Envelope) -> Recording | None:
    """The recording in ``path``, its frames held inside ``envelope``; None when it isn't one for this arm.

    A recording cut short (a Stop, a crash) keeps the frames before its first broken line.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines:
        return None
    try:
        header = json.loads(lines[0])
    except ValueError:
        return None
    if not isinstance(header, dict) or header.get("schema") != SCHEMA or header.get("joints") != list(JOINTS[kind]):
        return None
    rate = _number(header.get("rate_hz"))
    if rate is None or not 1.0 <= rate <= 1000.0:
        return None
    frames: list[Frame] = []
    for line in lines[1:]:
        frame = _frame(line, len(JOINTS[kind]))
        if frame is None:
            break
        frames.append((clip(frame[0], envelope), clamp01(frame[1])))
    if not frames:
        return None
    return Recording(path.name, rate, frames)


def newest_recording(kind: str, envelope: Envelope, folder: Path | None = None) -> Recording | None:
    """The newest recording for arm ``kind`` that Replay can play, or None."""
    for path in reversed(recordings(kind, folder or recordings_dir())):
        recording = load_recording(path, kind, envelope)
        if recording is not None:
            return recording
        log.warning("Skipped %s: it isn't a recording Replay can play.", path.name)
    return None


class Recorder:
    """A new recording: the header at once, then one line per frame, flushed every second."""

    def __init__(self, folder: Path, kind: str, robot_type: str, rate_hz: float) -> None:
        self.rate_hz = float(rate_hz)
        self.frames: list[Frame] = []
        self._flush_every = max(1, round(self.rate_hz * FLUSH_EVERY_S))
        self._file: IO[str] | None = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.path, self._file = self._create(folder, kind)
            rate: float | int = int(self.rate_hz) if self.rate_hz.is_integer() else self.rate_hz
            self._write({"schema": SCHEMA, "robot_type": robot_type, "rate_hz": rate, "joints": list(JOINTS[kind])})
            self._file.flush()
        except OSError as exc:
            self.close()
            raise RuntimeError(f"Replay can't save a recording in {folder} ({first_line(exc)}).") from exc

    @staticmethod
    def _create(folder: Path, kind: str) -> tuple[Path, IO[str]]:
        base = f"{kind}-{stamp()}"
        for n in range(1, 1000):
            path = folder / (f"{base}.jsonl" if n == 1 else f"{base}-{n}.jsonl")
            try:
                return path, path.open("x", encoding="utf-8")
            except FileExistsError:
                continue
        raise FileExistsError(f"too many recordings named {base}")

    def _write(self, data: dict[str, Any]) -> None:
        assert self._file is not None
        self._file.write(json.dumps(data, separators=(",", ":")) + "\n")

    def add(self, joints_deg: Sequence[float], gripper: float) -> None:
        frame = ([round(float(v), 3) for v in joints_deg], round(clamp01(gripper), 3))
        try:
            self._write({"t": round(len(self.frames) / self.rate_hz, 3), "joints_deg": frame[0], "gripper": frame[1]})
            self.frames.append(frame)
            if len(self.frames) % self._flush_every == 0:
                assert self._file is not None
                self._file.flush()
        except OSError as exc:
            raise RuntimeError(f"Replay can't save the recording ({first_line(exc)}).") from exc

    def close(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError as exc:
                log.warning("Couldn't save the end of the recording (%s).", first_line(exc))


# --------------------------------------------------------------------------- motion


def demo_frames(kind: str, envelope: Envelope) -> list[Frame]:
    """The built-in demo for arm ``kind``, as frames at ``FRAME_HZ``, held inside ``envelope``."""
    keys = DEMOS[kind]
    speed = DEMO_DEG_PER_S[kind]
    pose, gripper = clip(keys[0][0], envelope), keys[0][1]
    frames: list[Frame] = [(pose, gripper)]
    for goal_deg, goal_gripper, seconds in keys[1:]:
        goal = clip(goal_deg, envelope)
        gap = max(abs(b - a) for a, b in zip(pose, goal))
        count = max(1, math.ceil(max(seconds, gap / speed) * FRAME_HZ - 1e-9))
        for i in range(1, count + 1):
            f = i / count
            frames.append(([a + (b - a) * f for a, b in zip(pose, goal)], gripper + (goal_gripper - gripper) * f))
        pose, gripper = goal, goal_gripper
    return frames


def joint_limits_deg(robot: Any, robot_type: str) -> Envelope:
    """The joint limits the arm reports (its ``CAPABILITIES``), in degrees.

    A robot in its own process is a proxy, which carries no ``CAPABILITIES``: its adapter class does.
    """
    caps = getattr(robot, "CAPABILITIES", None)
    if not isinstance(caps, Mapping):
        from monorobot.hal.registry import get_adapter_class

        caps = get_adapter_class(robot_type).CAPABILITIES
    lo, hi = caps.get("joint_limits_lo") or [], caps.get("joint_limits_hi") or []
    return [(math.degrees(float(a)), math.degrees(float(b))) for a, b in zip(lo, hi)]


def al5d_envelope(limits_deg: Envelope) -> Envelope:
    """``AL5D_MARGIN`` of each joint's range inside its limits."""
    if len(limits_deg) < len(JOINTS["al5d"]):
        raise RuntimeError("Replay can't read this arm's joint limits.")
    return [(lo + AL5D_MARGIN * (hi - lo), hi - AL5D_MARGIN * (hi - lo))
            for lo, hi in limits_deg[: len(JOINTS["al5d"])]]


def leader_pose(leader: Any) -> Frame | None:
    """One read of the leader: its arm joints in degrees and its gripper opening (0 closed, 1 open).

    Telemetry gives both from one read of the arm; it counts the gripper the other way round (0
    open). A leader in its own process sends telemetry as a dict, one in the app's process as a
    ``TelemetryFrame``. None when the reading isn't usable (not numbers, or too few joints).
    """
    joints_count = len(JOINTS["so101"])
    frame = leader.get_telemetry()
    joints = frame["joints"] if isinstance(frame, Mapping) else frame.joints
    closed = frame["gripper"] if isinstance(frame, Mapping) else frame.gripper
    try:
        values = [math.degrees(float(v)) for v in joints[:joints_count]]
        opening = 1.0 - float(closed)
    except (TypeError, ValueError):
        return None
    if len(values) < joints_count or not all(math.isfinite(v) for v in [*values, opening]):
        return None
    return values, clamp01(opening)


class Glide:
    """One target per tick from ``start_deg`` to ``goal_deg``, on a straight line in joint space.

    Lasts ``max(min_s, largest joint gap / max_deg_per_s)``, so no joint moves faster than
    ``max_deg_per_s``. The gripper, given as ``(from, to)``, ramps over the same ticks.
    """

    def __init__(
        self,
        start_deg: Sequence[float],
        goal_deg: Sequence[float],
        max_deg_per_s: float,
        rate_hz: float,
        min_s: float = GLIDE_MIN_S,
        *,
        gripper: tuple[float, float] | None = None,
    ) -> None:
        self.start = [float(v) for v in start_deg]
        self.goal = [float(v) for v in goal_deg]
        gap = max(abs(g - s) for s, g in zip(self.start, self.goal))
        self.duration_s = max(min_s, gap / max_deg_per_s)
        self.ticks = max(1, math.ceil(self.duration_s * rate_hz - 1e-9))
        self.gripper = gripper
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.ticks

    def next(self) -> tuple[list[float], float | None]:
        """The next target: ``(joints_deg, gripper)``, the gripper None when the glide has none."""
        self.i = min(self.i + 1, self.ticks)
        f = self.i / self.ticks
        joints = [s + (g - s) * f for s, g in zip(self.start, self.goal)]
        if self.gripper is None:
            return joints, None
        a, b = self.gripper
        return joints, a + (b - a) * f


class Mirror:
    """The follower's targets while it copies the leader, one per tick, as Teleop makes them.

    ``phase`` is ``glide`` until a glide tick, at least ``GLIDE_MIN_S`` in, puts the follower on
    the leader's pose itself; then ``follow``, so the first follow tick has no gap left to close.
    Each glide tick moves on a straight line toward where the leader is now: the joint with the
    farthest to go moves at most ``LEADER_DEG_PER_S`` a second, and no more than its gap shared
    over the ticks left of ``GLIDE_MIN_S``. So a big gap takes the gap over
    ``LEADER_DEG_PER_S``, a small one ``GLIDE_MIN_S``, and a leader that moves away is still
    caught. Following, each joint target is at most ``STEP_DEG`` from the one before. Targets
    aim at the leader's pose held inside the envelope.
    """

    def __init__(self, start_deg: Sequence[float], start_gripper: float, rate_hz: float, envelope: Envelope) -> None:
        self.rate_hz = float(rate_hz)
        self.envelope = envelope
        self.joints = [float(v) for v in start_deg]
        self.gripper = clamp01(start_gripper)
        self.phase = "glide"
        self.glided = 0                         # glide ticks so far
        self.min_ticks = max(1, math.ceil(GLIDE_MIN_S * self.rate_hz - 1e-9))
        self._on_leader = False                 # the last glide tick ended on the leader

    def step(self, leader_deg: Sequence[float], leader_gripper: float) -> Frame:
        """The next target, ``(joints_deg, gripper)``, toward the leader as read this tick."""
        aim = clip(leader_deg, self.envelope)
        aim_gripper = clamp01(leader_gripper)
        if self.phase == "glide" and self._on_leader:
            self.phase = "follow"
        if self.phase == "glide":
            self._glide(aim)
            self.gripper = step_toward(self.gripper, aim_gripper, GRIPPER_GLIDE_PER_S / self.rate_hz)
        else:
            self.joints = [step_toward(j, a, STEP_DEG) for a, j in zip(aim, self.joints)]
            self.gripper = step_toward(self.gripper, aim_gripper, GRIPPER_STEP)
        return list(self.joints), self.gripper

    def _glide(self, aim: list[float]) -> None:
        """One glide tick toward ``aim``."""
        self.glided += 1
        ticks_left = max(1, self.min_ticks - self.glided + 1)
        gap = max(abs(a - j) for a, j in zip(aim, self.joints))
        most = min(LEADER_DEG_PER_S / self.rate_hz, gap / ticks_left)
        if gap <= most:
            self.joints = list(aim)
        else:
            share = most / gap
            self.joints = [j + (a - j) * share for a, j in zip(aim, self.joints)]
        self._on_leader = self.glided >= self.min_ticks and self.joints == aim


class StreamPlayer:
    """SO-101 playback: one ``send_teleop_action`` per tick.

    ``to_start``: a glide from where the arm is to the first frame, at most ``GLIDE_DEG_PER_S``
    and at least ``GLIDE_MIN_S``, the gripper ramped with it. ``play``: along the frames, on a
    straight line between two frames, at their own rate (at 60 Hz, each 30 Hz frame takes two
    ticks) wherever no joint then moves faster than ``PLAY_DEG_PER_S``. Where one would (a fast
    part of a recording, a jump in it, a header with too high a rate), the playback slows down
    to ``PLAY_DEG_PER_S`` along the same path, so it can last longer than the recording. Each
    joint target is at most ``PLAY_DEG_PER_S`` a second from the one before however the frames
    go, and the gripper at most ``GRIPPER_STEP`` a tick. Then ``done``.
    """

    def __init__(self, frames: list[Frame], frame_hz: float, start_deg: Sequence[float], start_gripper: float,
                 rate_hz: float) -> None:
        self.frames = frames
        self.per_tick = frame_hz / rate_hz        # frames a tick, at the recording's own rate
        self.most = PLAY_DEG_PER_S / rate_hz      # degrees a tick, on any joint
        first, first_gripper = frames[0]
        self._glide = Glide(start_deg, first, GLIDE_DEG_PER_S, rate_hz, gripper=(clamp01(start_gripper), first_gripper))
        self.joints = [float(v) for v in start_deg]
        self.gripper = clamp01(start_gripper)
        self.phase = "to_start"
        self.position = 0.0                       # how far along the frames the playback is

    def step(self, robot: Any) -> bool:
        """Send this tick's target; False, having sent nothing, once the playback is over."""
        if self.phase == "to_start":
            if not self._glide.done:
                self.joints, gripper = self._glide.next()
                self.gripper = clamp01(gripper if gripper is not None else self.gripper)
                return self._send(robot)
            self.phase = "play"
        if self.phase == "play":
            end = len(self.frames) - 1
            position = self._reach(min(end, self.position + self.per_tick))
            aim, aim_gripper = frame_at(self.frames, position)
            if position >= end and self.joints == aim and self.gripper == aim_gripper:
                self.phase = "done"
                return False
            self.position = position
            self.joints = [step_toward(j, a, self.most) for j, a in zip(self.joints, aim)]
            self.gripper = step_toward(self.gripper, aim_gripper, GRIPPER_STEP)
            return self._send(robot)
        return False

    def _reach(self, goal: float) -> float:
        """How far along the frames, up to ``goal``, this tick gets: at most ``most`` degrees of path.

        The path is measured straight piece by straight piece (a piece runs between two frames),
        each by the joint that moves the most on it, so no corner is cut and the pose reached is
        at most ``most`` from the one before on every joint. Past a frame, a piece too long to
        finish this tick waits for the next one, so every frame of a slowed part is played.
        """
        position, budget = self.position, self.most
        here, _ = frame_at(self.frames, position)
        while position < goal:
            piece_end = min(goal, math.floor(position) + 1.0)
            there, _ = frame_at(self.frames, piece_end)
            length = max(abs(b - a) for a, b in zip(here, there))
            if length <= budget:
                position, here, budget = piece_end, there, budget - length
            elif position == self.position:
                return position + (piece_end - position) * budget / length
            else:
                return position
        return position

    def _send(self, robot: Any) -> bool:
        robot.send_teleop_action([math.radians(v) for v in self.joints], self.gripper)
        return True


class StepPlayer:
    """AL5D playback: ``move_joints`` with a duration, the AL5D's own way to move slowly.

    ``to_start``: one move to the first frame, lasting the largest gap at ``GLIDE_DEG_PER_S`` and
    at least ``AL5D_FIRST_MIN_MS``; nothing more is sent until it ends. ``play``: one target every
    ``AL5D_EVERY_S``, lasting as long, each joint at most ``AL5D_STEP_DEG`` from the one before. The
    gripper, only open or closed on this arm, follows the frames (open from 0.5 up). Then ``done``.
    """

    def __init__(self, frames: list[Frame], frame_hz: float, start_deg: Sequence[float], rate_hz: float) -> None:
        self.frames = frames
        self.frame_hz = float(frame_hz)
        self.rate_hz = float(rate_hz)
        self.every = max(1, round(AL5D_EVERY_S * rate_hz))      # ticks between targets
        self.step_ms = round(self.every / self.rate_hz * 1000)
        self.joints = [float(v) for v in start_deg]
        self.gripper_open: bool | None = None                   # not known until Replay sets it
        self.phase = "to_start"
        self._wait = 0          # ticks until the next command
        self._t = 0.0           # playback time, seconds

    def step(self, robot: Any) -> bool:
        """Send this tick's commands, if any; False once the playback is over."""
        if self._wait > 0:
            self._wait -= 1
            return True
        if self.phase == "to_start":
            first, gripper = self.frames[0]
            gap = max(abs(a - b) for a, b in zip(first, self.joints))
            self._move(robot, first, max(AL5D_FIRST_MIN_MS, math.ceil(gap / GLIDE_DEG_PER_S * 1000 - 1e-9)))
            self._grip(robot, gripper)
            self.phase = "play"
            return True
        if self.phase == "play":
            end = len(self.frames) - 1
            self._t += self.every / self.rate_hz
            position = min(end, self._t * self.frame_hz)
            aim, gripper = frame_at(self.frames, position)
            self._grip(robot, gripper)
            if position >= end and self.joints == aim:
                self.phase = "done"
                return False
            self._move(robot, [step_toward(j, a, AL5D_STEP_DEG) for j, a in zip(self.joints, aim)], self.step_ms)
            return True
        return False

    def _move(self, robot: Any, joints_deg: list[float], duration_ms: int) -> None:
        robot.move_joints([math.radians(v) for v in joints_deg], duration_ms=duration_ms)
        self.joints = list(joints_deg)
        self._wait = max(0, math.ceil(duration_ms / 1000 * self.rate_hz - 1e-9) - 1)

    def _grip(self, robot: Any, opening: float) -> None:
        want_open = opening >= 0.5
        if want_open != self.gripper_open:
            if want_open:
                robot.gripper_open()
            else:
                robot.gripper_close()
            self.gripper_open = want_open


# --------------------------------------------------------------------------- the app


class Replay(App):
    def setup(self) -> None:
        robot_type = self.config.robot.type
        if robot_type in SO101:
            self.kind = "so101"
            self.envelope = [SO101_ENVELOPE_DEG[name] for name in JOINTS["so101"]]
        elif robot_type in AL5D:
            self.kind = "al5d"
            self.envelope = al5d_envelope(joint_limits_deg(self.robot, robot_type))
        else:
            raise RuntimeError("Replay doesn't know this robot.")
        joints = len(JOINTS[self.kind])
        start = [math.degrees(v) for v in self.robot.get_joint_positions()][:joints]
        # The leader is the SO-101's: with any other arm it isn't used.
        self.leader = getattr(self, "peers", {}).get("leader") if self.kind == "so101" else None
        self.mirror: Mirror | None = None
        self.recorder: Recorder | None = None
        self.player: StreamPlayer | StepPlayer | None = None
        self.followed = 0          # ticks spent recording
        self.skipped = 0           # leader readings that weren't usable
        self.ended = False         # the playback is over and finish() called
        self.ticks_per_frame = max(1, round(self.rate_hz / FRAME_HZ))
        if self.leader is not None:
            self.mode = "record"
            self.mirror = Mirror(start, clamp01(self.robot.gripper_get_width()), self.rate_hz, self.envelope)
            log.info("Reading the leader.")
            return
        self.mode = "play"
        recording = newest_recording(self.kind, self.envelope)
        if recording is None:
            frames, frame_hz = demo_frames(self.kind, self.envelope), FRAME_HZ
            log.info("Playing the demo.")
        else:
            frames, frame_hz = recording.frames, recording.rate_hz
            log.info("Playing %s.", recording.name)
        self.player = self._player(frames, frame_hz, start)

    @property
    def phase(self) -> str:
        """``glide`` and ``record`` while recording, then ``to_start``, ``play`` and ``done``."""
        player = getattr(self, "player", None)
        if player is not None:
            return player.phase
        mirror = getattr(self, "mirror", None)
        if mirror is not None:
            return "glide" if mirror.phase == "glide" else "record"
        return "setup"

    def _player(self, frames: list[Frame], frame_hz: float, start: list[float],
                gripper: float | None = None) -> StreamPlayer | StepPlayer:
        if self.kind == "al5d":
            return StepPlayer(frames, frame_hz, start, self.rate_hz)
        if gripper is None:
            gripper = clamp01(self.robot.gripper_get_width())
        return StreamPlayer(frames, frame_hz, start, gripper, self.rate_hz)

    def loop(self) -> None:
        if self.ended:
            return
        if self.player is None:
            self._record()
            return
        if self.player.step(self.robot):
            return
        self.ended = True
        log.info("Done.")
        self.finish()

    def _record(self) -> None:
        assert self.mirror is not None
        pose = leader_pose(self.leader)
        if pose is None:
            # The follower keeps its last target this tick.
            if self.skipped == 0:
                log.warning("The leader sent a reading that isn't usable, so the follower waits for the next one.")
            self.skipped += 1
            return
        before = self.mirror.phase
        joints, gripper = self.mirror.step(*pose)
        if before == "glide" and self.mirror.phase == "follow":
            log.info("Recording %d s: move the leader.", round(RECORD_S))
            self.recorder = Recorder(recordings_dir(), self.kind, self.config.robot.type,
                                     self.rate_hz / self.ticks_per_frame)
        self.robot.send_teleop_action([math.radians(v) for v in joints], gripper)
        if self.recorder is None:
            return
        if self.followed % self.ticks_per_frame == 0:
            self.recorder.add(joints, gripper)
        self.followed += 1
        total = round(RECORD_S * self.rate_hz)
        if self.followed >= total:
            self._play_back(joints, gripper)
        elif self.followed % max(1, round(REPORT_EVERY_S * self.rate_hz)) == 0:
            log.info("Recording, %d s left.", round((total - self.followed) / self.rate_hz))

    def _play_back(self, joints: list[float], gripper: float) -> None:
        """The recording is over: save it, keep the newest, and play it back from where the arm is."""
        recorder, self.recorder = self.recorder, None
        assert recorder is not None
        recorder.close()
        trim(self.kind, recorder.path.parent)
        log.info("Recorded %d frames, playing it back.", len(recorder.frames))
        recording = load_recording(recorder.path, self.kind, self.envelope)
        if recording is None:      # the file is gone or unreadable: play what was recorded
            recording = Recording(recorder.path.name, recorder.rate_hz, recorder.frames)
        self.player = self._player(recording.frames, recording.rate_hz, joints, gripper)

    def on_shutdown(self) -> None:
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            # Stopped while recording: keep what was recorded so far.
            self.recorder = None
            recorder.close()
            if recorder.frames:
                trim(self.kind, recorder.path.parent)
                log.info("Stopped. Kept the %d frames recorded so far.", len(recorder.frames))
                return
            try:
                recorder.path.unlink(missing_ok=True)      # nothing in it but the header
            except OSError as exc:
                log.warning("Couldn't delete the empty recording %s (%s).", recorder.path.name, first_line(exc))
        if getattr(self, "mode", None) is not None and not self.ended:
            log.info("Stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(Replay.from_config("robot.yaml").run())
