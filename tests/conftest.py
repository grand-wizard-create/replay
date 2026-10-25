"""Test doubles and motion checks for Replay.

``main.py`` is loaded by path. ``FakeArm`` is an SO-101 follower double and ``FakeAL5D`` an AL5D
double; both keep a joint state and record every target the app commands, in order, with the
app's phase at the time. ``FakeLeader`` is a leader arm a test moves by hand.
``assert_motion_ok`` checks SO-101 targets against the rules every seed app follows (no target
more than 3 degrees from the one before, glides no faster than their speed, inside the envelope
and the URDF limits, never ``move_joints`` on an SO-101); ``assert_al5d_motion_ok``
checks the AL5D's timed moves. ``sim_follower`` and ``sim_leader`` are the real simulated SO-101
pair, in this process, with their state in the test's temp folder; ``run_under_sdk`` runs the app
on them with the SDK's own ``run()``; ``start_on_sim_pair`` runs ``monorobot start`` on them.

Run with the engine's Python: ``"$MR/.venv/bin/python" -m pytest -q``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

APP_DIR = Path(__file__).resolve().parents[1]
APP_CLASS = "Replay"

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
REST_DEG: tuple[float, ...] = (0.0, -98.0, 94.0, -94.0, 0.0)     # the follower's folded rest pose
MID_DEG: tuple[float, ...] = (10.0, -20.0, 30.0, 10.0, 5.0)
AL5D_HOME_DEG: tuple[float, ...] = (0.0, 90.0, -90.0, 0.0)       # where the AL5D is after it connects

MAX_STEP_DEG = 3.0          # D37: no target more than 3 degrees from the last one
MAX_DEG_PER_S = 45.0        # D37: big moves stream at most 45 degrees a second
ENVELOPE_MARGIN_DEG = 10.0  # inside the URDF limits
PAN_LIMIT_DEG = 60.0
AL5D_MARGIN = 0.15          # the AL5D demo: 15 percent of each range inside the limits
EPS = 1e-9

SIM_SPEEDUP = 4.0           # the simulated arm's clock runs this much faster than real time
SINGLE_GOAL = "single goal"
# The follower keeps every goal within 10 degrees of where the arm is, and says so when it cuts one:
# a goal that far from the arm is a jump the app asked for, so the simulated runs fail on it too.
CUT_GOAL = "was cut to"
TOKEN_VARS = ("MONOROBOT_GITHUB_TOKEN", "MONOROBOT_SIGNING_KEY", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN")
# The sim-only interlock and its two switches: on in every test and in every engine a test starts.
SIM_ONLY_SWITCHES = {"MONOROBOT_SIM_ONLY": "1", "MONOROBOT_USB_WATCH": "0", "MONOROBOT_FAKE_CAMERAS": "0"}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: runs monorobot start in a process of its own (tens of seconds)")


def load_main() -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{APP_DIR.name.replace('-', '_')}_main", APP_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def main() -> ModuleType:
    return load_main()


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HOME and MONOROBOT_HOME point into the test's temp folder, checked before the test starts:
    nothing lands in the real home or in ~/.monorobot (CONTRACTS 0.5).

    No app data folder from outside, no tokens, and the sim-only interlock is on: no serial port,
    no USB watcher, no real camera.
    """
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MONOROBOT_HOME", str(tmp_path / "monorobot-home"))
    for name in ("HOME", "MONOROBOT_HOME"):
        assert Path(os.environ[name]).resolve().is_relative_to(tmp_path.resolve()), f"{name} is outside the temp folder"
    assert Path.home().resolve().is_relative_to(tmp_path.resolve()), "the home folder is outside the temp folder"
    monkeypatch.delenv("MONOROBOT_APP_DATA", raising=False)
    for name in TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in SIM_ONLY_SWITCHES.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The app's data folder, given as ``MONOROBOT_APP_DATA`` as Robot App Store gives it. Not created."""
    folder = tmp_path / "data" / "replay"
    monkeypatch.setenv("MONOROBOT_APP_DATA", str(folder))
    return folder


# --------------------------------------------------------------------------- doubles


class FakeArm:
    """An SO-101 follower double: keeps a joint state and records every commanded target, in order.

    ``targets`` holds ``(method, joints_deg, gripper, label)``; ``label()`` names the app's phase
    at each command. Reads are counted in ``reads``.
    """

    def __init__(self, start_deg: tuple[float, ...] | list[float] = REST_DEG, gripper: float = 0.0) -> None:
        self.start_deg = [float(v) for v in start_deg]
        self.joints_deg = list(self.start_deg)
        self.start_gripper = float(gripper)
        self.gripper = float(gripper)
        self.targets: list[tuple[str, list[float], float | None, str]] = []
        self.reads: dict[str, int] = {"get_joint_positions": 0, "gripper_get_width": 0, "get_telemetry": 0}
        self.label: Callable[[], str] = lambda: ""

    def get_joint_positions(self) -> list[float]:
        self.reads["get_joint_positions"] += 1
        return [math.radians(v) for v in self.joints_deg]

    def gripper_get_width(self) -> float:
        self.reads["gripper_get_width"] += 1
        return self.gripper

    def get_telemetry(self) -> dict[str, Any]:
        self.reads["get_telemetry"] += 1
        return {"joints": self.get_joint_positions(), "gripper": 1.0 - self.gripper}

    def send_teleop_action(self, arm_joints_rad: list[float], gripper_width: float) -> bool:
        assert len(arm_joints_rad) == 5, arm_joints_rad
        assert 0.0 <= gripper_width <= 1.0, gripper_width
        self._record("send_teleop_action", [math.degrees(v) for v in arm_joints_rad], float(gripper_width))
        return True

    def move_joints(self, joints: list[float], speed: float = 0.5, **kwargs: Any) -> bool:
        self._record("move_joints", [math.degrees(v) for v in joints], None)
        return True

    def gripper_open(self, width: float | None = None) -> bool:
        self._record("gripper_open", list(self.joints_deg), 1.0)
        return True

    def gripper_close(self, force: float = 10.0) -> bool:
        self._record("gripper_close", list(self.joints_deg), 0.0)
        return True

    def _record(self, method: str, joints_deg: list[float], gripper: float | None) -> None:
        self.targets.append((method, joints_deg, gripper, self.label()))
        self.joints_deg = list(joints_deg)
        if gripper is not None:
            self.gripper = gripper

    def labels(self) -> list[str]:
        return [label for *_, label in self.targets]

    def sent(self, label: str | None = None) -> list[list[float]]:
        """The joint targets sent, in order (only those sent in phase ``label``, when given)."""
        return [joints for _, joints, _, lab in self.targets if label is None or lab == label]

    def grippers(self, label: str | None = None) -> list[float | None]:
        return [gripper for _, _, gripper, lab in self.targets if label is None or lab == label]


class FakeAL5D:
    """An AL5D double: four joints, timed ``move_joints``, a gripper that is only open or closed.

    ``commands`` holds ``(method, joints_deg, duration_ms, tick, label)``; ``clock()`` gives the
    app's tick and ``label()`` its phase at each command. With ``capabilities`` it carries them
    as the adapter does in the app's process; without, it is like a robot in its own process,
    whose proxy carries none.
    """

    def __init__(self, start_deg: tuple[float, ...] | list[float] = AL5D_HOME_DEG,
                 capabilities: dict[str, Any] | None = None) -> None:
        self.start_deg = [float(v) for v in start_deg]
        self.joints_deg = list(self.start_deg)
        if capabilities is not None:
            self.CAPABILITIES = capabilities
        self.commands: list[tuple[str, list[float] | None, int | None, int, str]] = []
        self.clock: Callable[[], int] = lambda: 0
        self.label: Callable[[], str] = lambda: ""

    def get_joint_positions(self) -> list[float]:
        return [math.radians(v) for v in self.joints_deg]

    def gripper_get_width(self) -> float:
        return 0.5

    def move_joints(self, joints: list[float], speed: float = 0.5, duration_ms: int | None = None) -> bool:
        assert len(joints) == 4, joints
        self.joints_deg = [math.degrees(v) for v in joints]
        self.commands.append(("move_joints", list(self.joints_deg), duration_ms, self.clock(), self.label()))
        return True

    def send_teleop_action(self, arm_joints_rad: list[float], gripper_width: float) -> bool:
        self.commands.append(("send_teleop_action", None, None, self.clock(), self.label()))
        return True

    def gripper_open(self, width: float | None = None) -> bool:
        self.commands.append(("gripper_open", None, None, self.clock(), self.label()))
        return True

    def gripper_close(self, force: float = 10.0) -> bool:
        self.commands.append(("gripper_close", None, None, self.clock(), self.label()))
        return True

    def moves(self) -> list[tuple[list[float], int | None, int, str]]:
        return [(joints, ms, tick, label) for method, joints, ms, tick, label in self.commands
                if method == "move_joints" and joints is not None]

    def grips(self) -> list[str]:
        return [method for method, *_ in self.commands if method.startswith("gripper_")]


class FakeLeader:
    """An SO-101 leader double that a hand moves: set ``joints_deg`` and ``opening`` between ticks.

    Telemetry is its one read, as on the real leader: joints in radians, the gripper counted the
    other way round (0 open, 1 closed).
    """

    def __init__(self, joints_deg: tuple[float, ...] | list[float] = MID_DEG, opening: float = 0.5) -> None:
        self.joints_deg = [float(v) for v in joints_deg]
        self.opening = float(opening)
        self.reads = 0

    def get_telemetry(self) -> Any:
        self.reads += 1
        return {"joints": [math.radians(v) for v in self.joints_deg], "gripper": 1.0 - self.opening,
                "pose": [0.0] * 7, "is_moving": False}

    def get_joint_positions(self) -> list[float]:
        return [math.radians(v) for v in self.joints_deg]

    def gripper_get_width(self) -> float:
        return self.opening

    def is_connected(self) -> bool:
        return True


# --------------------------------------------------------------------------- running the app


def make_app(main: ModuleType, robot: Any, robot_type: str = "hf_so101_follower",
             peers: dict[str, Any] | None = None) -> Any:
    """The app on ``robot``, configured from its own ``robot.yaml`` with ``robot_type`` as the run's type.

    ``peers`` is set by hand, as the SDK sets it from ``also_uses`` at run time.
    """
    from monorobot.sdk.app_config import AppConfig

    config = AppConfig.load(APP_DIR / "robot.yaml")
    config.robot.type = robot_type
    app = getattr(main, APP_CLASS)(config)
    app.robot = robot
    app.peers = dict(peers or {})
    if isinstance(robot, (FakeArm, FakeAL5D)):
        robot.label = lambda: getattr(app, "phase", "")
    if isinstance(robot, FakeAL5D):
        robot.clock = lambda: app.tick
    return app


def _count_finish(app: Any) -> list[int]:
    calls: list[int] = []
    original = app.finish

    def finish() -> None:
        calls.append(app.tick)
        original()

    app.finish = finish
    app.finish_calls = calls
    return calls


def run_ticks(
    app: Any,
    n: int,
    *,
    before_tick: Callable[[Any], None] | None = None,
    until: Callable[[Any], bool] | None = None,
) -> Any:
    """``setup()``, then ``loop()`` up to ``n`` times with ``app.tick`` advanced, as the SDK runs it.

    Stops after the tick that calls ``finish()``, or once ``until(app)`` is true after a tick.
    No sleeping: apps use ``tick / rate_hz`` as their clock.
    """
    calls = _count_finish(app)
    app.setup()
    for _ in range(n):
        if calls:
            break
        if before_tick is not None:
            before_tick(app)
        app.loop()
        app.tick += 1
        if until is not None and until(app):
            break
    return app


def more_ticks(app: Any, n: int, before_tick: Callable[[Any], None] | None = None) -> None:
    """``loop()`` up to ``n`` more times, after ``run_ticks``, with no new ``setup()``."""
    for _ in range(n):
        if app.finish_calls:
            break
        if before_tick is not None:
            before_tick(app)
        app.loop()
        app.tick += 1


def run_paced(app: Any, n: int, *, speedup: float = SIM_SPEEDUP) -> Any:
    """``run_ticks``, paced so the simulated arms' clock (``speedup`` times real time) keeps step with it."""
    calls = _count_finish(app)
    app.setup()
    period = 1.0 / app.rate_hz / speedup
    next_tick = time.monotonic()
    for _ in range(n):
        if calls:
            break
        next_tick += period
        app.loop()
        app.tick += 1
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    return app


# --------------------------------------------------------------------------- the envelopes and the checks


def urdf_limits_deg() -> list[tuple[float, float]]:
    """The follower's joint limits from its URDF, in degrees to one decimal, in ``ARM_JOINTS`` order.

    The URDF writes them in radians to five decimals (100 degrees is 1.74533), hence the rounding.
    """
    from monorobot.hal.adapters.hf_so101.follower import _URDF_PATH

    limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(_URDF_PATH).getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("name") in ARM_JOINTS and limit is not None:
            limits[joint.get("name", "")] = (round(math.degrees(float(limit.get("lower", "0"))), 1),
                                             round(math.degrees(float(limit.get("upper", "0"))), 1))
    return [limits[name] for name in ARM_JOINTS]


def envelope_deg() -> list[tuple[float, float]]:
    """10 degrees inside the URDF limits, and the base within 60 degrees either way."""
    out = [(lo + ENVELOPE_MARGIN_DEG, hi - ENVELOPE_MARGIN_DEG) for lo, hi in urdf_limits_deg()]
    out[0] = (max(out[0][0], -PAN_LIMIT_DEG), min(out[0][1], PAN_LIMIT_DEG))
    return out


def al5d_envelope_deg() -> list[tuple[float, float]]:
    """15 percent of each range inside the AL5D's ``JOINT_LIMITS``, read from the adapter module."""
    from monorobot.hal.adapters.lynxmotion_al5d.adapter import JOINT_LIMITS

    out = []
    for lo, hi in JOINT_LIMITS:
        lo_deg, hi_deg = math.degrees(float(lo)), math.degrees(float(hi))
        margin = AL5D_MARGIN * (hi_deg - lo_deg)
        out.append((lo_deg + margin, hi_deg - margin))
    return out


def outside_by(joints_deg: list[float], envelope: list[tuple[float, float]]) -> list[float]:
    """How far each joint is outside the envelope (0 inside)."""
    return [max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(joints_deg, envelope)]


def assert_motion_ok(
    arm: FakeArm,
    rate_hz: float,
    phases: dict[str, float] | None = None,
    *,
    max_deg_per_s: float = MAX_DEG_PER_S,
    gripper_steps: dict[str, float] | None = None,
) -> None:
    """Every target the app commanded on the SO-101 follows the seed app motion rules.

    - The first target is within the per-tick limit (and 3 degrees) of the pose the arm started in.
    - Consecutive targets differ, per joint, by at most the phase's degrees a second (``phases``,
      by label; ``max_deg_per_s`` otherwise) divided by ``rate_hz``, and never by more than 3 degrees.
    - Every target is inside the envelope, or, in a glide that starts outside it, no joint is
      further out than in the target before it.
    - Every target is inside the follower's URDF limits. This is the real limit check: the SDK's
      safety check for the SO-101 follower runs too, but the SO-101 adapters declare no joint
      limits, so it passes any target today (the AL5D's check, in ``assert_al5d_motion_ok``, is real).
    - ``move_joints`` is never called on an SO-101 (it ignores ``speed`` there and jumps).
    - The gripper stays in [0, 1] and moves per tick at most the phase's step (``gripper_steps``,
      by label; else closed to open in no less than a second).
    """
    from monorobot.hal.adapters.hf_so101.follower import SO101FollowerAdapter
    from monorobot.hal.safety import SafetyMonitor

    assert arm.targets, "the app commanded nothing"
    methods = {method for method, *_ in arm.targets}
    assert "move_joints" not in methods, "move_joints was called on an SO-101"
    monitor = SafetyMonitor.for_adapter(SO101FollowerAdapter)
    urdf = urdf_limits_deg()
    envelope = envelope_deg()
    prev, prev_gripper = arm.start_deg, arm.start_gripper
    for i, (_method, joints, gripper, label) in enumerate(arm.targets):
        limit = min(MAX_STEP_DEG, (phases or {}).get(label, max_deg_per_s) / rate_hz)
        for name, a, b in zip(ARM_JOINTS, prev, joints):
            assert abs(b - a) <= limit + EPS, (
                f"target {i} ({label}): {name} moves {abs(b - a):.3f} degrees in one tick (limit {limit:.3f})"
            )
        out, prev_out = outside_by(joints, envelope), outside_by(prev, envelope)
        if any(out):
            assert all(o <= p + EPS for o, p in zip(out, prev_out)), (
                f"target {i} ({label}): {joints} goes further outside the envelope"
            )
        assert not any(o > EPS for o in outside_by(joints, urdf)), (
            f"target {i} ({label}): {joints} is outside the URDF limits {urdf}"
        )
        monitor.validate_call("move_joints", [[math.radians(v) for v in joints]], {})
        if gripper is not None:
            assert 0.0 <= gripper <= 1.0
            most = (gripper_steps or {}).get(label, 1.0 / rate_hz)
            assert abs(gripper - prev_gripper) <= most + EPS, f"target {i} ({label}): the gripper snaps"
            prev_gripper = gripper
        prev = joints


def assert_al5d_motion_ok(arm: FakeAL5D, rate_hz: float, envelope: list[tuple[float, float]] | None = None) -> None:
    """Every AL5D command follows the AL5D rule (2.2 of the seed app plan).

    - Only ``move_joints`` and the gripper calls; never ``send_teleop_action``.
    - Every ``move_joints`` carries ``duration_ms``. The first lasts at least 1500 ms and at least
      the largest gap from the start pose at 45 degrees a second; each later one lasts 100 ms and
      is at most 3 degrees from the one before on every joint.
    - No move is sent before the one before it has ended: the ticks between two moves cover the
      first one's duration, and the later ones are 100 ms apart.
    - Every target is inside ``envelope`` (the AL5D's limits when not given) and passes the SDK's
      safety check for the AL5D.
    """
    from monorobot.hal.adapters.lynxmotion_al5d.adapter import AL5DAdapter
    from monorobot.hal.safety import SafetyMonitor

    methods = {method for method, *_ in arm.commands}
    assert methods <= {"move_joints", "gripper_open", "gripper_close"}, methods
    moves = arm.moves()
    assert moves, "the app moved nothing"
    monitor = SafetyMonitor.for_adapter(AL5DAdapter)
    # A monitor with no limits passes any target: fail loudly rather than check nothing.
    assert monitor.caps.joint_limits_lo and monitor.caps.joint_limits_hi, "the SDK's AL5D safety check has no limits"
    if envelope is None:
        envelope = [(math.degrees(lo), math.degrees(hi)) for lo, hi in
                    zip(AL5DAdapter.CAPABILITIES["joint_limits_lo"], AL5DAdapter.CAPABILITIES["joint_limits_hi"])]
    first_joints, first_ms, _, _ = moves[0]
    assert isinstance(first_ms, int), moves[0]
    gap = max(abs(a - b) for a, b in zip(arm.start_deg, first_joints))
    assert first_ms >= 1500 and first_ms >= gap / 45.0 * 1000 - EPS, (first_ms, gap)
    every = round(0.1 * rate_hz)
    prev_joints, prev_ms, prev_tick = first_joints, first_ms, moves[0][2]
    for i, (joints, ms, tick, label) in enumerate(moves[1:], start=1):
        assert ms == 100, f"move {i} ({label}) lasts {ms} ms"
        for a, b in zip(prev_joints, joints):
            assert abs(b - a) <= MAX_STEP_DEG + EPS, f"move {i} ({label}): {prev_joints} to {joints}"
        assert (tick - prev_tick) / rate_hz * 1000 >= prev_ms - EPS, f"move {i} ({label}) cuts the one before short"
        if i > 1:
            assert tick - prev_tick == every, f"move {i} ({label}) comes {tick - prev_tick} ticks after the one before"
        prev_joints, prev_ms, prev_tick = joints, ms, tick
    for i, (joints, ms, _, label) in enumerate(moves):
        assert not any(outside_by(joints, envelope)), f"move {i} ({label}): {joints} is outside {envelope}"
        monitor.validate_call("move_joints", [[math.radians(v) for v in joints]], {"duration_ms": ms})


# --------------------------------------------------------------------------- the simulated pair


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _sim_arm(kind: str, robot_id: str, tmp_path: Path, **kwargs: Any) -> Any:
    pytest.importorskip("monorobot.hal.adapters.sim_so101")
    from monorobot.hal.registry import load_adapter

    arm = load_adapter(kind, isolate=False, robot_id=robot_id, state_dir=tmp_path / "sim", **kwargs)
    assert arm.connect()
    return arm


@pytest.fixture
def sim_follower(tmp_path: Path) -> Any:
    """The real simulated SO-101 follower, connected, in this process, its clock ``SIM_SPEEDUP`` times real time.

    Fails the test when the arm logs its "single goal ... degrees from present" warning, or cuts a
    goal it was sent to 10 degrees from the arm ("was cut to").
    """
    records = _Records()
    sim_log = logging.getLogger("monorobot.hal.adapters.sim_so101")
    sim_log.addHandler(records)
    arm = _sim_arm("sim_so101_follower", "sim-follower-1", tmp_path, time_scale=SIM_SPEEDUP)
    try:
        yield arm
    finally:
        if arm.is_connected():
            arm.disconnect()
        sim_log.removeHandler(records)
    jumps = [m for m in records.messages if SINGLE_GOAL in m or CUT_GOAL in m]
    assert not jumps, jumps
    assert arm.big_goals == 0 and getattr(arm, "cut_goals", 0) == 0


def sim_joints_deg(arm: Any) -> list[float]:
    return [math.degrees(v) for v in arm.get_joint_positions()]


def wait_until_still(arm: Any, timeout_s: float = 5.0) -> list[float]:
    """The simulated arm's joints once two reads 50 ms apart agree."""
    deadline = time.monotonic() + timeout_s
    last = sim_joints_deg(arm)
    while time.monotonic() < deadline:
        time.sleep(0.05)
        now = sim_joints_deg(arm)
        if max(abs(a - b) for a, b in zip(now, last)) < 1e-3:
            return now
        last = now
    return last


def run_under_sdk(
    main: ModuleType,
    tmp_path: Path,
    *,
    leader: str = "on",
    timeout_s: float = 60.0,
) -> tuple[Any, int, dict[str, Any]]:
    """Run the app with the SDK's own ``run()`` on the simulated pair; return ``(app, exit code, follower state)``.

    The app's ``robot.yaml`` with the simulated follower in place of the real one and the
    simulated leader in place of the real one in ``also_uses``; the SDK builds the leader, as
    ``self.peers["leader"]``. ``leader`` is ``on``; ``unplugged`` (its state file says it isn't
    connected, so the SDK leaves it out); or ``off`` (no ``also_uses``). Both run in this process.
    The follower's clock runs ``SIM_SPEEDUP`` times real time, so its fold to rest is quick; the
    leader keeps real time, so the app (at its own rate) sees a hand move at a human pace. A run
    still going after ``timeout_s`` is ended with ``finish()`` and fails the test. The follower
    state is its state file once ``run()`` has returned. Fails on a "single goal" or "was cut to" warning.
    """
    pytest.importorskip("monorobot.hal.adapters.sim_so101")
    from monorobot.hal.adapters.sim_so101._state import update_state
    from monorobot.sdk.app_config import AppConfig

    sim = tmp_path / "sim"
    config = AppConfig.load(APP_DIR / "robot.yaml")
    config.robot.type = "sim_so101_follower"
    config.robot.isolate = False
    config.robot.kwargs = {**config.robot.kwargs, "robot_id": "sim-follower-1",
                           "state_dir": str(sim), "time_scale": SIM_SPEEDUP}
    if leader == "off":
        config.also_uses = []
    else:
        assert [peer.name for peer in config.also_uses] == ["leader"]
        peer = config.also_uses[0]
        assert peer.required is False
        peer.type = "sim_so101_leader"
        peer.isolate = False
        peer.kwargs = {**peer.kwargs, "robot_id": "sim-leader-1", "state_dir": str(sim)}
        if leader == "unplugged":
            update_state(sim / "sim-leader-1.json", {"connected": False})
    app = getattr(main, APP_CLASS)(config)
    done, timed_out = threading.Event(), threading.Event()

    def watch() -> None:
        if not done.wait(timeout_s):
            timed_out.set()
            app.finish()

    records = _Records()
    sim_log = logging.getLogger("monorobot.hal.adapters.sim_so101")
    sim_log.addHandler(records)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        code = app.run()
    finally:
        done.set()
        watcher.join(timeout=1.0)
        sim_log.removeHandler(records)
    assert not timed_out.is_set(), f"the run took over {timeout_s} s"
    jumps = [m for m in records.messages if SINGLE_GOAL in m or CUT_GOAL in m]
    assert not jumps, jumps
    assert app.robot.big_goals == 0 and getattr(app.robot, "cut_goals", 0) == 0
    state = json.loads((sim / "sim-follower-1.json").read_text())
    return app, code, state


# --------------------------------------------------------------------------- processes the tests start


def sandbox_env(tmp_path: Path) -> dict[str, str]:
    """The environment of an engine a test starts: home and monorobot home under the temp folder,
    no tokens, no app data folder from outside, an empty GitHub CLI config, and the sim-only
    interlock on."""
    home = tmp_path / "proc-home"
    (home / "gh").mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS and k != "MONOROBOT_APP_DATA"}
    env.update({"HOME": str(home), "MONOROBOT_HOME": str(home / ".monorobot"), "GH_CONFIG_DIR": str(home / "gh")})
    env.update(SIM_ONLY_SWITCHES)
    for name in ("HOME", "MONOROBOT_HOME"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), f"{name} is outside the temp folder"
    return env


def copy_app(tmp_path: Path, where: str) -> Path:
    copy = tmp_path / where / APP_DIR.name
    shutil.copytree(APP_DIR, copy, ignore=shutil.ignore_patterns(".git", "dist", "__pycache__", ".pytest_cache"))
    return copy


def built_manifest(tmp_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Run ``monorobot build`` on a copy of the app folder; return the package's manifest and member names.

    The engine is the one running these tests. The build runs in a sandbox home with no tokens and
    the sim-only interlock on.
    """
    copy = copy_app(tmp_path, "build")
    out = tmp_path / "build" / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "monorobot", "build", str(copy), "--out-dir", str(out)],
        cwd=copy, env=sandbox_env(tmp_path), capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    packages = list(out.glob("*.mrbot"))
    assert len(packages) == 1, packages
    with tarfile.open(packages[0], "r:gz") as tar:
        names = tar.getnames()
        member = tar.extractfile("manifest.json")
        assert member is not None
        manifest = json.loads(member.read())
    return manifest, names


def sim_pair_copy(tmp_path: Path, *, record_s: float) -> tuple[Path, Path, Path]:
    """A copy of the app on the simulated pair; return ``(app folder, robot settings file, sim state folder)``.

    The copy records ``record_s`` seconds instead of 15. Its ``robot.yaml`` names the simulated
    types: ``monorobot start`` reads no run settings file yet that could swap a type. The
    follower's id and state folder come from a robot settings file passed with ``--binding``, as a
    binding on ``sim_so101_follower``; the leader's are in ``also_uses`` (a binding reaches only
    the main robot so far). Everything else is the app's own. Checked before anything starts: no
    robot it names is a real one, and none has a port.
    """
    copy = copy_app(tmp_path, "start")
    sim = tmp_path / "start" / "sim"
    code = (copy / "main.py").read_text()
    assert code.count("\nRECORD_S = 15.0\n") == 1
    (copy / "main.py").write_text(code.replace("\nRECORD_S = 15.0\n", f"\nRECORD_S = {float(record_s)!r}\n"))
    data = yaml.safe_load((copy / "robot.yaml").read_text())
    data["robot"]["type"] = "sim_so101_follower"
    for peer in data.get("also_uses") or []:
        peer["type"] = "sim_so101_leader"
        peer["kwargs"] = {**(peer.get("kwargs") or {}), "robot_id": "sim-leader-1", "state_dir": str(sim)}
    (copy / "robot.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    binding = tmp_path / "start" / "bindings.yaml"
    binding.write_text(yaml.safe_dump(
        {"robots": {"sim_so101_follower": {"kwargs": {"robot_id": "sim-follower-1", "state_dir": str(sim)}}}}
    ))
    written = yaml.safe_load((copy / "robot.yaml").read_text())
    robots = [written["robot"], *(written.get("also_uses") or [])]
    assert all(robot["type"].startswith("sim_so101_") for robot in robots), robots
    assert all("port" not in (robot.get("kwargs") or {}) for robot in robots), robots
    assert list(yaml.safe_load(binding.read_text())["robots"]) == ["sim_so101_follower"]
    return copy, binding, sim


def start_on_sim_pair(copy: Path, binding: Path, tmp_path: Path) -> subprocess.Popen[str]:
    """``monorobot start <copy> --binding <file>`` in a process of its own, in the sandbox, its log lines on stdout."""
    return subprocess.Popen(
        [sys.executable, "-m", "monorobot", "start", str(copy), "--binding", str(binding)],
        cwd=copy, env=sandbox_env(tmp_path), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )


class LineReader:
    """Collects a process's output lines on a thread, so a test can wait for one."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.lines: list[str] = []
        self._new = threading.Condition()
        self._thread = threading.Thread(target=self._read, args=(proc,), daemon=True)
        self._thread.start()

    def _read(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with self._new:
                self.lines.append(line.rstrip("\n"))
                self._new.notify_all()

    def wait_for(self, text: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._new:
            while not any(text in line for line in self.lines):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._new.wait(left)
            return True

    def text(self) -> str:
        self._thread.join(timeout=5.0)
        return "\n".join(self.lines)
