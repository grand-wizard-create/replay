# Replay

Record a motion with the leader arm, then play it back on the follower arm. Without the leader,
Replay plays your newest recording, or a short demo when there is none yet.

## What it runs on

- The SO-101 follower arm, or the simulated SO-101 follower.
- The SO-101 leader arm, or the simulated SO-101 leader. Optional: with it Replay records, without
  it Replay plays back.
- The Lynxmotion AL5D arm, which plays back only.

Every move is smooth: the arm never jumps. With the leader, the follower first glides onto it at
no more than 30 degrees a second, then copies it for 15 s while Replay records, then glides back
to where the recording began and plays it. Playback follows the recording, but no joint moves
faster than 45 degrees a second: a quick part of a recording plays slower than you moved it. It
keeps a little inside the arm's limits. When the playback ends, Replay ends, and the SO-101 folds
back to rest.

Recordings stay on this Mac, in the app's own data folder, under `recordings/`. Replay keeps the
20 newest for each arm. Uninstalling Replay keeps them.

## Run it from Robot App Store

Find Replay in Discover, install it, choose your arm, and click Run. To record, plug in the
leader arm first, wait for "Recording 15 s" in the log, then move the leader.

## Run it from a terminal

```
monorobot pull grand-wizard-create/replay
monorobot robot bind hf_so101_follower --kwarg port=<port>
```

To record, Replay needs the leader's port too. A robot binding reaches only the follower for now,
so the leader's port goes in the app's own `robot.yaml`. Open `~/.monorobot/apps/replay/robot.yaml`
and add the port to the leader's `kwargs`, under `also_uses`:

```yaml
    kwargs: {use_default_calibration: true, port: <leader port>}
```

Leave it out to play back only. Then start it:

```
monorobot start replay
```

Replace `<port>` and `<leader port>` with the USB ports of your two arms.

## Stop it

In Robot App Store, press Space or Esc. In a terminal, press Ctrl+C: the follower folds back to
rest. Press Ctrl+C a second time to stop the arm where it is. A recording you stop keeps what it
recorded so far.

## License

Apache License 2.0. See `LICENSE`.
