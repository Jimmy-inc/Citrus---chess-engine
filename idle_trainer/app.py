"""Menu bar app: trains the chessbot net whenever the Mac is idle.

Run with:
    .venv/bin/python3 -m idle_trainer

Picks a dataset and net size (blocks/channels) from the menu or the full
Settings window, watches system idle time, and runs train.py as a
subprocess during idle windows. Pausing (config change, going active
again, switching to Paused mode, or quitting) sends SIGTERM, which
train.py already treats as "save checkpoint and exit cleanly".
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import rumps
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from Foundation import NSURL

from idle_trainer.config import (
    NET_SIZE_PRESETS,
    PROJECT_ROOT,
    USER_DATA_DIR,
    ckpt_dir_for,
    is_compatible_dataset,
    load_config,
    save_config,
)
from idle_trainer.idle import idle_seconds

TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
# Also doubles as how quickly a pending SIGTERM gets noticed: Python only
# processes signals when control returns to the interpreter, which here
# only happens on this timer's tick (rumps otherwise blocks in Cocoa's
# native run loop). Keep this short - idle_seconds() is a cheap syscall.
IDLE_CHECK_INTERVAL_SECONDS = 2
GRACEFUL_STOP_TIMEOUT_SECONDS = 15


def pick_dataset_file():
    """Native Finder-style open panel, restricted to the data/ dir."""
    from AppKit import NSOpenPanel

    panel = NSOpenPanel.openPanel()
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(False)
    panel.setAllowsMultipleSelection_(False)
    data_dir = USER_DATA_DIR / "data"
    if data_dir.exists():
        panel.setDirectoryURL_(NSURL.fileURLWithPath_(str(data_dir)))
    if panel.runModal() == 1:  # NSModalResponseOK / NSFileHandlingPanelOKButton
        return panel.URLs()[0].path()
    return None


class IdleTrainerApp(rumps.App):
    def __init__(self):
        super().__init__("Idle Chess Trainer", title="I", quit_button=None)
        self.config = load_config()
        self.proc = None
        self.log_file = None
        self.last_idle_seconds = 0.0
        self.settings_controller = None

        self.status_item = rumps.MenuItem("Status: starting…")
        self.settings_item = rumps.MenuItem(
            "Settings…", callback=self.open_settings
        )
        self.dataset_item = rumps.MenuItem(self._dataset_label())
        self.choose_dataset_item = rumps.MenuItem(
            "Choose dataset…", callback=self.choose_dataset
        )

        self.netsize_menu = rumps.MenuItem("Net size")
        for name, (blocks, channels) in NET_SIZE_PRESETS.items():
            item = rumps.MenuItem(
                f"{name}  ({blocks} blocks × {channels} channels)",
                callback=self._make_netsize_callback(name),
            )
            item.state = name == self.config["net_size"]
            self.netsize_menu.add(item)

        self.threshold_item = rumps.MenuItem(
            "Idle threshold…", callback=self.set_threshold
        )
        self.quit_item = rumps.MenuItem("Quit", callback=self.quit_clicked)

        self.menu = [
            self.status_item,
            None,
            self.settings_item,
            None,
            self.dataset_item,
            self.choose_dataset_item,
            self.netsize_menu,
            self.threshold_item,
            None,
            self.quit_item,
        ]

        signal.signal(signal.SIGTERM, self._signal_quit)
        signal.signal(signal.SIGINT, self._signal_quit)

        self.timer = rumps.Timer(self.tick, IDLE_CHECK_INTERVAL_SECONDS)
        self.timer.start()

    # ------------------------------------------------------------ labels

    def _dataset_label(self):
        return f"Dataset: {Path(self.config['dataset']).name}"

    def _make_netsize_callback(self, name):
        def callback(sender):
            for item in self.netsize_menu.values():
                item.state = False
            sender.state = True
            self.config["net_size"] = name
            save_config(self.config)
            self._restart_if_training()

        return callback

    # ------------------------------------------------------------ menu actions

    def open_settings(self, _sender):
        from idle_trainer.settings_window import SettingsWindowController

        if self.settings_controller is None:
            self.settings_controller = (
                SettingsWindowController.alloc().init().setup_(self)
            )
        self.settings_controller.show()

    def choose_dataset(self, _sender):
        path = pick_dataset_file()
        if not path:
            return
        if not is_compatible_dataset(path):
            rumps.alert(
                title="This file isn't a training dataset",
                message=(
                    f"{Path(path).name} isn't in the record format train.py "
                    f"expects, so training would crash immediately. Choose "
                    f"a dataset built by extract.py instead (e.g. "
                    f"positions.bin or one of the tcec*.bin files)."
                ),
            )
            return
        self.config["dataset"] = path
        save_config(self.config)
        self.dataset_item.title = self._dataset_label()
        self._restart_if_training()

    def set_threshold(self, _sender):
        response = rumps.Window(
            title="Idle threshold",
            message="Start training after the Mac has been idle for this many seconds:",
            default_text=str(self.config["idle_threshold_seconds"]),
            ok="Set",
            cancel="Cancel",
        ).run()
        if not response.clicked:
            return
        try:
            seconds = int(response.text)
        except ValueError:
            return
        if seconds > 0:
            self.config["idle_threshold_seconds"] = seconds
            save_config(self.config)

    def quit_clicked(self, _sender):
        self.stop_training(wait=True)
        rumps.quit_application()

    def _signal_quit(self, _signum, _frame):
        self.stop_training(wait=True)
        rumps.quit_application()

    # ------------------------------------------------------------ training lifecycle

    def is_training(self):
        return self.proc is not None and self.proc.poll() is None

    def start_training(self):
        blocks, channels = NET_SIZE_PRESETS[self.config["net_size"]]
        ckpt_dir = ckpt_dir_for(self.config)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data", self.config["dataset"],
            "--blocks", str(blocks),
            "--channels", str(channels),
            "--ckpt-dir", ckpt_dir,
            "--max-gpu-pct", str(self.config.get("max_gpu_pct", 100)),
        ]
        self.log_file = open(Path(ckpt_dir) / "train.log", "a")
        env = os.environ.copy()
        mem_pct = self.config.get("max_gpu_mem_pct", 100)
        if mem_pct < 100:
            # Apple Silicon has unified memory, so this is the closest real
            # lever to a "max RAM" cap: it constrains how much of that
            # shared pool PyTorch's MPS allocator will use before erroring,
            # rather than a literal whole-process RAM limit (macOS has no
            # reliable per-process one of those to offer here). Low must
            # stay <= high or MPS init itself raises; keep the same ~0.82
            # ratio between them that PyTorch's own defaults (1.4 / 1.7) use.
            high_ratio = 1.7 * mem_pct / 100.0
            env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_ratio)
            env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(high_ratio * 1.4 / 1.7)
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=self.log_file,
            stderr=subprocess.STDOUT, env=env,
        )

    def stop_training(self, wait):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            if wait:
                try:
                    self.proc.wait(timeout=GRACEFUL_STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if wait or self.proc.poll() is not None:
            self.proc = None
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None

    def _restart_if_training(self):
        # Config changed - pause now (without blocking the menu click); the
        # next tick() will notice the old process has exited and relaunch
        # with the new config, since we're still past the idle threshold.
        if self.is_training():
            self.stop_training(wait=False)

    # ------------------------------------------------------------ idle loop

    def tick(self, _timer):
        idle = idle_seconds()
        self.last_idle_seconds = idle
        mode = self.config.get("mode", "auto")

        if mode == "paused":
            should_train = False
        elif mode == "always":
            should_train = True
        else:
            should_train = idle >= self.config["idle_threshold_seconds"]

        if not self.is_training():
            if self.proc is not None:
                # Previous subprocess has since exited; finish cleanup.
                self.stop_training(wait=True)
            if should_train:
                self.start_training()
        elif not should_train:
            self.stop_training(wait=False)

        self._refresh_status()

    def _refresh_status(self):
        if self.is_training():
            self.title = "T"
            self.status_item.title = "Status: training"
        else:
            self.title = "I"
            mode = self.config.get("mode", "auto")
            if mode == "paused":
                self.status_item.title = "Status: paused"
            elif mode == "always":
                self.status_item.title = "Status: starting…"
            else:
                remaining = max(
                    0,
                    self.config["idle_threshold_seconds"]
                    - int(self.last_idle_seconds),
                )
                self.status_item.title = f"Status: waiting for idle ({remaining}s left)"


def main():
    # Menu-bar-only, no Dock icon - matters even before this is packaged
    # into a proper .app bundle.
    NSApplication.sharedApplication().setActivationPolicy_(
        NSApplicationActivationPolicyAccessory
    )
    IdleTrainerApp().run()


if __name__ == "__main__":
    main()
