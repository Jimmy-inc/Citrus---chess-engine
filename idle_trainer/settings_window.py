"""Native settings window for the idle-time trainer menu bar app.

Settings panel on the left (plain frame-based AppKit layout, fixed pixel
rows - it doesn't need to reflow) and the Play board on the right, both
in one window. The window is resizable/full-screen capable; the board
fills whatever space it's given (see chess_view.py) while the settings
panel stays a fixed width, pinned to the top-left.
"""

import glob
import threading
from pathlib import Path

import objc
from AppKit import (
    NSAlert,
    NSApp,
    NSBackingStoreBuffered,
    NSBezelBorder,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSMakeSize,
    NSPopUpButton,
    NSScrollView,
    NSSegmentedControl,
    NSSegmentStyleRounded,
    NSSlider,
    NSTextAlignmentCenter,
    NSTextField,
    NSTextView,
    NSView,
    NSViewHeightSizable,
    NSViewMinYMargin,
    NSViewWidthSizable,
    NSWindow,
    NSWindowCollectionBehaviorFullScreenPrimary,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSNotificationCenter, NSObject, NSTimer
from PyObjCTools import AppHelper

from idle_trainer.config import (
    DEFAULTS,
    NET_SIZE_PRESETS,
    PROJECT_ROOT,
    USER_DATA_DIR,
    ckpt_dir_for,
    is_compatible_dataset,
    save_config,
)

WIDTH = 460
HEIGHT = 924
NET_SIZE_NAMES = list(NET_SIZE_PRESETS.keys())
MODE_LABELS = ["Auto (idle-based)", "Always train", "Paused"]
MODE_VALUES = ["auto", "always", "paused"]


def _label(text, x, y, w, bold=False):
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, 20))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    if bold:
        field.setFont_(NSFont.boldSystemFontOfSize_(13))
    return field


def _read_step_count(ckpt_dir):
    """Step count from a net's checkpoint dir, or None if it's never been
    trained. Prefers latest.pt (more current) over best.pt."""
    import torch
    from pathlib import Path

    for name in ("latest.pt", "best.pt"):
        path = Path(ckpt_dir) / name
        if path.exists():
            try:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
                return ckpt.get("step")
            except Exception:
                return None
    return None


def _log_tail(path, max_lines=30):
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return "(no training output yet)"
    return "".join(lines[-max_lines:]).strip() or "(no training output yet)"


class SettingsWindowController(NSObject):

    def init(self):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        self.trainer_app = None
        self.timer = None
        self.window = None
        return self

    def setup_(self, trainer_app):
        self.trainer_app = trainer_app
        self._build_ui()
        NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self, "windowWillClose:", "NSWindowWillCloseNotification", self.window
        )
        return self

    # ------------------------------------------------------------ layout

    def _build_ui(self):
        board_width = 500
        total_width = WIDTH + board_width

        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, total_width, HEIGHT), style, NSBackingStoreBuffered, False
        )
        window.setTitle_("Idle Chess Trainer")
        window.center()
        window.setReleasedWhenClosed_(False)
        window.setMinSize_(NSMakeSize(WIDTH + 320, 480))
        window.setCollectionBehavior_(NSWindowCollectionBehaviorFullScreenPrimary)
        self.window = window

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, total_width, HEIGHT))

        settings_view = self._build_settings_panel()
        settings_view.setAutoresizingMask_(NSViewMinYMargin)
        root.addSubview_(settings_view)

        from idle_trainer.play_tab import PlayTabController

        self.play_controller = PlayTabController.alloc().init().setup_(self.trainer_app)
        board_frame = NSMakeRect(WIDTH, 0, total_width - WIDTH, HEIGHT)
        self.play_controller.view.setFrame_(board_frame)
        self.play_controller.view.setAutoresizingMask_(
            NSViewWidthSizable | NSViewHeightSizable
        )
        root.addSubview_(self.play_controller.view)

        window.setContentView_(root)

    def _build_settings_panel(self):
        config = self.trainer_app.config
        y = HEIGHT - 60

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, HEIGHT))

        content.addSubview_(_label("Status", 20, y, 200, bold=True))
        self.status_label = _label("", 20, y - 22, WIDTH - 40, bold=False)
        content.addSubview_(self.status_label)
        y -= 56

        content.addSubview_(_label("Dataset", 20, y, 200, bold=True))
        y -= 26
        self.dataset_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 130, 26)
        )
        self.dataset_popup.setTarget_(self)
        self.dataset_popup.setAction_("datasetChanged:")
        content.addSubview_(self.dataset_popup)
        self._reload_dataset_options(config["dataset"])

        browse_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WIDTH - 100, y, 80, 26)
        )
        browse_btn.setTitle_("Browse…")
        browse_btn.setBezelStyle_(1)
        browse_btn.setTarget_(self)
        browse_btn.setAction_("browseClicked:")
        content.addSubview_(browse_btn)
        y -= 40

        content.addSubview_(_label("Net size", 20, y, 200, bold=True))
        y -= 26
        self.netsize_seg = NSSegmentedControl.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 40, 26)
        )
        self.netsize_seg.setSegmentStyle_(NSSegmentStyleRounded)
        self.netsize_seg.setSegmentCount_(len(NET_SIZE_NAMES))
        for i, name in enumerate(NET_SIZE_NAMES):
            blocks, channels = NET_SIZE_PRESETS[name]
            self.netsize_seg.setLabel_forSegment_(f"{name} ({blocks}×{channels})", i)
            self.netsize_seg.setWidth_forSegment_((WIDTH - 40) / len(NET_SIZE_NAMES), i)
        self.netsize_seg.setSelectedSegment_(NET_SIZE_NAMES.index(config["net_size"]))
        self.netsize_seg.setTarget_(self)
        self.netsize_seg.setAction_("netSizeChanged:")
        content.addSubview_(self.netsize_seg)
        y -= 22

        seg_w = (WIDTH - 40) / len(NET_SIZE_NAMES)
        self.step_count_labels = {}
        for i, name in enumerate(NET_SIZE_NAMES):
            lbl = _label("…", 20 + i * seg_w, y, seg_w, bold=False)
            lbl.setFont_(NSFont.systemFontOfSize_(10))
            lbl.setTextColor_(NSColor.secondaryLabelColor())
            lbl.setAlignment_(NSTextAlignmentCenter)
            content.addSubview_(lbl)
            self.step_count_labels[name] = lbl
        y -= 32

        content.addSubview_(_label("Mode", 20, y, 200, bold=True))
        y -= 26
        self.mode_seg = NSSegmentedControl.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 40, 26)
        )
        self.mode_seg.setSegmentStyle_(NSSegmentStyleRounded)
        self.mode_seg.setSegmentCount_(len(MODE_LABELS))
        for i, label in enumerate(MODE_LABELS):
            self.mode_seg.setLabel_forSegment_(label, i)
            self.mode_seg.setWidth_forSegment_((WIDTH - 40) / len(MODE_LABELS), i)
        self.mode_seg.setSelectedSegment_(MODE_VALUES.index(config.get("mode", "auto")))
        self.mode_seg.setTarget_(self)
        self.mode_seg.setAction_("modeChanged:")
        content.addSubview_(self.mode_seg)
        y -= 40

        content.addSubview_(
            _label("Idle threshold before training starts", 20, y, 300, bold=True)
        )
        y -= 26
        self.threshold_slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 130, 24)
        )
        self.threshold_slider.setMinValue_(30)
        self.threshold_slider.setMaxValue_(1800)
        self.threshold_slider.setDoubleValue_(config["idle_threshold_seconds"])
        self.threshold_slider.setContinuous_(True)
        self.threshold_slider.setTarget_(self)
        self.threshold_slider.setAction_("thresholdChanged:")
        content.addSubview_(self.threshold_slider)
        self.threshold_value_label = _label(
            f"{config['idle_threshold_seconds']}s", WIDTH - 100, y - 2, 80
        )
        content.addSubview_(self.threshold_value_label)
        y -= 40

        content.addSubview_(_label("Max GPU usage", 20, y, 300, bold=True))
        y -= 26
        self.gpu_pct_slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 130, 24)
        )
        self.gpu_pct_slider.setMinValue_(10)
        self.gpu_pct_slider.setMaxValue_(100)
        self.gpu_pct_slider.setDoubleValue_(config.get("max_gpu_pct", 100))
        self.gpu_pct_slider.setContinuous_(True)
        self.gpu_pct_slider.setTarget_(self)
        self.gpu_pct_slider.setAction_("gpuPctChanged:")
        content.addSubview_(self.gpu_pct_slider)
        self.gpu_pct_value_label = _label(
            f"{config.get('max_gpu_pct', 100):.0f}%", WIDTH - 100, y - 2, 80
        )
        content.addSubview_(self.gpu_pct_value_label)
        y -= 40

        content.addSubview_(_label("Max GPU memory", 20, y, 300, bold=True))
        y -= 26
        self.gpu_mem_slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(20, y, WIDTH - 130, 24)
        )
        self.gpu_mem_slider.setMinValue_(20)
        self.gpu_mem_slider.setMaxValue_(100)
        self.gpu_mem_slider.setDoubleValue_(config.get("max_gpu_mem_pct", 100))
        self.gpu_mem_slider.setContinuous_(True)
        self.gpu_mem_slider.setTarget_(self)
        self.gpu_mem_slider.setAction_("gpuMemPctChanged:")
        content.addSubview_(self.gpu_mem_slider)
        self.gpu_mem_value_label = _label(
            f"{config.get('max_gpu_mem_pct', 100):.0f}%", WIDTH - 100, y - 2, 80
        )
        content.addSubview_(self.gpu_mem_value_label)
        y -= 40

        content.addSubview_(_label("Recent output", 20, y, 200, bold=True))
        y -= 310
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, y, WIDTH - 40, 300))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(NSBezelBorder)
        text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH - 40, 300))
        text_view.setEditable_(False)
        text_view.setFont_(NSFont.userFixedPitchFontOfSize_(11))
        scroll.setDocumentView_(text_view)
        self.log_view = text_view
        content.addSubview_(scroll)
        y -= 50

        quit_btn = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 120, 28))
        quit_btn.setTitle_("Quit App")
        quit_btn.setBezelStyle_(1)
        quit_btn.setTarget_(self)
        quit_btn.setAction_("quitClicked:")
        content.addSubview_(quit_btn)

        self.step_counts = {name: None for name in NET_SIZE_NAMES}
        self._refresh()
        self._refresh_step_counts_async()
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0, self, "refreshTick:", None, True
        )
        return content

    def _reload_dataset_options(self, current_path):
        self.dataset_popup.removeAllItems()
        data_dir = USER_DATA_DIR / "data"
        # Only offer files train.py can actually read - e.g. evals*.bin is a
        # different record format (built for train_evals.py, not train.py)
        # and would crash the moment training tried to memmap it.
        options = sorted(
            p for p in glob.glob(str(data_dir / "*.bin")) if is_compatible_dataset(p)
        )

        if not is_compatible_dataset(current_path):
            # Someone (a stale config, or a Browse pick from before this
            # check existed) left an unusable dataset selected - fix it now
            # rather than let it crash-loop the next time training starts.
            fallback = options[0] if options else DEFAULTS["dataset"]
            self.trainer_app.config["dataset"] = fallback
            save_config(self.trainer_app.config)
            self.trainer_app.dataset_item.title = self.trainer_app._dataset_label()
            self.trainer_app._restart_if_training()
            current_path = fallback

        if current_path not in options:
            options.append(current_path)
        for path in options:
            self.dataset_popup.addItemWithTitle_(path)
        self.dataset_popup.selectItemWithTitle_(current_path)

    # ------------------------------------------------------------ actions

    def datasetChanged_(self, sender):
        path = self.dataset_popup.titleOfSelectedItem()
        self.trainer_app.config["dataset"] = path
        save_config(self.trainer_app.config)
        self.trainer_app.dataset_item.title = self.trainer_app._dataset_label()
        self.trainer_app._restart_if_training()
        self._refresh()
        self._refresh_step_counts_async()

    def browseClicked_(self, sender):
        from idle_trainer.app import pick_dataset_file

        path = pick_dataset_file()
        if not path:
            return
        if not is_compatible_dataset(path):
            alert = NSAlert.alloc().init()
            alert.setMessageText_("This file isn't a training dataset")
            alert.setInformativeText_(
                f"{Path(path).name} isn't in the record format train.py "
                f"expects, so training would crash immediately if you "
                f"picked it. Choose a dataset built by extract.py instead "
                f"(e.g. positions.bin or one of the tcec*.bin files)."
            )
            alert.addButtonWithTitle_("OK")
            alert.runModal()
            return
        self.trainer_app.config["dataset"] = path
        save_config(self.trainer_app.config)
        self.trainer_app.dataset_item.title = self.trainer_app._dataset_label()
        self._reload_dataset_options(path)
        self.trainer_app._restart_if_training()
        self._refresh()
        self._refresh_step_counts_async()

    def netSizeChanged_(self, sender):
        old_name = self.trainer_app.config["net_size"]
        new_name = NET_SIZE_NAMES[self.netsize_seg.selectedSegment()]
        if new_name == old_name:
            return

        old_steps = self.step_counts.get(old_name) or 0
        new_steps = self.step_counts.get(new_name) or 0
        if new_steps < old_steps:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Switch to a less-trained net?")
            if new_steps == 0:
                detail = (
                    f"{new_name} has no training steps yet - it's a completely "
                    f"separate, untrained network, not a resized version of "
                    f"{old_name} ({old_steps:,} steps)."
                )
            else:
                detail = (
                    f"{new_name} has {new_steps:,} training steps, versus "
                    f"{old_steps:,} for {old_name}. These are separate networks "
                    f"with independent training progress, not the same net at "
                    f"a different size."
                )
            alert.setInformativeText_(detail)
            alert.addButtonWithTitle_("Switch Anyway")
            alert.addButtonWithTitle_("Cancel")
            response = alert.runModal()
            if response != 1000:  # NSAlertFirstButtonReturn
                self.netsize_seg.setSelectedSegment_(NET_SIZE_NAMES.index(old_name))
                return

        self.trainer_app.config["net_size"] = new_name
        save_config(self.trainer_app.config)
        for item in self.trainer_app.netsize_menu.values():
            item.state = item.title.startswith(new_name)
        self.trainer_app._restart_if_training()
        self._refresh()

    def modeChanged_(self, sender):
        mode = MODE_VALUES[self.mode_seg.selectedSegment()]
        self.trainer_app.config["mode"] = mode
        save_config(self.trainer_app.config)
        if mode == "paused":
            self.trainer_app.stop_training(wait=False)
        self._refresh()

    def thresholdChanged_(self, sender):
        seconds = int(self.threshold_slider.doubleValue())
        self.trainer_app.config["idle_threshold_seconds"] = seconds
        self.threshold_value_label.setStringValue_(f"{seconds}s")
        save_config(self.trainer_app.config)

    def gpuPctChanged_(self, sender):
        pct = int(self.gpu_pct_slider.doubleValue())
        self.trainer_app.config["max_gpu_pct"] = pct
        self.gpu_pct_value_label.setStringValue_(f"{pct}%")
        save_config(self.trainer_app.config)

    def gpuMemPctChanged_(self, sender):
        pct = int(self.gpu_mem_slider.doubleValue())
        self.trainer_app.config["max_gpu_mem_pct"] = pct
        self.gpu_mem_value_label.setStringValue_(f"{pct}%")
        save_config(self.trainer_app.config)

    def quitClicked_(self, sender):
        self.trainer_app.quit_clicked(None)

    def refreshTick_(self, timer):
        self._refresh()

    def windowWillClose_(self, notification):
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None

    # ------------------------------------------------------------ display

    def _refresh(self):
        app = self.trainer_app
        if app.is_training():
            self.status_label.setStringValue_("Training now (system is idle)")
        else:
            remaining = max(
                0, app.config["idle_threshold_seconds"] - int(app.last_idle_seconds)
            )
            if app.config.get("mode") == "paused":
                self.status_label.setStringValue_("Paused")
            elif app.config.get("mode") == "always":
                self.status_label.setStringValue_("Waiting to start (always-on mode)")
            else:
                self.status_label.setStringValue_(
                    f"Waiting for idle - {remaining}s left"
                )
        log_path = ckpt_dir_for(app.config) + "/train.log"
        self.log_view.setString_(_log_tail(log_path))

    def _refresh_step_counts_async(self):
        dataset = self.trainer_app.config["dataset"]
        threading.Thread(
            target=self._step_counts_worker, args=(dataset,), daemon=True
        ).start()

    def _step_counts_worker(self, dataset):
        counts = {}
        for name in NET_SIZE_NAMES:
            ckpt_dir = ckpt_dir_for({"dataset": dataset, "net_size": name})
            counts[name] = _read_step_count(ckpt_dir)
        AppHelper.callAfter(self._apply_step_counts, counts)

    def _apply_step_counts(self, counts):
        self.step_counts = counts
        for name, label in self.step_count_labels.items():
            steps = counts.get(name)
            text = "not trained" if steps is None else f"{steps:,} steps"
            label.setStringValue_(text)

    def show(self):
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self._refresh()
