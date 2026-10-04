"""The "Play" tab: lets you play a casual game against the current best
checkpoint, using the same pure-network move selection as play.py's
think() (no tree search - just the policy head, quick enough to run
inline on a background thread)."""

import sys
import threading
from pathlib import Path

import objc
import chess
from AppKit import (
    NSButton,
    NSButtonTypeSwitch,
    NSFont,
    NSMakeRect,
    NSSegmentedControl,
    NSSegmentStyleRounded,
    NSSlider,
    NSTextField,
    NSView,
)
from Foundation import NSObject, NSTimer
from PyObjCTools import AppHelper

from idle_trainer.chess_view import ChessBoardView
from idle_trainer.config import PROJECT_ROOT, USER_DATA_DIR, ckpt_dir_for
from idle_trainer.eval_bar import EvalBarView, bot_value_to_white_cp, cp_to_white_fraction
from idle_trainer import stockfish_eval

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TEMPERATURE = 0.1  # matches play.py's own default: mostly the top move
DEFAULT_SIMS = 400  # matches search_batched.py's own default
DEFAULT_SELF_PLAY_DELAY = 1.0  # seconds between self-play moves


CHECKPOINT_MODE_CAPTIONS = {
    "best": "Best: highest validation score so far (may lag behind training)",
    "latest": "Latest: most recent training step - not yet validated",
}


def _find_checkpoint(config, mode="best"):
    """"latest" is this run's own resume checkpoint only - there's no
    project-wide "latest" to sensibly fall back to. "best" prefers the
    checkpoint for the app's current dataset/net-size config, falling back
    to the project's main hand-trained checkpoint."""
    per_config_dir = Path(ckpt_dir_for(config))
    if mode == "latest":
        path = per_config_dir / "latest.pt"
        return path if path.exists() else None

    per_config = per_config_dir / "best.pt"
    main_best = USER_DATA_DIR / "checkpoints" / "best.pt"
    for path in (per_config, main_best):
        if path.exists():
            return path
    return None


class PlayTabContainerView(NSView):
    """Manual (non-Auto Layout) 3-row layout: status bar / board / controls,
    kept in sync on every resize so full screen and window resizing work."""

    BAR_WIDTH = 28
    BAR_GAP = 6
    BAR_LABEL_H = 16

    def initWithFrame_(self, frame):
        self = objc.super(PlayTabContainerView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.status_label = None
        self.board_view = None
        self.bottom_bar = None
        self.bot_eval_bar = None
        self.sf_eval_bar = None
        self.bot_eval_label = None
        self.sf_eval_label = None
        self.eval_bars_visible = False
        return self

    def layout_now(self):
        if self.board_view is None:
            return
        w, h = self.bounds().size.width, self.bounds().size.height
        margin = 16
        top_h = 26
        bottom_h = 148

        self.status_label.setFrame_(
            NSMakeRect(margin, h - top_h - margin, w - 2 * margin, top_h)
        )
        self.bottom_bar.setFrame_(NSMakeRect(0, 0, w, bottom_h + margin))
        board_top = h - top_h - 2 * margin
        board_bottom = bottom_h + margin

        board_x = margin
        if self.eval_bars_visible:
            bars_w = 2 * self.BAR_WIDTH + self.BAR_GAP
            bar_h = board_top - board_bottom - self.BAR_LABEL_H - 2
            for i, (label, bar) in enumerate(
                ((self.bot_eval_label, self.bot_eval_bar),
                 (self.sf_eval_label, self.sf_eval_bar))
            ):
                x = margin + i * (self.BAR_WIDTH + self.BAR_GAP)
                label.setFrame_(
                    NSMakeRect(x, board_top - self.BAR_LABEL_H, self.BAR_WIDTH, self.BAR_LABEL_H)
                )
                bar.setFrame_(NSMakeRect(x, board_bottom, self.BAR_WIDTH, bar_h))
            board_x = margin + bars_w + margin

        self.board_view.setFrame_(
            NSMakeRect(board_x, board_bottom, w - margin - board_x, board_top - board_bottom)
        )

    def resizeSubviewsWithOldSize_(self, old_size):
        self.layout_now()

    def setFrame_(self, frame):
        objc.super(PlayTabContainerView, self).setFrame_(frame)
        self.layout_now()


class PlayTabController(NSObject):

    def init(self):
        self = objc.super(PlayTabController, self).init()
        if self is None:
            return None
        self.trainer_app = None
        self.view = None
        self.model = None
        self.model_path = None
        self.human_color = chess.WHITE
        return self

    def setup_(self, trainer_app):
        self.trainer_app = trainer_app
        self._build_ui()
        self._load_model_if_needed()
        return self

    # ------------------------------------------------------------ layout

    def _build_ui(self):
        container = PlayTabContainerView.alloc().initWithFrame_(NSMakeRect(0, 0, 460, 460))

        status = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 24))
        status.setBezeled_(False)
        status.setDrawsBackground_(False)
        status.setEditable_(False)
        status.setSelectable_(False)
        status.setFont_(NSFont.boldSystemFontOfSize_(13))
        status.setStringValue_("Loading…")
        container.addSubview_(status)
        container.status_label = status

        board = ChessBoardView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        board.on_human_move = self.on_human_move
        container.addSubview_(board)
        container.board_view = board

        def eval_title(text):
            f = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 28, 16))
            f.setBezeled_(False)
            f.setDrawsBackground_(False)
            f.setEditable_(False)
            f.setSelectable_(False)
            f.setAlignment_(1)  # NSTextAlignmentCenter
            f.setFont_(NSFont.systemFontOfSize_(9))
            f.setStringValue_(text)
            f.setHidden_(True)
            container.addSubview_(f)
            return f

        bot_eval_label = eval_title("Bot")
        sf_eval_label = eval_title("SF")
        bot_eval_bar = EvalBarView.alloc().initWithFrame_(NSMakeRect(0, 0, 28, 100))
        sf_eval_bar = EvalBarView.alloc().initWithFrame_(NSMakeRect(0, 0, 28, 100))
        bot_eval_bar.setHidden_(True)
        sf_eval_bar.setHidden_(True)
        container.addSubview_(bot_eval_bar)
        container.addSubview_(sf_eval_bar)
        container.bot_eval_label = bot_eval_label
        container.sf_eval_label = sf_eval_label
        container.bot_eval_bar = bot_eval_bar
        container.sf_eval_bar = sf_eval_bar
        self.bot_eval_label = bot_eval_label
        self.sf_eval_label = sf_eval_label
        self.bot_eval_bar = bot_eval_bar
        self.sf_eval_bar = sf_eval_bar

        bottom_bar = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 164))

        checkpoint_seg = NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(16, 120, 140, 26))
        checkpoint_seg.setSegmentStyle_(NSSegmentStyleRounded)
        checkpoint_seg.setSegmentCount_(2)
        checkpoint_seg.setLabel_forSegment_("Best", 0)
        checkpoint_seg.setLabel_forSegment_("Latest", 1)
        checkpoint_seg.setWidth_forSegment_(70, 0)
        checkpoint_seg.setWidth_forSegment_(70, 1)
        checkpoint_seg.setSelectedSegment_(0)
        checkpoint_seg.setTarget_(self)
        checkpoint_seg.setAction_("checkpointModeChanged:")
        bottom_bar.addSubview_(checkpoint_seg)
        self.checkpoint_seg = checkpoint_seg

        checkpoint_caption = NSTextField.alloc().initWithFrame_(NSMakeRect(166, 124, 320, 18))
        checkpoint_caption.setBezeled_(False)
        checkpoint_caption.setDrawsBackground_(False)
        checkpoint_caption.setEditable_(False)
        checkpoint_caption.setSelectable_(False)
        checkpoint_caption.setFont_(NSFont.systemFontOfSize_(10))
        checkpoint_caption.setStringValue_(CHECKPOINT_MODE_CAPTIONS["best"])
        bottom_bar.addSubview_(checkpoint_caption)
        self.checkpoint_caption = checkpoint_caption

        new_game_btn = NSButton.alloc().initWithFrame_(NSMakeRect(16, 84, 100, 26))
        new_game_btn.setTitle_("New Game")
        new_game_btn.setBezelStyle_(1)
        new_game_btn.setTarget_(self)
        new_game_btn.setAction_("newGameClicked:")
        bottom_bar.addSubview_(new_game_btn)

        color_seg = NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(126, 84, 160, 26))
        color_seg.setSegmentStyle_(NSSegmentStyleRounded)
        color_seg.setSegmentCount_(2)
        color_seg.setLabel_forSegment_("Play as White", 0)
        color_seg.setLabel_forSegment_("Play as Black", 1)
        color_seg.setWidth_forSegment_(80, 0)
        color_seg.setWidth_forSegment_(80, 1)
        color_seg.setSelectedSegment_(0)
        color_seg.setTarget_(self)
        color_seg.setAction_("colorChanged:")
        bottom_bar.addSubview_(color_seg)

        search_check = NSButton.alloc().initWithFrame_(NSMakeRect(16, 48, 80, 26))
        search_check.setButtonType_(NSButtonTypeSwitch)
        search_check.setTitle_("Search")
        search_check.setState_(0)
        search_check.setTarget_(self)
        search_check.setAction_("searchToggled:")
        bottom_bar.addSubview_(search_check)
        self.search_check = search_check

        sims_field = NSTextField.alloc().initWithFrame_(NSMakeRect(100, 48, 60, 24))
        sims_field.setStringValue_(str(DEFAULT_SIMS))
        sims_field.setTarget_(self)
        sims_field.setAction_("simsChanged:")
        sims_field.setHidden_(True)
        bottom_bar.addSubview_(sims_field)
        self.sims_field = sims_field

        eval_check = NSButton.alloc().initWithFrame_(NSMakeRect(180, 48, 100, 26))
        eval_check.setButtonType_(NSButtonTypeSwitch)
        eval_check.setTitle_("Eval bar")
        eval_check.setState_(0)
        eval_check.setTarget_(self)
        eval_check.setAction_("evalBarToggled:")
        bottom_bar.addSubview_(eval_check)
        self.eval_check = eval_check

        info = NSTextField.alloc().initWithFrame_(NSMakeRect(310, 48, 290, 26))
        info.setBezeled_(False)
        info.setDrawsBackground_(False)
        info.setEditable_(False)
        info.setSelectable_(False)
        info.setFont_(NSFont.systemFontOfSize_(11))
        bottom_bar.addSubview_(info)
        self.info_label = info

        self_play_check = NSButton.alloc().initWithFrame_(NSMakeRect(16, 16, 90, 26))
        self_play_check.setButtonType_(NSButtonTypeSwitch)
        self_play_check.setTitle_("Self-play")
        self_play_check.setState_(0)
        self_play_check.setTarget_(self)
        self_play_check.setAction_("selfPlayToggled:")
        bottom_bar.addSubview_(self_play_check)
        self.self_play_check = self_play_check

        def slider_with_label(x, title, min_v, max_v, default_v, action, value_text):
            label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, 24, 40, 16))
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setFont_(NSFont.systemFontOfSize_(10))
            label.setStringValue_(title)
            bottom_bar.addSubview_(label)

            slider = NSSlider.alloc().initWithFrame_(NSMakeRect(x, 8, 90, 20))
            slider.setMinValue_(min_v)
            slider.setMaxValue_(max_v)
            slider.setDoubleValue_(default_v)
            slider.setContinuous_(True)
            slider.setTarget_(self)
            slider.setAction_(action)
            bottom_bar.addSubview_(slider)

            value_label = NSTextField.alloc().initWithFrame_(NSMakeRect(x + 94, 16, 44, 16))
            value_label.setBezeled_(False)
            value_label.setDrawsBackground_(False)
            value_label.setEditable_(False)
            value_label.setSelectable_(False)
            value_label.setFont_(NSFont.systemFontOfSize_(10))
            value_label.setStringValue_(value_text)
            bottom_bar.addSubview_(value_label)
            return slider, value_label

        self.delay_slider, self.delay_value_label = slider_with_label(
            116, "Delay", 0.1, 5.0, DEFAULT_SELF_PLAY_DELAY,
            "delayChanged:", f"{DEFAULT_SELF_PLAY_DELAY:.1f}s",
        )
        self.temp_slider, self.temp_value_label = slider_with_label(
            306, "Temp", 0.0, 2.0, DEFAULT_TEMPERATURE,
            "tempChanged:", f"{DEFAULT_TEMPERATURE:.2f}",
        )

        container.addSubview_(bottom_bar)
        container.bottom_bar = bottom_bar

        container.layout_now()
        self.view = container
        self.board = board
        self.status_label = status
        # Bumped on every New Game / checkpoint switch; a bot-move or
        # eval-bar computation still running in the background from before
        # that checks this when it finishes and discards its result if it's
        # gone stale, instead of applying a move to a game that's since
        # moved on (or, worse, no longer legal on the reset board).
        self.game_id = 0
        self.checkpoint_mode = "best"
        self.search_enabled = False
        self.sims = DEFAULT_SIMS
        self.eval_bars_enabled = False
        self.temperature = DEFAULT_TEMPERATURE
        self.self_play_enabled = False
        self.self_play_delay = DEFAULT_SELF_PLAY_DELAY
        self.self_play_timer = None

    # ------------------------------------------------------------ model

    def _load_model_if_needed(self):
        path = _find_checkpoint(self.trainer_app.config, self.checkpoint_mode)
        if path is None:
            # Callers (newGameClicked_) call _refresh_status() right after
            # this, which sets the same "no checkpoint" message itself - so
            # setting status_label here too would just get overwritten.
            self.info_label.setStringValue_("")
            self.model = None
            self.model_path = None
            self.board.interactive = False
            return
        if path == self.model_path and self.model is not None:
            return
        import torch
        from train import ChessNet

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = ChessNet(blocks=ckpt.get("blocks", 6), channels=ckpt.get("channels", 128))
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.model_path = path
        self.board.interactive = True
        step = ckpt.get("step", "?")
        self.info_label.setStringValue_(f"vs {path.name} (step {step})")
        self._refresh_status()

    # ------------------------------------------------------------ actions

    def newGameClicked_(self, sender):
        self.game_id += 1
        self._cancel_self_play_timer()
        self._load_model_if_needed()
        self.board.flipped = self.human_color == chess.BLACK
        self.board.reset_game()
        self._refresh_status()
        if self.eval_bars_enabled:
            self._update_eval_bars_async()
        if self.model is None:
            return
        if self.self_play_enabled:
            self.board.interactive = False
            self._schedule_self_play_move()
        elif self.board.board.turn != self.human_color:
            self._bot_move_async()

    def colorChanged_(self, sender):
        self.human_color = chess.WHITE if sender.selectedSegment() == 0 else chess.BLACK
        self.newGameClicked_(sender)

    def checkpointModeChanged_(self, sender):
        self.checkpoint_mode = "latest" if sender.selectedSegment() == 1 else "best"
        self.checkpoint_caption.setStringValue_(CHECKPOINT_MODE_CAPTIONS[self.checkpoint_mode])
        self.newGameClicked_(sender)

    def searchToggled_(self, sender):
        self.search_enabled = sender.state() == 1
        self.sims_field.setHidden_(not self.search_enabled)

    def simsChanged_(self, sender):
        try:
            sims = int(sender.stringValue())
        except ValueError:
            sims = DEFAULT_SIMS
        self.sims = max(1, sims)
        sender.setStringValue_(str(self.sims))

    def delayChanged_(self, sender):
        self.self_play_delay = sender.doubleValue()
        self.delay_value_label.setStringValue_(f"{self.self_play_delay:.1f}s")

    def tempChanged_(self, sender):
        self.temperature = sender.doubleValue()
        self.temp_value_label.setStringValue_(f"{self.temperature:.2f}")

    def selfPlayToggled_(self, sender):
        self.self_play_enabled = sender.state() == 1
        if self.self_play_enabled:
            self.board.interactive = False
            if self.model is not None and not self.board.board.is_game_over():
                self._schedule_self_play_move()
        else:
            self._cancel_self_play_timer()
            self.board.interactive = (
                self.board.board.turn == self.human_color
                and not self.board.board.is_game_over()
            )
        self._refresh_status()

    def evalBarToggled_(self, sender):
        self.eval_bars_enabled = sender.state() == 1
        for v in (self.bot_eval_label, self.sf_eval_label,
                  self.bot_eval_bar, self.sf_eval_bar):
            v.setHidden_(not self.eval_bars_enabled)
        self.view.eval_bars_visible = self.eval_bars_enabled
        self.view.layout_now()
        if self.eval_bars_enabled:
            self._update_eval_bars_async()

    # ------------------------------------------------------------ game flow

    def on_human_move(self):
        self._refresh_status()
        if self.eval_bars_enabled:
            self._update_eval_bars_async()
        if self.board.board.is_game_over():
            self._refresh_status()
            return
        self._bot_move_async()

    def _current_sims(self):
        # Read straight from the field rather than the cached self.sims -
        # NSTextField's action only fires on Enter/losing focus, so relying
        # on that being caught in time meant a move could start using a
        # stale value from before the user finished typing a new one.
        try:
            sims = int(self.sims_field.stringValue())
        except ValueError:
            sims = self.sims
        return max(1, sims)

    def _bot_move_async(self):
        if self.model is None:
            return
        self.board.interactive = False
        sims = self._current_sims() if self.search_enabled else None
        if sims is not None:
            self.status_label.setStringValue_(f"Bot is thinking… ({sims} sims)")
        else:
            self.status_label.setStringValue_("Bot is thinking…")
        # Snapshot everything the worker needs now, on the main thread -
        # self.model can be swapped out (New Game, checkpoint switch) while
        # the worker is mid-computation, and it should keep using whatever
        # was true when the move was dispatched, not whatever's current by
        # the time it happens to read it.
        model = self.model
        board_copy = self.board.board.copy()
        game_id = self.game_id
        threading.Thread(
            target=self._bot_move_worker,
            args=(model, board_copy, sims, game_id),
            daemon=True,
        ).start()

    def _bot_move_worker(self, model, board_copy, sims, game_id):
        import search_batched

        # Both paths go through search_batched.device_for(), which moves the
        # model to MPS (if available) once and remembers where - so whichever
        # path runs first decides the device, and the other must agree, or a
        # later call crashes on a CPU/MPS tensor mismatch.
        device = search_batched.device_for(model)
        used_search = self.search_enabled

        if used_search:
            root = search_batched.run_search(model, board_copy, sims, syzygy="")
            move, ranked = search_batched.choose(root, temperature=self.temperature)
            san = board_copy.san(move)
            value = -ranked[0][1].value
        else:
            move, san, value = self._think_policy_only(model, board_copy, device)

        AppHelper.callAfter(
            self._apply_bot_move, move, san, value, used_search, sims, game_id
        )

    def _think_policy_only(self, model, board, device):
        """Same move selection as play.py's think(), but explicit about the
        device so it stays consistent with search_batched's (see above)."""
        import torch
        import torch.nn.functional as F
        from play import encode_board

        with torch.no_grad():
            logits, value = model(encode_board(board).to(device))
            logits = logits[0]

            by_index = {}
            for m in board.legal_moves:
                idx = m.from_square * 64 + m.to_square
                if idx not in by_index or m.promotion == chess.QUEEN:
                    by_index[idx] = m
            indices = list(by_index.keys())
            scores = logits[indices]

            if self.temperature <= 0:
                pick = scores.argmax().item()
            else:
                probs = F.softmax(scores / self.temperature, dim=0)
                pick = torch.multinomial(probs, 1).item()
            chosen = by_index[indices[pick]]

            return chosen, board.san(chosen), value.item()

    def _apply_bot_move(self, move, san, value, used_search, sims, game_id):
        if game_id != self.game_id:
            return  # New Game / checkpoint switch happened while this was in flight
        self.board.push_bot_move(move)
        if not self.self_play_enabled:
            self.board.interactive = True
        suffix = f", {sims} sims" if used_search else ""
        self.info_label.setStringValue_(f"bot played {san}  (eval {value:+.2f}{suffix})")
        self._refresh_status()
        if self.eval_bars_enabled:
            self._update_eval_bars_async()
        if self.self_play_enabled and not self.board.board.is_game_over():
            self._schedule_self_play_move()

    # ------------------------------------------------------------ self-play

    def _schedule_self_play_move(self):
        self._cancel_self_play_timer()
        self.self_play_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                self.self_play_delay, self, "selfPlayTick:", None, False
            )
        )

    def selfPlayTick_(self, timer):
        self.self_play_timer = None
        if not self.self_play_enabled or self.board.board.is_game_over():
            return
        self._bot_move_async()

    def _cancel_self_play_timer(self):
        if self.self_play_timer is not None:
            self.self_play_timer.invalidate()
            self.self_play_timer = None

    def _refresh_status(self):
        board = self.board.board
        if board.is_game_over():
            outcome = board.outcome()
            self.status_label.setStringValue_(f"Game over: {outcome.termination.name}")
            self.board.interactive = False
        elif self.model is None:
            label = "latest" if self.checkpoint_mode == "latest" else "best"
            self.status_label.setStringValue_(f"No {label} checkpoint yet")
        elif self.self_play_enabled:
            self.status_label.setStringValue_("Watching bot vs. bot…")
        elif board.turn == self.human_color:
            self.status_label.setStringValue_("Your move")
        else:
            self.status_label.setStringValue_("Bot is thinking…")

    # ------------------------------------------------------------ eval bars

    def _update_eval_bars_async(self):
        if self.model is None or self.board.board.is_game_over():
            return
        model = self.model
        board_copy = self.board.board.copy()
        game_id = self.game_id
        threading.Thread(
            target=self._eval_bars_worker,
            args=(model, board_copy, game_id),
            daemon=True,
        ).start()

    def _eval_bars_worker(self, model, board, game_id):
        import search_batched

        device = search_batched.device_for(model)
        bot_white_cp = self._bot_eval_white_cp(model, board, device)

        sf_white_cp = None
        if stockfish_eval.available():
            try:
                sf_white_cp = stockfish_eval.white_cp(board)
            except Exception:
                sf_white_cp = None

        AppHelper.callAfter(self._apply_eval_bars, bot_white_cp, sf_white_cp, game_id)

    def _bot_eval_white_cp(self, model, board, device):
        import torch
        from play import encode_board

        with torch.no_grad():
            _, value = model(encode_board(board).to(device))
        return bot_value_to_white_cp(value.item(), board.turn == chess.WHITE)

    def _apply_eval_bars(self, bot_white_cp, sf_white_cp, game_id):
        if game_id != self.game_id:
            return
        self.bot_eval_bar.set_value(
            cp_to_white_fraction(bot_white_cp), f"{bot_white_cp/100:+.1f}"
        )
        if sf_white_cp is None:
            self.sf_eval_bar.set_value(0.5, "N/A")
        else:
            self.sf_eval_bar.set_value(
                cp_to_white_fraction(sf_white_cp), f"{sf_white_cp/100:+.1f}"
            )
