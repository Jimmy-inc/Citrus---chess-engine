"""A thin vertical eval bar (white fill from the bottom = White's share of
the position, like lichess's), plus the bot-value <-> centipawn conversion.

The net's value head is a tanh output "from the perspective of whoever is
to move" (see train.py) - not centipawns. There's no ground truth to
calibrate an exact conversion against, so this uses the standard
tanh/centipawn relationship engines like this one are implicitly trained
under (value ~= tanh(cp / K)) and inverts it: cp = K * atanh(value).
K = 600 was picked so the range "looks like" familiar centipawn swings
(0.5 -> ~330cp, 0.9 -> ~880cp) - it's an approximation, not a calibration.
"""

import math

import objc
from AppKit import NSBezierPath, NSColor, NSFont, NSAttributedString, NSView
from AppKit import NSFontAttributeName, NSForegroundColorAttributeName
from Foundation import NSMakeRect

CP_SCALE = 600.0


def bot_value_to_white_cp(value_from_side_to_move, side_to_move_is_white):
    """value_from_side_to_move: the net's raw output, in (-1, 1)."""
    v = max(-0.999, min(0.999, value_from_side_to_move))
    cp = CP_SCALE * math.atanh(v)
    return cp if side_to_move_is_white else -cp


def cp_to_white_fraction(cp):
    """Standard win-probability-style squashing for the bar fill."""
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


class EvalBarView(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(EvalBarView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.fraction_white = 0.5
        self.label_text = "—"
        self.title_text = ""
        return self

    def set_value(self, fraction_white, label_text):
        self.fraction_white = max(0.0, min(1.0, fraction_white))
        self.label_text = label_text
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height

        NSColor.blackColor().setFill()
        NSBezierPath.fillRect_(bounds)

        white_h = h * self.fraction_white
        NSColor.whiteColor().setFill()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, w, white_h))

        font = NSFont.boldSystemFontOfSize_(10)
        # Put the label in whichever half has more contrast room, and use
        # the opposite color from that half's fill so it stays legible.
        if self.fraction_white > 0.5:
            label_y = 4
            color = NSColor.blackColor()
        else:
            label_y = h - 16
            color = NSColor.whiteColor()
        attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color}
        NSAttributedString.alloc().initWithString_attributes_(
            self.label_text, attrs
        ).drawInRect_(NSMakeRect(0, label_y, w, 14))
