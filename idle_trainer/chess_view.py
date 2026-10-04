"""A custom-drawn, click-to-move chess board for the Play tab.

Board and pieces are rendered via python-chess's bundled "cburnett" SVG
piece set (chess.svg.board) - the same set lichess uses - rather than
hand-drawn shapes, so they look like real chess pieces instead of an
approximation. chess.svg.board also highlights the last move's from/to
squares itself, in the same olive tone lichess uses.

Square-index math (_metrics/_square_to_rect/_point_to_square) intentionally
mirrors chess.svg's own layout formula exactly (see svg.py: x = file_index
or 7-file_index, y = 7-rank_index or rank_index, depending on orientation)
so the legal-move dots and click hit-testing always land exactly on top of
the squares the SVG image actually drew them at.
"""

from pathlib import Path

import objc
import chess
import chess.svg
from AppKit import NSBezierPath, NSColor, NSImage, NSSound, NSView
from Foundation import NSData, NSMakeRect, NSPoint

DEST_DOT = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.15, 0.15, 0.35)
SELECTED_FILL = "#ffff0055"

MOVE_SOUND_PATH = Path(__file__).resolve().parent / "assets" / "move.mp3"


def _svg_to_nsimage(svg_text):
    data_bytes = svg_text.encode("utf-8")
    ns_data = NSData.dataWithBytes_length_(data_bytes, len(data_bytes))
    image = NSImage.alloc().initWithData_(ns_data)
    return image if image is not None and image.isValid() else None


def _play_move_sound():
    # A fresh NSSound per play, rather than reusing one instance - two moves
    # landing close together (e.g. a fast bot reply) should both be heard,
    # not have the second cut the first off.
    if not MOVE_SOUND_PATH.exists():
        return
    sound = NSSound.alloc().initWithContentsOfFile_byReference_(
        str(MOVE_SOUND_PATH), True
    )
    if sound is not None:
        sound.play()


class ChessBoardView(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(ChessBoardView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.board = chess.Board()
        self.flipped = False
        self.selected_square = None
        self.legal_dest_squares = set()
        self.last_move = None
        self.interactive = True
        self.on_human_move = None  # callable(), invoked after a legal human move
        self._cached_image = None
        self._cached_key = None
        return self

    # ------------------------------------------------------------ geometry

    def _metrics(self):
        w, h = self.bounds().size.width, self.bounds().size.height
        size = min(w, h) / 8.0
        ox = (w - size * 8) / 2.0
        oy = (h - size * 8) / 2.0
        return size, ox, oy

    def _square_to_rect(self, square):
        size, ox, oy = self._metrics()
        file_, rank = chess.square_file(square), chess.square_rank(square)
        if not self.flipped:
            vcol, vrow = file_, 7 - rank
        else:
            vcol, vrow = 7 - file_, rank
        x = ox + vcol * size
        y = oy + (7 - vrow) * size
        return NSMakeRect(x, y, size, size)

    def _point_to_square(self, point):
        size, ox, oy = self._metrics()
        if size <= 0:
            return None
        col = int((point.x - ox) // size)
        row_from_top = 7 - int((point.y - oy) // size)
        if not (0 <= col <= 7 and 0 <= row_from_top <= 7):
            return None
        if not self.flipped:
            file_, rank = col, 7 - row_from_top
        else:
            file_, rank = 7 - col, row_from_top
        return chess.square(file_, rank)

    # ------------------------------------------------------------ drawing

    def isFlipped(self):
        return False

    def drawRect_(self, rect):
        NSColor.windowBackgroundColor().setFill()
        NSBezierPath.fillRect_(self.bounds())

        size, ox, oy = self._metrics()
        if size <= 0:
            return

        pixel_size = max(1, int(size * 8))
        key = (
            id(self.board), len(self.board.move_stack), self.flipped,
            self.last_move, self.selected_square, pixel_size,
        )
        if key == self._cached_key:
            image = self._cached_image
        else:
            fill = {}
            if self.selected_square is not None:
                fill[self.selected_square] = SELECTED_FILL
            svg_text = chess.svg.board(
                self.board,
                orientation=chess.BLACK if self.flipped else chess.WHITE,
                lastmove=self.last_move,
                fill=fill,
                coordinates=False,
                size=pixel_size,
            )
            image = _svg_to_nsimage(svg_text)
            self._cached_image = image
            self._cached_key = key

        if image is not None:
            image.drawInRect_(NSMakeRect(ox, oy, size * 8, size * 8))

        for square in self.legal_dest_squares:
            r = self._square_to_rect(square)
            pad = r.size.width * 0.35
            dot = NSMakeRect(
                r.origin.x + pad, r.origin.y + pad,
                r.size.width - 2 * pad, r.size.height - 2 * pad,
            )
            DEST_DOT.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(dot).fill()

    # ------------------------------------------------------------ input

    def mouseDown_(self, event):
        if not self.interactive or self.board.is_game_over():
            return
        point = self.convertPoint_fromView_(event.locationInWindow(), None)
        square = self._point_to_square(NSPoint(point.x, point.y))
        if square is None:
            return

        if self.selected_square is not None and square in self.legal_dest_squares:
            self._push_move(self.selected_square, square)
            self.selected_square = None
            self.legal_dest_squares = set()
            self.setNeedsDisplay_(True)
            if self.on_human_move is not None:
                self.on_human_move()
            return

        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_dest_squares = {
                m.to_square for m in self.board.legal_moves if m.from_square == square
            }
        else:
            self.selected_square = None
            self.legal_dest_squares = set()
        self.setNeedsDisplay_(True)

    def _push_move(self, from_square, to_square):
        candidates = [
            m for m in self.board.legal_moves
            if m.from_square == from_square and m.to_square == to_square
        ]
        move = next((m for m in candidates if m.promotion == chess.QUEEN), None) or candidates[0]
        self.board.push(move)
        self.last_move = move
        _play_move_sound()

    # ------------------------------------------------------------ control

    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_dest_squares = set()
        self.last_move = None
        self.interactive = True
        self.setNeedsDisplay_(True)

    def push_bot_move(self, move):
        self.board.push(move)
        self.last_move = move
        self.setNeedsDisplay_(True)
        _play_move_sound()
