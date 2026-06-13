"""
chess_ui.py — tkinter интерфейс для ChessGame.
Логика полностью в chess_game.py, здесь только отображение.
"""
import tkinter
import tkinter.messagebox
from chessAI.chess import (ChessGame, Move, DuckMove,
                           Pawn, Knight, Bishop, Rook, Queen, King)


class ChessUI:
    def __init__(self, root):
        self.root   = root
        self.game   = ChessGame()
        self.canvas = tkinter.Canvas(root, width=640, height=640)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<B1-Motion>",       self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.selected_piece = None
        self.drag_offset    = (0, 0)
        self.piece_ids      = {}   # piece → canvas id

        self._draw_board()
        self._draw_pieces()

    # ── Отрисовка ────────────────────────────────

    def _draw_board(self):
        for i in range(8):
            for j in range(8):
                color = "#EEE" if (i + j) % 2 == 0 else "#777"
                self.canvas.create_rectangle(
                    i*80, j*80, (i+1)*80, (j+1)*80, fill=color)

    def _draw_pieces(self):
        for piece in self.game.pieces:
            cid = self.canvas.create_text(
                piece.row * 80 + 40,
                piece.col * 80 + 40,
                text=piece.rep,
                font=("Arial", 50)
            )
            self.piece_ids[piece] = cid
        # Рисуем утку
        duck = self.game.duck
        if duck.col is not None:
            self.canvas.create_text(
                duck.row * 80 + 40,
                duck.col * 80 + 40,
                text="🦆", font=("Arial", 40), tags="duck"
            )
        # Если фаза утки — подсвечиваем доступные клетки
        if self.game.duck_phase:
            self._highlight_duck_squares()

    def _highlight_duck_squares(self):
        """Подсвечиваем клетки куда можно поставить утку."""
        for dm in self.game.duck_actions():
            self.canvas.create_rectangle(
                dm.row * 80 + 2, dm.col * 80 + 2,
                dm.row * 80 + 78, dm.col * 80 + 78,
                outline="#FFD700", width=3, tags="highlight"
            )

    def _redraw(self):
        """Полная перерисовка после хода."""
        self.canvas.delete("all")
        self._draw_board()
        self.piece_ids = {}
        self._draw_pieces()

    def _move_canvas_piece(self, piece):
        cid = self.piece_ids.get(piece)
        if cid:
            x = piece.row * 80 + 40
            y = piece.col * 80 + 40
            self.canvas.coords(cid, x, y)

    # ── Мышь ────────────────────────────────────

    def on_click(self, event):
        # В фазе утки — клик сразу ставит утку
        if self.game.duck_phase:
            col = event.y // 80
            row = event.x // 80
            if 0 <= col < 8 and 0 <= row < 8:
                dm = DuckMove(col, row)
                if dm.to_index() in self.game.legal_actions():
                    self.game.step(dm.to_index())
                    self._redraw()
                    if self.game.done:
                        result = self.game.winner()
                        msg = "Ничья!" if result == "draw" else f"Победили {'Белые' if result == 'w' else 'Чёрные'}!"
                        tkinter.messagebox.showinfo("Конец игры", msg)
            return

        item = self.canvas.find_closest(event.x, event.y)
        for p, cid in self.piece_ids.items():
            if cid == item[0] and p.color == self.game.turn:
                self.selected_piece = p
                x = p.row * 80 + 40
                y = p.col * 80 + 40
                self.drag_offset = (x - event.x, y - event.y)
                self.canvas.tag_raise(cid)
                return

    def on_motion(self, event):
        if self.game.duck_phase or not self.selected_piece:
            return
        dx, dy = self.drag_offset
        cid = self.piece_ids.get(self.selected_piece)
        if cid:
            self.canvas.coords(cid, event.x + dx, event.y + dy)

    def on_release(self, event):
        if not self.selected_piece:
            return

        to_col = event.y // 80
        to_row = event.x // 80

        if 0 <= to_col < 8 and 0 <= to_row < 8:
            legal = self.game.legal_moves()

            # Проверяем — есть ли ходы с превращением на эту клетку
            promo_moves = [m for m in legal
                           if (m.from_col == self.selected_piece.col
                               and m.from_row == self.selected_piece.row
                               and m.to_col == to_col
                               and m.to_row == to_row
                               and m.promotion is not None)]

            normal_moves = [m for m in legal
                            if (m.from_col == self.selected_piece.col
                                and m.from_row == self.selected_piece.row
                                and m.to_col == to_col
                                and m.to_row == to_row
                                and m.promotion is None)]

            if promo_moves:
                chosen = self._ask_promotion()
                target = next((m for m in promo_moves
                               if m.promotion == chosen), promo_moves[0])
                self.game.step(target.to_index() + 64)
                self._redraw()
            elif normal_moves:
                self.game.step(normal_moves[0].to_index() + 64)
                self._redraw()
            else:
                self._move_canvas_piece(self.selected_piece)

            # После хода фигурой — ждём хода утки (не показываем конец)
            if self.game.done and not self.game.duck_phase:
                result = self.game.winner()
                msg = "Ничья!" if result == "draw" else f"Победили {'Белые' if result == 'w' else 'Чёрные'}!"
                tkinter.messagebox.showinfo("Конец игры", msg)
        else:
            self._move_canvas_piece(self.selected_piece)

        self.selected_piece = None

    def _ask_promotion(self):
        """Диалог выбора фигуры при превращении пешки."""
        from chessAI.chess import Queen, Rook, Bishop, Knight
        dialog = tkinter.Toplevel(self.root)
        dialog.title("Выберите фигуру")
        dialog.resizable(False, False)
        dialog.grab_set()

        chosen = [Queen]  # по умолчанию ферзь

        options = [
            ("Ферзь ♕",  Queen),
            ("Ладья ♖",  Rook),
            ("Слон ♗",   Bishop),
            ("Конь ♘",   Knight),
        ]

        tkinter.Label(dialog, text="В кого превратить пешку?",
                      font=("Arial", 14), pady=10).pack()

        for label, piece_class in options:
            pc = piece_class  # захват в замыкании
            def make_handler(cls):
                def handler():
                    chosen[0] = cls
                    dialog.destroy()
                return handler
            tkinter.Button(dialog, text=label, font=("Arial", 16),
                           width=12, pady=5,
                           command=make_handler(pc)).pack(pady=2)

        dialog.wait_window()
        return chosen[0]


# ── Запуск ───────────────────────────────────────

if __name__ == "__main__":
    root = tkinter.Tk()
    root.title("Duck Chess")
    ChessUI(root)
    root.mainloop()