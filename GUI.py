# GUI.py - tkinter rozhranie pre Duck Chess (hra dvoch hráčov).
# Herná logika je úplne v chess.py - tu je iba zobrazenie a obsluha myši.
#
# Spustenie:
#     python GUI.py

import tkinter
import tkinter.messagebox
from chess import (ChessGame, Move, DuckMove,
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
        self.piece_ids      = {}   # figúra -> id objektu na plátne

        self._draw_board()
        self._draw_pieces()

    # --- Kreslenie ---------------------------------------------

    def _draw_board(self):
        # Nakreslíme šachovnicu - striedame svetlé a tmavé polia
        for i in range(8):
            for j in range(8):
                color = "#EEE" if (i + j) % 2 == 0 else "#777"
                self.canvas.create_rectangle(
                    i*80, j*80, (i+1)*80, (j+1)*80, fill=color)

    def _draw_pieces(self):
        # Nakreslíme všetky figúry na ich aktuálnych pozíciách
        for piece in self.game.pieces:
            cid = self.canvas.create_text(
                piece.row * 80 + 40,
                piece.col * 80 + 40,
                text=piece.rep,
                font=("Arial", 50)
            )
            self.piece_ids[piece] = cid

        # Nakreslíme kačicu ak už bola postavená
        duck = self.game.duck
        if duck.col is not None:
            self.canvas.create_text(
                duck.row * 80 + 40,
                duck.col * 80 + 40,
                text="🦆", font=("Arial", 40), tags="duck"
            )

        # Ak je fáza kačice — zvýrazníme dostupné polia
        if self.game.duck_phase:
            self._highlight_duck_squares()

    def _highlight_duck_squares(self):
        # Zvýrazníme zlatým rámčekom polia, kam možno kačicu umiestniť
        for dm in self.game.duck_actions():
            self.canvas.create_rectangle(
                dm.row * 80 + 2, dm.col * 80 + 2,
                dm.row * 80 + 78, dm.col * 80 + 78,
                outline="#FFD700", width=3, tags="highlight"
            )

    def _redraw(self):
        # Úplné prekreslenie po každom ťahu
        self.canvas.delete("all")
        self._draw_board()
        self.piece_ids = {}
        self._draw_pieces()

    def _move_canvas_piece(self, piece):
        # Vrátime figúru na jej pôvodné miesto (neplatný ťah - pustenie mimo dosky)
        cid = self.piece_ids.get(piece)
        if cid:
            x = piece.row * 80 + 40
            y = piece.col * 80 + 40
            self.canvas.coords(cid, x, y)

    # --- Myš ---------------------------------------------

    def on_click(self, event):
        # Fáza kačice - klik ihneď umiestni kačicu na zvolené pole
        if self.game.duck_phase:
            col = event.y // 80
            row = event.x // 80
            if 0 <= col < 8 and 0 <= row < 8:
                dm = DuckMove(col, row)
                if dm.to_index() in self.game.legal_actions():
                    self.game.step(dm.to_index())
                    self._redraw()
                    if self.game.done:
                        self._show_result()
            return

        # Bežná fáza - vyberieme figúru pod kurzorom
        item = self.canvas.find_closest(event.x, event.y)
        for p, cid in self.piece_ids.items():
            if cid == item[0] and p.color == self.game.turn:
                self.selected_piece = p
                x = p.row * 80 + 40
                y = p.col * 80 + 40
                self.drag_offset = (x - event.x, y - event.y)
                # Vybraná figúra sa zobrazí nad ostatnými
                self.canvas.tag_raise(cid)
                return

    def on_motion(self, event):
        # Presúvame figúru myšou - aktualizujeme jej polohu na plátne
        if self.game.duck_phase or not self.selected_piece:
            return
        dx, dy = self.drag_offset
        cid = self.piece_ids.get(self.selected_piece)
        if cid:
            self.canvas.coords(cid, event.x + dx, event.y + dy)

    def on_release(self, event):
        # Pustili sme figúru - overíme či je ťah legálny a vykonáme ho
        if not self.selected_piece:
            return

        to_col = event.y // 80
        to_row = event.x // 80

        if 0 <= to_col < 8 and 0 <= to_row < 8:
            legal = self.game.legal_moves()

            # Ťahy s premenou pešiaka na cieľové pole
            promo_moves = [m for m in legal
                           if (m.from_col == self.selected_piece.col
                               and m.from_row == self.selected_piece.row
                               and m.to_col == to_col
                               and m.to_row == to_row
                               and m.promotion is not None)]

            # Bežné ťahy na cieľové pole (bez premeny)
            normal_moves = [m for m in legal
                            if (m.from_col == self.selected_piece.col
                                and m.from_row == self.selected_piece.row
                                and m.to_col == to_col
                                and m.to_row == to_row
                                and m.promotion is None)]

            if promo_moves:
                # Pýtame sa hráča na čo premeniť pešiaka
                chosen = self._ask_promotion()
                target = next((m for m in promo_moves
                               if m.promotion == chosen), promo_moves[0])
                self.game.step(target.to_index() + 64)
                self._redraw()
            elif normal_moves:
                self.game.step(normal_moves[0].to_index() + 64)
                self._redraw()
            else:
                # Neplatný ťah - vrátime figúru na pôvodné miesto
                self._move_canvas_piece(self.selected_piece)

            # Koniec hry zobrazíme až po ťahu kačice (nie hneď po zajatí kráľa)
            if self.game.done and not self.game.duck_phase:
                self._show_result()
        else:
            # Pustili sme figúru mimo dosky - vrátime ju späť
            self._move_canvas_piece(self.selected_piece)

        self.selected_piece = None

    # --- Dialógy ---------------------------------------------

    def _ask_promotion(self):
        # Dialógové okno pre výber figúry pri premene pešiaka
        dialog = tkinter.Toplevel(self.root)
        dialog.title("Premena pešiaka")
        dialog.resizable(False, False)
        dialog.grab_set()   # blokujeme hlavné okno kým hráč nevyberie

        chosen = [Queen]   # predvolená voľba - dáma

        options = [
            ("Dáma ♕",    Queen),
            ("Veža ♖",    Rook),
            ("Strelec ♗", Bishop),
            ("Jazdec ♘",  Knight),
        ]

        tkinter.Label(dialog, text="Na čo premeniť pešiaka?",
                      font=("Arial", 14), pady=10).pack()

        for label, piece_class in options:
            def make_handler(cls):
                def handler():
                    chosen[0] = cls
                    dialog.destroy()
                return handler
            tkinter.Button(dialog, text=label, font=("Arial", 16),
                           width=12, pady=5,
                           command=make_handler(piece_class)).pack(pady=2)

        dialog.wait_window()
        return chosen[0]

    def _show_result(self):
        # Zobrazíme výsledok hry v dialógovom okne
        result = self.game.winner()
        if result == "draw":
            msg = "Remíza!"
        elif result == "w":
            msg = "Vyhrali Biele! ♔"
        else:
            msg = "Vyhrali Čierne! ♚"
        tkinter.messagebox.showinfo("Koniec hry", msg)


# --- Spustenie ---------------------------------------------

if __name__ == "__main__":
    root = tkinter.Tk()
    root.title("Duck Chess")
    ChessUI(root)
    root.mainloop()
