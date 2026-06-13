# play.py — hra proti natrénovanému agentovi Duck Chess.

import sys
import os
import threading
import tkinter
import tkinter.messagebox
import numpy as np
import torch

from chessAI.chess import ChessGame, DuckMove, Knight, Bishop, Rook, Queen
from chessAI.neural_net import DuckChessNet
from chessAI.mcts import MCTS


# ---------------------------------------------
#  Argumenty príkazového riadku
# ---------------------------------------------

def parse_args():
    args = {
        "checkpoint": "checkpoints/latest.pt",
        "sims":       200,
        "color":      "w",   # farba hráča: 'w' alebo 'b'
    }
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.endswith(".pt"):
            args["checkpoint"] = arg
        elif arg == "--sims" and i + 1 < len(sys.argv):
            args["sims"] = int(sys.argv[i + 1]); i += 1
        elif arg == "--color" and i + 1 < len(sys.argv):
            args["color"] = sys.argv[i + 1]; i += 1
        i += 1
    return args


# ---------------------------------------------
#  UI s podporou agenta
# ---------------------------------------------

class PlayUI:
    def __init__(self, root, network, device, player_color, num_sims):
        self.root         = root
        self.game         = ChessGame()
        self.network      = network
        self.device       = device
        self.player_color = player_color   # farba človeka
        self.num_sims     = num_sims
        self.agent_thinking = False        # agent premýšľa - blokujeme UI

        self.mcts = MCTS(
            network,
            device,
            num_simulations = num_sims,
            c_puct          = 1.5,
            dirichlet_alpha = 0.03,   # takmer žiadny šum - agent hrá vážne
        )

        # --- Plátno dosky ---------------------------------------------
        self.canvas = tkinter.Canvas(root, width=640, height=640)
        self.canvas.pack()

        # --- Stavový riadok ---------------------------------------------
        self.status_var = tkinter.StringVar()
        self.status_var.set(self._status_text())
        status_bar = tkinter.Label(
            root, textvariable=self.status_var,
            font=("Arial", 13), pady=6, bg="#222", fg="#eee"
        )
        status_bar.pack(fill=tkinter.X)

        # --- Tlačidlo novej hry ---------------------------------------------
        tkinter.Button(
            root, text="Nová hra", font=("Arial", 12),
            command=self._new_game, pady=4
        ).pack(fill=tkinter.X)

        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<B1-Motion>",       self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.selected_piece = None
        self.drag_offset    = (0, 0)
        self.piece_ids      = {}

        self._redraw()

        # Ak agent ťahá ako prvý (hráč hrá čiernymi) - spustíme agenta
        if self.player_color == "b":
            self.root.after(500, self._agent_move_async)

    # --- Stav ---------------------------------------------

    def _status_text(self):
        if self.game.done:
            r = self.game.winner()
            if r == "draw":
                return "Remíza!"
            winner_name = "Biele" if r == "w" else "Čierne"
            you_won = (r == self.player_color)
            return f"{'Vyhrali ste!' if you_won else 'Agent vyhral!'} ({winner_name})"

        whose = self.game.turn
        if self.game.duck_phase:
            who = "Vy" if whose == self.player_color else "Agent"
            return f"{who}: postavte kačicu"
        if whose == self.player_color:
            return f"Váš ťah ({'Biele' if whose == 'w' else 'Čierne'})"
        else:
            return f"Agent premýšľa... ({'Biele' if whose == 'w' else 'Čierne'})"

    def _update_status(self):
        self.status_var.set(self._status_text())

    # --- Kreslenie ---------------------------------------------

    def _draw_board(self):
        for i in range(8):
            for j in range(8):
                color = "#EEE" if (i + j) % 2 == 0 else "#777"
                self.canvas.create_rectangle(
                    i*80, j*80, (i+1)*80, (j+1)*80, fill=color
                )

    def _draw_pieces(self):
        for piece in self.game.pieces:
            cid = self.canvas.create_text(
                piece.row * 80 + 40, piece.col * 80 + 40,
                text=piece.rep, font=("Arial", 50)
            )
            self.piece_ids[piece] = cid

        duck = self.game.duck
        if duck.col is not None:
            self.canvas.create_text(
                duck.row * 80 + 40, duck.col * 80 + 40,
                text="🦆", font=("Arial", 40), tags="duck"
            )

        # Zvýraznenie pre fázu kačice (len ak je ťah hráča)
        if self.game.duck_phase and self.game.turn == self.player_color:
            for dm in self.game.duck_actions():
                self.canvas.create_rectangle(
                    dm.row * 80 + 2, dm.col * 80 + 2,
                    dm.row * 80 + 78, dm.col * 80 + 78,
                    outline="#FFD700", width=3, tags="highlight"
                )

    def _redraw(self):
        self.canvas.delete("all")
        self._draw_board()
        self.piece_ids = {}
        self._draw_pieces()
        self._update_status()

    def _move_canvas_piece(self, piece):
        cid = self.piece_ids.get(piece)
        if cid:
            self.canvas.coords(cid, piece.row * 80 + 40, piece.col * 80 + 40)

    # --- Ťah agenta ---------------------------------------------

    def _agent_move_async(self):
        # Spustí ťah agenta v samostatnom vlákne aby UI nezamrzlo
        if self.game.done or self.game.turn == self.player_color:
            return
        self.agent_thinking = True
        self._update_status()
        threading.Thread(target=self._agent_move_thread, daemon=True).start()

    def _agent_move_thread(self):
        # Agent premýšľa (v pozadí)
        policy = self.mcts.run(self.game, temperature=0)  # vždy chamtivo
        action = int(np.argmax(policy))
        # Vrátime sa do hlavného vlákna pre aktualizáciu UI
        self.root.after(0, lambda: self._apply_agent_action(action))

    def _apply_agent_action(self, action):
        # Aplikujeme ťah agenta (v hlavnom vlákne)
        self.game.step(action)
        self.agent_thinking = False
        self._redraw()

        if self.game.done:
            self._show_result()
            return

        # Ak po ťahu figúrou nastáva fáza kačice - agent tiež postaví kačicu
        if self.game.duck_phase and self.game.turn != self.player_color:
            self.root.after(100, self._agent_move_async)
        # Ak prešiel ťah na hráča — čakáme na jeho akciu

    # --- Myš (ťahy hráča) ---------------------------------------------

    def on_click(self, event):
        # Blokujeme kým agent premýšľa
        if self.agent_thinking:
            return
        # Nie je ťah hráča
        if self.game.turn != self.player_color:
            return

        # Fáza kačice
        if self.game.duck_phase:
            col, row = event.y // 80, event.x // 80
            if 0 <= col < 8 and 0 <= row < 8:
                dm = DuckMove(col, row)
                if dm.to_index() in self.game.legal_actions():
                    self.game.step(dm.to_index())
                    self._redraw()
                    if self.game.done:
                        self._show_result()
                    else:
                        # Teraz je ťah agenta
                        self.root.after(300, self._agent_move_async)
            return

        # Výber figúry
        item = self.canvas.find_closest(event.x, event.y)
        for p, cid in self.piece_ids.items():
            if cid == item[0] and p.color == self.player_color:
                self.selected_piece = p
                x, y = p.row * 80 + 40, p.col * 80 + 40
                self.drag_offset = (x - event.x, y - event.y)
                self.canvas.tag_raise(cid)
                return

    def on_motion(self, event):
        if self.agent_thinking or self.game.duck_phase:
            return
        if not self.selected_piece:
            return
        dx, dy = self.drag_offset
        cid = self.piece_ids.get(self.selected_piece)
        if cid:
            self.canvas.coords(cid, event.x + dx, event.y + dy)

    def on_release(self, event):
        if self.agent_thinking or not self.selected_piece:
            return

        to_col, to_row = event.y // 80, event.x // 80

        if 0 <= to_col < 8 and 0 <= to_row < 8:
            legal = self.game.legal_moves()

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
                self._after_player_move()
            elif normal_moves:
                self.game.step(normal_moves[0].to_index() + 64)
                self._after_player_move()
            else:
                self._move_canvas_piece(self.selected_piece)
        else:
            self._move_canvas_piece(self.selected_piece)

        self.selected_piece = None

    def _after_player_move(self):
        # Po ťahu figúrou hráčom
        self._redraw()
        if self.game.done:
            self._show_result()
            return
        # Fáza kačice - hráč postaví sám (on_click to spracuje)
        # Agent bude zavolaný po tom, ako hráč postaví kačicu

    # ── Dialógy ──────────────────────────────────────

    def _ask_promotion(self):
        dialog = tkinter.Toplevel(self.root)
        dialog.title("Premena pešiaka")
        dialog.resizable(False, False)
        dialog.grab_set()
        chosen = [Queen]
        options = [("Dáma ♕", Queen), ("Veža ♖", Rook),
                   ("Strelec ♗", Bishop), ("Jazdec ♘", Knight)]
        tkinter.Label(dialog, text="Na čo premeniť?",
                      font=("Arial", 14), pady=10).pack()
        for label, cls in options:
            def make_handler(c):
                def h(): chosen[0] = c; dialog.destroy()
                return h
            tkinter.Button(dialog, text=label, font=("Arial", 16),
                           width=12, pady=5,
                           command=make_handler(cls)).pack(pady=2)
        dialog.wait_window()
        return chosen[0]

    def _show_result(self):
        msg = self._status_text()
        self._update_status()
        tkinter.messagebox.showinfo("Koniec hry", msg)

    def _new_game(self):
        self.game         = ChessGame()
        self.agent_thinking = False
        self._redraw()
        if self.player_color == "b":
            self.root.after(500, self._agent_move_async)


# ---------------------------------------------
#  Načítanie checkpointu a spustenie
# ---------------------------------------------

def load_network(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint nebol nájdený: {checkpoint_path}")
        print("Najprv spusti train.py - vytvorí checkpoints/latest.pt")
        sys.exit(1)

    ckpt    = torch.load(checkpoint_path, map_location=device)
    config  = ckpt.get("config", {})
    network = DuckChessNet(
        num_res_blocks = config.get("num_res_blocks", 6),
        channels       = config.get("channels", 128),
    ).to(device)
    network.load_state_dict(ckpt["network"])
    network.eval()

    iteration = ckpt.get("iteration", "?")
    print(f"Načítaný checkpoint: {checkpoint_path}  (generácia {iteration})")
    return network


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}")
    print(f"Simulácií MCTS: {args['sims']}")
    print(f"Hráš: {'Bielymi' if args['color'] == 'w' else 'Čiernymi'}")

    network = load_network(args["checkpoint"], device)

    root = tkinter.Tk()
    root.title("Duck Chess - Človek vs Agent")
    PlayUI(
        root,
        network      = network,
        device       = device,
        player_color = args["color"],
        num_sims     = args["sims"],
    )
    root.mainloop()