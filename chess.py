# chess.py - logika Duck Chess bez grafiky.
# Pripravené na použitie s MCTS a neurónovou sieťou.
#
# Kľúčový rozdiel oproti bežným šachom:
#   - Šach a mat NEEXISTUJÚ. Hra sa končí zajatím kráľa.
#   - Po každom ťahu figúrou je hráč povinný premiestniť kačicu.
#   - Kačica blokuje pohyb figúr ako bežná figúra.

import numpy as np
import copy


# ---------------------------------------------
#  Figúry
# ---------------------------------------------

class Piece:
    def __init__(self, color, col, row):
        self.color = color
        self.col   = col
        self.row   = row

class Pawn(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.start = True
        self.rep = "♟" if color == "b" else "♙"

class Knight(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.rep = "♞" if color == "b" else "♘"

class Bishop(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.rep = "♝" if color == "b" else "♗"

class Rook(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.has_moved = False
        self.rep = "♜" if color == "b" else "♖"

class Queen(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.rep = "♛" if color == "b" else "♕"

class King(Piece):
    def __init__(self, color, col, row):
        super().__init__(color, col, row)
        self.has_moved = False
        self.rep = "♚" if color == "b" else "♔"

# Kačica - neutrálna figúra, ktorú treba premiestniť po každom ťahu
class Duck:
    def __init__(self, col=None, row=None):
        self.col = col   # None = kačica ešte nebola postavená (prvý ťah)
        self.row = row
        self.rep = "🦆"


# ---------------------------------------------
#  Reprezentácia ťahu
# ---------------------------------------------

# Jeden ťah: ktorá figúra, odkiaľ, kam, a na čo sa premeniť (ak ide o pešiaka)
class Move:
    # Indexy figúr pre premenu: 0=Dáma 1=Veža 2=Strelec 3=Jazdec
    PROMOTION_PIECES = [Queen, Rook, Bishop, Knight]

    def __init__(self, piece, to_col, to_row, promotion=None):
        self.piece     = piece
        self.from_col  = piece.col
        self.from_row  = piece.row
        self.to_col    = to_col
        self.to_row    = to_row
        self.promotion = promotion   # trieda figúry (Queen/Rook/Bishop/Knight) alebo None

    def to_index(self):
        # Jedinečné číslo ťahu:
        #   Bežné ťahy:       0..4095   (64×64)
        #   S premenou:       4096..4351 (64 polí × 4 figúry)
        if self.promotion is not None:
            promo_idx = self.PROMOTION_PIECES.index(self.promotion)
            to_sq = self.to_col * 8 + self.to_row
            return 4096 + to_sq * 4 + promo_idx
        from_sq = self.from_col * 8 + self.from_row
        to_sq   = self.to_col   * 8 + self.to_row
        return from_sq * 64 + to_sq

    @staticmethod
    def from_index(idx, game):
        # Obnov ťah z jeho čísla
        if idx >= 4096:
            # Premena pešiaka
            idx2      = idx - 4096
            promo_idx = idx2 % 4
            to_sq     = idx2 // 4
            to_col, to_row = divmod(to_sq, 8)
            promotion = Move.PROMOTION_PIECES[promo_idx]
            # Hľadáme pešiaka, ktorý sa môže dostať na (to_col, to_row)
            for p in game.pieces:
                if (type(p) == Pawn and p.color == game.turn
                        and abs(p.row - to_row) <= 1):
                    return Move(p, to_col, to_row, promotion)
            return None
        from_sq = idx // 64
        to_sq   = idx  % 64
        from_col, from_row = divmod(from_sq, 8)
        to_col,   to_row   = divmod(to_sq,   8)
        piece = game.piece_at(from_col, from_row)
        if piece is None:
            return None
        return Move(piece, to_col, to_row)

    def __repr__(self):
        cols = "abcdefgh"
        base = (f"{type(self.piece).__name__} "
                f"{cols[self.from_row]}{8 - self.from_col}"
                f"→{cols[self.to_row]}{8 - self.to_col}")
        if self.promotion:
            base += f"={self.promotion.__name__}"
        return base


# Premiestnenie kačice na pole (col, row)
class DuckMove:
    def __init__(self, col, row):
        self.col = col
        self.row = row

    def to_index(self):
        # Index 0..63
        return self.col * 8 + self.row

    @staticmethod
    def from_index(idx):
        col, row = divmod(idx, 8)
        return DuckMove(col, row)

    def __repr__(self):
        cols = "abcdefgh"
        return f"Kačica→{cols[self.row]}{8 - self.col}"


# ---------------------------------------------
#  Hlavná logika hry
# ---------------------------------------------

# Logika Duck Chess - bez tkinter, bez grafiky.
#
# Pravidlá Duck Chess (rozdiely oproti bežným šachom):
#   - Šach a mat NEEXISTUJÚ. Víťazstvo = zajatie kráľa súpera.
#   - Po každom ťahu figúrou treba premiestniť kačicu (duck_phase=True).
#   - Kačica blokuje ťahy ako bežná figúra, ale nemožno ju zobrať.
#
# Metódy pre RL:
#     legal_actions()   -> zoznam indexov legálnych ťahov
#     step(action)      -> vykoná ťah
#     get_observation() -> stav dosky ako numpy pole pre neurónovú sieť
#     clone()           -> hlboká kópia pre simulácie MCTS
#     is_terminal()     -> či sa hra skončila
#     winner()          -> 'w', 'b' alebo 'draw'
class ChessGame:

    def __init__(self):
        self._init_pieces()
        self.turn       = "w"
        self.lastTurne  = None     # pre branie en passant
        self.done       = False
        self.result     = None     # 'w', 'b', 'draw'
        self.move_count = 0        # počítadlo ťahov

        # Duck Chess
        self.duck       = Duck()   # kačica (col=None kým nebola postavená)
        self.duck_phase = False    # False = ťah figúrou, True = ťah kačicou

        # História pre remízy
        self.position_history = {}
        self.no_capture_moves = 0  # pravidlo 50 ťahov

    def _init_pieces(self):
        self.blackPice = {
            Rook("b",0,0), Rook("b",0,7),
            Knight("b",0,1), Knight("b",0,6),
            Bishop("b",0,2), Bishop("b",0,5),
            King("b",0,4), Queen("b",0,3),
        }
        for i in range(8):
            self.blackPice.add(Pawn("b",1,i))

        self.whitePice = {
            Rook("w",7,0), Rook("w",7,7),
            Knight("w",7,1), Knight("w",7,6),
            Bishop("w",7,2), Bishop("w",7,5),
            King("w",7,4), Queen("w",7,3),
        }
        for i in range(8):
            self.whitePice.add(Pawn("w",6,i))

        self.pieces = self.whitePice | self.blackPice

        for p in self.pieces:
            if type(p) == King and p.color == "w":
                self.wKing = p
            elif type(p) == King and p.color == "b":
                self.bKing = p

    # --- Pomocné metódy ---------------------------------------------

    def _position_hash(self):
        # Hash aktuálnej pozície pre kontrolu trojnásobného opakovania
        pieces_tuple = tuple(sorted(
            (type(p).__name__, p.color, p.col, p.row) for p in self.pieces
        ))
        duck_pos = (self.duck.col, self.duck.row)
        return hash((pieces_tuple, duck_pos, self.turn))

    def _is_insufficient_material(self):
        # Remíza kvôli nedostatku materiálu
        minor = (Knight, Bishop)
        w_pieces = [p for p in self.whitePice if type(p) != King]
        b_pieces = [p for p in self.blackPice if type(p) != King]

        # Kráľ proti kráľovi
        if not w_pieces and not b_pieces:
            return True

        # Jeden ľahký kus proti prázdnej strane
        if len(w_pieces) == 1 and not b_pieces and type(w_pieces[0]) in minor:
            return True
        if len(b_pieces) == 1 and not w_pieces and type(b_pieces[0]) in minor:
            return True

        # Strelec proti strelcovi na rovnakej farbe polí
        if (len(w_pieces) == 1 and len(b_pieces) == 1
                and type(w_pieces[0]) == Bishop
                and type(b_pieces[0]) == Bishop):
            w_sq = (w_pieces[0].col + w_pieces[0].row) % 2
            b_sq = (b_pieces[0].col + b_pieces[0].row) % 2
            if w_sq == b_sq:
                return True

        return False

    def piece_at(self, col, row):
        for p in self.pieces:
            if p.col == col and p.row == row:
                return p
        return None

    def cell_occupied(self, col, row):
        if self.duck.col == col and self.duck.row == row:
            return True
        return any(p.col == col and p.row == row for p in self.pieces)

    def duck_actions(self):
        # Všetky polia, kam možno postaviť kačicu (ľubovoľné voľné pole)
        result = []
        for col in range(8):
            for row in range(8):
                if not any(p.col == col and p.row == row for p in self.pieces):
                    result.append(DuckMove(col, row))
        return result

    def cell_occupied_by_enemy(self, col, row, color):
        return any(p.col == col and p.row == row and p.color != color
                   for p in self.pieces)

    def can_pass_from(self, piece, col, row):
        # Kontrola, či nie sú figúry v ceste (pre vežu/strelca/dámu). Kačica tiež blokuje.
        if col == piece.col and row != piece.row:
            mini, maxi = min(row, piece.row), max(row, piece.row)
            for i in range(mini + 1, maxi):
                if self.cell_occupied(piece.col, i):
                    return False
            return True
        elif col != piece.col and row == piece.row:
            mini, maxi = min(col, piece.col), max(col, piece.col)
            for i in range(mini + 1, maxi):
                if self.cell_occupied(i, piece.row):
                    return False
            return True
        elif abs(col - piece.col) == abs(row - piece.row):
            col_step = 1 if col > piece.col else -1
            row_step = 1 if row > piece.row else -1
            c, r = piece.col + col_step, piece.row + row_step
            while (c, r) != (col, row):
                if self.cell_occupied(c, r):
                    return False
                c += col_step
                r += row_step
            return True
        return False

    # --- Generovanie ťahov ---------------------------------------------

    def _pseudo_legal(self, piece):
        # Všetky polia, kam môže figúra ísť podľa pravidiel pohybu
        moves = []
        c, r = piece.col, piece.row

        if type(piece) == Pawn:
            direction = -1 if piece.color == "w" else 1
            nc = c + direction
            if 0 <= nc < 8 and not self.cell_occupied(nc, r):
                moves.append((nc, r))
                if piece.start:
                    nc2 = c + 2 * direction
                    if 0 <= nc2 < 8 and not self.cell_occupied(nc2, r):
                        moves.append((nc2, r))
            for dr in [-1, 1]:
                nr = r + dr
                if 0 <= nc < 8 and 0 <= nr < 8:
                    if self.cell_occupied_by_enemy(nc, nr, piece.color):
                        moves.append((nc, nr))
            # Branie en passant
            if self.lastTurne is not None and type(self.lastTurne) == Pawn:
                lt = self.lastTurne
                if (lt.color != piece.color
                        and lt.col == c
                        and abs(lt.row - r) == 1):
                    ep_col = c + direction
                    if 0 <= ep_col < 8:
                        moves.append((ep_col, lt.row))

        elif type(piece) == Knight:
            for dc, dr in [(-2,-1),(-2,1),(-1,-2),(-1,2),
                           (1,-2),(1,2),(2,-1),(2,1)]:
                nc, nr = c+dc, r+dr
                if 0 <= nc < 8 and 0 <= nr < 8:
                    # Kačicu nemožno zobrať - preskakujeme pole kačice
                    if self.duck.col == nc and self.duck.row == nr:
                        continue
                    if not self.cell_occupied(nc, nr) or self.cell_occupied_by_enemy(nc, nr, piece.color):
                        moves.append((nc, nr))

        elif type(piece) == Bishop:
            for dc, dr in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nc, nr = c+dc, r+dr
                while 0 <= nc < 8 and 0 <= nr < 8:
                    if self.cell_occupied(nc, nr):
                        if self.cell_occupied_by_enemy(nc, nr, piece.color):
                            # Kačicu nemožno zobrať
                            if not (self.duck.col == nc and self.duck.row == nr):
                                moves.append((nc, nr))
                        break
                    moves.append((nc, nr))
                    nc += dc; nr += dr

        elif type(piece) == Rook:
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
                nc, nr = c+dc, r+dr
                while 0 <= nc < 8 and 0 <= nr < 8:
                    if self.cell_occupied(nc, nr):
                        if self.cell_occupied_by_enemy(nc, nr, piece.color):
                            if not (self.duck.col == nc and self.duck.row == nr):
                                moves.append((nc, nr))
                        break
                    moves.append((nc, nr))
                    nc += dc; nr += dr

        elif type(piece) == Queen:
            for dc, dr in [(-1,-1),(-1,0),(-1,1),(0,-1),
                           (0,1),(1,-1),(1,0),(1,1)]:
                nc, nr = c+dc, r+dr
                while 0 <= nc < 8 and 0 <= nr < 8:
                    if self.cell_occupied(nc, nr):
                        if self.cell_occupied_by_enemy(nc, nr, piece.color):
                            if not (self.duck.col == nc and self.duck.row == nr):
                                moves.append((nc, nr))
                        break
                    moves.append((nc, nr))
                    nc += dc; nr += dr

        elif type(piece) == King:
            for dc in [-1,0,1]:
                for dr in [-1,0,1]:
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = c+dc, r+dr
                    if 0 <= nc < 8 and 0 <= nr < 8:
                        if self.duck.col == nc and self.duck.row == nr:
                            continue
                        if not self.cell_occupied(nc, nr) or self.cell_occupied_by_enemy(nc, nr, piece.color):
                            moves.append((nc, nr))

            # Rošáda - len ak kráľ a veža sa ešte nepohli, cesta je voľná
            # V Duck Chess šach neexistuje, takže is_in_check vynechávame
            if not piece.has_moved:
                back_row = 7 if piece.color == "w" else 0
                # Krátka rošáda
                rook_k = self.piece_at(back_row, 7)
                if (rook_k and type(rook_k) == Rook and not rook_k.has_moved
                        and not self.cell_occupied(back_row, 5)
                        and not self.cell_occupied(back_row, 6)):
                    moves.append((back_row, 6))
                # Dlhá rošáda
                rook_q = self.piece_at(back_row, 0)
                if (rook_q and type(rook_q) == Rook and not rook_q.has_moved
                        and not self.cell_occupied(back_row, 1)
                        and not self.cell_occupied(back_row, 2)
                        and not self.cell_occupied(back_row, 3)):
                    moves.append((back_row, 2))

        return moves

    def legal_moves(self):
        # Zoznam objektov Move, ktoré sú legálne.
        # V Duck Chess šach neexistuje - všetky pseudolegálne ťahy sú legálne.
        # Jediné obmedzenie: kačicu nemožno zobrať.
        # (Branie kačice je už vylúčené v _pseudo_legal cez kontrolu duck.col/row.)
        result    = []
        my_pieces = self.whitePice if self.turn == "w" else self.blackPice
        promo_row = 0 if self.turn == "w" else 7

        for piece in list(my_pieces):
            for col, row in self._pseudo_legal(piece):
                if type(piece) == Pawn and col == promo_row:
                    for promo in Move.PROMOTION_PIECES:
                        result.append(Move(piece, col, row, promo))
                else:
                    result.append(Move(piece, col, row))

        return result

    def legal_actions(self):
        # Zoznam indexov legálnych ťahov (pre neurónovú sieť a MCTS).
        # duck_phase=True  -> indexy ťahov kačice (0..63)
        # duck_phase=False -> indexy šachových ťahov (64..4415)
        if self.duck_phase:
            return [m.to_index() for m in self.duck_actions()]
        return [m.to_index() + 64 for m in self.legal_moves()]

    # --- Vykonanie ťahu ---------------------------------------------

    def step(self, action):
        # Vykonaj ťah podľa indexu.
        # duck_phase=False: action >= 64 — šachový ťah.
        # duck_phase=True:  action 0..63 — ťah kačice.
        if self.duck_phase:
            # Postavíme kačicu
            duck_move = DuckMove.from_index(action)
            self.duck.col = duck_move.col
            self.duck.row = duck_move.row
            self.duck_phase = False
            # Zmeníme ťah a aktualizujeme stav až po ťahu kačice
            self.turn = "b" if self.turn == "w" else "w"
            self.move_count += 1
            h = self._position_hash()
            self.position_history[h] = self.position_history.get(h, 0) + 1
            self._update_status()
        else:
            # Šachový ťah (index posunutý o 64)
            move = Move.from_index(action - 64, self)
            if move is None:
                raise ValueError(f"Žiadna figúra pre ťah {action}")
            self._apply_move(move)
            # Po šachovom ťahu prejdeme do fázy kačice
            # (ak hra neskončila zajatím kráľa)
            if not self.done:
                self.duck_phase = True

    def _apply_move(self, move):
        piece  = move.piece
        to_col = move.to_col
        to_row = move.to_row

        # Branie en passant
        if (type(piece) == Pawn
                and self.lastTurne is not None
                and type(self.lastTurne) == Pawn
                and to_row == self.lastTurne.row
                and to_col != piece.col
                and self.lastTurne.col == piece.col
                and not self.cell_occupied(to_col, to_row)):
            ep = self.lastTurne
            self.pieces.discard(ep)
            self.whitePice.discard(ep)
            self.blackPice.discard(ep)

        # Zapamätáme si pre branie en passant
        if type(piece) == Pawn and abs(to_col - piece.col) == 2:
            self.lastTurne = piece
        else:
            self.lastTurne = None

        # Odoberie pešiakovi počiatočný príznak
        if type(piece) == Pawn:
            piece.start = False

        # Bežné branie
        captured = self.piece_at(to_col, to_row)
        if captured and captured.color != piece.color:
            self.pieces.discard(captured)
            self.whitePice.discard(captured)
            self.blackPice.discard(captured)
            self.no_capture_moves = 0

            # --- KĽÚČOVÉ PRAVIDLO DUCK CHESS ---------------------------------------------
            # Zobrali sme kráľa - hra okamžite končí, vyhral útočník.
            # Po tomto ťahu kačice nie je.
            if type(captured) == King:
                self.done   = True
                self.result = piece.color
                return
            # ---------------------------------------------
        else:
            self.no_capture_moves += 1

        # Rošáda
        if type(piece) == King and abs(to_row - piece.row) == 2:
            back_row = piece.col
            if to_row == 6:
                rook = self.piece_at(back_row, 7)
                if rook:
                    rook.col, rook.row = back_row, 5
            else:
                rook = self.piece_at(back_row, 0)
                if rook:
                    rook.col, rook.row = back_row, 3
            piece.has_moved = True

        if type(piece) == King:
            piece.has_moved = True
        if type(piece) == Rook:
            piece.has_moved = True

        # Premena pešiaka
        if type(piece) == Pawn and (to_col == 0 or to_col == 7):
            promo_class = move.promotion if move.promotion is not None else Queen
            self.pieces.discard(piece)
            if piece.color == "w":
                self.whitePice.discard(piece)
                new_piece = promo_class("w", to_col, to_row)
                self.whitePice.add(new_piece)
            else:
                self.blackPice.discard(piece)
                new_piece = promo_class("b", to_col, to_row)
                self.blackPice.add(new_piece)
            self.pieces.add(new_piece)
        else:
            piece.col = to_col
            piece.row = to_row

    def _update_status(self):
        # Kontrolujeme podmienky remízy po úplnom ťahu (figúra + kačica).
        # Mat neexistuje — len zajatie kráľa (riešené v _apply_move).
        #
        # Remíza je možná len cez:
        #   1. Trojnásobné opakovanie pozície
        #   2. 100 poloťahov bez brania (pravidlo 50 ťahov)
        #   3. Limit 150 ťahov (dočasná poistka kým sa agent učí)
        # Holý kráľ - NIE je remíza, môže vyhrať.
        if self.done:
            return

        # Trojnásobné opakovanie pozície
        if self.position_history and max(self.position_history.values()) >= 3:
            self.result = "draw"
            self.done   = True
            return

        # Pravidlo 50 ťahov (100 poloťahov bez brania a ťahov pešiakom)
        if self.no_capture_moves >= 100:
            self.result = "draw"
            self.done   = True
            return

        # Limit ťahov - záložná možnosť kým sa agent učí
        # (neskôr možno zvýšiť alebo odstrániť)
        if self.move_count >= 150:
            self.result = "draw"
            self.done   = True

    # --- Metódy pre RL ---------------------------------------------

    def is_terminal(self):
        # Skončila sa hra?
        return self.done

    def winner(self):
        # 'w', 'b', 'draw' alebo None ak hra ešte prebieha
        return self.result

    def clone(self):
        # Hlboká kópia pre simulácie MCTS
        return copy.deepcopy(self)

    def get_observation(self):
        # Stav dosky ako numpy pole tvaru (16, 8, 8).
        #
        # Kanály:
        #   0  - biele pešiaky
        #   1  - biele jazdce
        #   2  - biele strelce
        #   3  - biele veže
        #   4  - biele dámy
        #   5  - biele kráľe
        #   6  - čierne pešiaky
        #   7  - čierne jazdce
        #   8  - čierne strelce
        #   9  - čierne veže
        #   10 - čierne dámy
        #   11 - čierne kráľe
        #   12 - pozícia kačice
        #   13 - fáza ťahu (všetky 1 = ťah kačice)
        #   14 - čí rad (všetky 1 ak biele)
        #   15 - zrkadlo aktuálneho hráča (1 = hrajú biele)
        obs = np.zeros((16, 8, 8), dtype=np.float32)
        flip = (self.turn == "b")
        piece_to_channel = {
            ("w", Pawn):   0, ("w", Knight): 1, ("w", Bishop): 2,
            ("w", Rook):   3, ("w", Queen):  4, ("w", King):   5,
            ("b", Pawn):   6, ("b", Knight): 7, ("b", Bishop): 8,
            ("b", Rook):   9, ("b", Queen): 10, ("b", King):  11,
        }

        for p in self.pieces:
            ch = piece_to_channel.get((p.color, type(p)))
            if ch is not None:
                col = (7 - p.col) if flip else p.col
                row = (7 - p.row) if flip else p.row
                obs[ch, col, row] = 1.0

        if self.duck.col is not None:
            col = (7 - self.duck.col) if flip else self.duck.col
            row = (7 - self.duck.row) if flip else self.duck.row
            obs[12, col, row] = 1.0

        if self.duck_phase:
            obs[13, :, :] = 1.0

        if self.turn == "w":
            obs[14, :, :] = 1.0

        obs[15, :, :] = 1.0

        return obs

    def action_size(self):
        # Veľkosť priestoru akcií:
        #   0..63      - ťahy kačice
        #   64..4159   - bežné šachové ťahy (64×64 + posun 64)
        #   4160..4415 - ťahy s premenou
        return 64 + 64 * 64 + 64 * 4  # 4416

    # --- Ladenie ---------------------------------------------

    def render(self):
        # Vypíše dosku do konzoly
        board = [["·"] * 8 for _ in range(8)]
        for p in self.pieces:
            board[p.col][p.row] = p.rep
        if self.duck.col is not None:
            board[self.duck.col][self.duck.row] = "🦆"
        print("  a b c d e f g h")
        for i, row in enumerate(board):
            print(f"{8-i} {' '.join(row)} {8-i}")
        print("  a b c d e f g h")
        phase = "→ postav kačicu" if self.duck_phase else ""
        print(f"Ťah: {'Biele' if self.turn == 'w' else 'Čierne'} {phase}| "
              f"Odohrané ťahy: {self.move_count}")
        if self.done:
            print(f"Hra skončila: {self.result}")
        print()