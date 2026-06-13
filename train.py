# train.py - úplný trénovací cyklus AlphaZero pre Duck Chess.
import os
import json
import time
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from chess       import ChessGame
from neural_net  import DuckChessNet
from mcts        import MCTS


# ---------------------------------------------
#  Konfigurácia
# ---------------------------------------------

CONFIG = {
    # Sieť
    "num_res_blocks": 6,       # počet ResBlock (viac = inteligentnejšie, ale pomalšie)
    "channels":       128,     # kanály vnútri ResNet

    # MCTS
    "num_simulations":   50,   # simulácií na ťah
    "mcts_max_depth":    20,   # maximálna hĺbka jednej simulácie (bez toho zasekne!)
    "c_puct":            1.5,  # rovnováha explorácie vs využitia
    "dirichlet_alpha":   0.5,  # šum v koreni stromu

    # Sebahráčske učenie
    "games_per_iter":    50,   # partií za jednu generáciu
    "temp_threshold":    30,   # po tomto ťahu temperature → 0 (chamtivý výber)

    # Trénovanie
    "iterations":        100,  # počet generácií (self-play → trénovanie → opakovanie)
    "batch_size":        256,
    "lr":                0.001,
    "weight_decay":      1e-4,  # L2 regularizácia
    "epochs_per_iter":   5,     # epoch trénovania na každej generácii
    "replay_buffer_size": 50000,

    # Ukladanie
    "checkpoint_dir":   "checkpoints",
    "save_every":       5,      # uložiť každých N generácií
}


# ---------------------------------------------
#  Sebahráčske učenie: jedna partia
# ---------------------------------------------

def self_play_game(network, device, config, game_num=0, verbose=True):
    # Odohrá jednu partiu samu so sebou.
    # Vráti zoznam trénovacích príkladov:
    #     [(observation, mcts_policy, value), ...]
    cols = "abcdefgh"
    game = ChessGame()

    mcts = MCTS(
        network,
        device,
        num_simulations = config["num_simulations"],
        c_puct          = config["c_puct"],
        dirichlet_alpha = config["dirichlet_alpha"],
        max_depth       = config["mcts_max_depth"],
    )

    history  = []
    move_num = 0

    if verbose:
        print(f"\n  Partia {game_num} začala")

    while not game.is_terminal():
        temperature = 1.0 if move_num < config["temp_threshold"] else 0.0

        obs    = game.get_observation()
        policy = mcts.run(game, temperature=temperature)
        history.append((obs, policy, game.turn))

        action = np.random.choice(len(policy), p=policy)

        if verbose:
            if game.duck_phase:
                duck_col, duck_row = action // 8, action % 8
                print(f"    🦆 kačica -> {cols[duck_row]}{8 - duck_col}", flush=True)
            else:
                from chess import Move as M
                mv = M.from_index(action - 64, game)
                if mv:
                    who = "B" if game.turn == "w" else "Č"
                    promo = f"={mv.promotion.__name__[0]}" if mv.promotion else ""
                    print(f"    Ťah {move_num+1:>3}. [{who}] {mv}{promo}", flush=True)

        game.step(action)
        move_num += 1

    result = game.winner()
    if verbose:
        label = {"w": "Vyhrali Biele ♔", "b": "Vyhrali Čierne ♚", "draw": "Remíza"}.get(result, "?")
        print(f"  → {label} za {move_num} poloťahov", flush=True)

    training_data = []
    game_length = len(history)

    for i, (obs, policy, player) in enumerate(history):
        if result == "draw":
            value = -1.0
        elif result == player:
            value = 1.0
        else:
            value = -1.0

        # Útlm - skoré ťahy sú menej isté
        # ťah 0 -> koeficient 0.5, posledný ťah -> 1.0
        decay = 0.5 + 0.5 * (i / max(game_length - 1, 1))
        value *= decay

        # Penalizácia víťaza za dlhé víťazstvo
        length_penalty = max(0.0, 1.0 - game_length / 150)
        if result != "draw" and result == player:
            value *= (0.7 + 0.3 * length_penalty)

        training_data.append((obs, policy, value))

    return training_data, result


# ---------------------------------------------
#  Trénovanie siete na nazbieraných dátach
# ---------------------------------------------

def train_step(network, optimizer, replay_buffer, config, device):
    # Jedna epocha trénovania na náhodnej dávke z replay bufferu
    if len(replay_buffer) < config["batch_size"]:
        return None

    network.train()

    batch = random.sample(replay_buffer, config["batch_size"])
    obs_b, policy_b, value_b = zip(*batch)

    obs_t    = torch.tensor(np.array(obs_b),    dtype=torch.float32).to(device)
    policy_t = torch.tensor(np.array(policy_b), dtype=torch.float32).to(device)
    value_t  = torch.tensor(np.array(value_b),  dtype=torch.float32).to(device)

    policy_logits, value_pred = network(obs_t)

    # Stratová funkcia z článku AlphaZero: l = (z - v)² - π⊤ log p + c||θ||²
    value_loss  = F.mse_loss(value_pred.squeeze(-1), value_t)
    log_policy  = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_t * log_policy).sum(dim=1).mean()
    loss        = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss":        loss.item(),
        "value_loss":  value_loss.item(),
        "policy_loss": policy_loss.item(),
    }


# ---------------------------------------------
#  Ukážková partia do konzoly
# ---------------------------------------------

def show_game(network, device, sims=50):
    # Odohrá jednu partiu a vypíše každý ťah
    cols = "abcdefgh"
    game = ChessGame()
    mcts = MCTS(
        network, device,
        num_simulations=sims,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        max_depth=20,
    )

    print("\n" + "─"*40)
    print("  UKÁŽKOVÁ PARTIA")
    print("─"*40)
    game.render()

    move_num = 0
    while not game.is_terminal():
        policy = mcts.run(game, temperature=0)
        action = int(np.argmax(policy))

        if game.duck_phase:
            duck_col, duck_row = action // 8, action % 8
            print(f"  🦆 Kačica -> {cols[duck_row]}{8 - duck_col}")
        else:
            from chess import Move
            mv = Move.from_index(action - 64, game)
            if mv:
                turn_name = "Biele" if game.turn == "w" else "Čierne"
                promo = f"={mv.promotion.__name__}" if mv.promotion else ""
                print(f"\nŤah {move_num + 1}. {turn_name}: {mv}{promo}")

        game.step(action)

        if not game.duck_phase:
            game.render()
            move_num += 1

        if move_num >= 30:
            print("  ... (zobrazených prvých 30 ťahov)")
            break

    result = game.winner()
    labels = {"draw": "Remíza", "w": "Vyhrali Biele ♔", "b": "Vyhrali Čierne ♚"}
    print(f"Výsledok: {labels.get(result, '?')}")
    print("─"*40)


# ---------------------------------------------
#  Hlavný trénovací cyklus
# ---------------------------------------------

def train(config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    network = DuckChessNet(
        num_res_blocks = config["num_res_blocks"],
        channels       = config["channels"],
    ).to(device)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Parametre siete: {total_params:,}")

    replay_buffer = deque(maxlen=config["replay_buffer_size"])

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    log_path     = os.path.join(config["checkpoint_dir"], "training_log.json")
    training_log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)
        print(f"Načítaný log: {len(training_log)} generácií")

    start_iter = 0
    ckpt_path  = os.path.join(config["checkpoint_dir"], "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        network.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"] + 1
        print(f"Načítaný checkpoint, pokračujeme od generácie {start_iter}")

    # Načítame buffer ak existuje
    buf_path = os.path.join(config["checkpoint_dir"], "buffer.pkl")
    if os.path.exists(buf_path):
        with open(buf_path, "rb") as f:
            replay_buffer.extend(pickle.load(f))
        print(f"Načítaný buffer: {len(replay_buffer)} pozícií")

    # --- Hlavný cyklus ---------------------------------------------
    for iteration in range(start_iter, config["iterations"]):
        iter_start = time.time()
        print(f"\n{'='*50}")
        print(f"GENERÁCIA {iteration + 1} / {config['iterations']}")
        print(f"{'='*50}")

        # --- Fáza 1: Sebahráčske učenie ---------------------------------------------
        print(f"\n[Self-play] Hráme {config['games_per_iter']} partií...")
        results  = {"w": 0, "b": 0, "draw": 0}
        new_data = []

        for game_num in range(config["games_per_iter"]):
            game_data, result = self_play_game(
                network, device, config,
                game_num=game_num + 1, verbose=True
            )
            new_data.extend(game_data)
            key = result if result in results else "draw"
            results[key] += 1

        replay_buffer.extend(new_data)
        avg_game_len = len(new_data) / max(config["games_per_iter"], 1)
        print(f"Buffer: {len(replay_buffer)} pozícií | "
              f"Priem. dĺžka partie: {avg_game_len:.1f} ťahov")
        print(f"Výsledky: Biele {results['w']} | "
              f"Čierne {results['b']} | Remíza {results['draw']}")

        # --- Fáza 2: Trénovanie siete ---------------------------------------------
        print(f"\n[Trénovanie] Trénujeme sieť ({config['epochs_per_iter']} epoch)...")
        all_metrics = []
        for epoch in range(config["epochs_per_iter"]):
            m = train_step(network, optimizer, list(replay_buffer), config, device)
            if m:
                all_metrics.append(m)
                print(f"  Epocha {epoch + 1} | "
                      f"Loss: {m['loss']:.4f} | "
                      f"Hodnota: {m['value_loss']:.4f} | "
                      f"Politika: {m['policy_loss']:.4f}")

        # --- Zaznamenáme generáciu ---------------------------------------------
        elapsed   = time.time() - iter_start
        log_entry = {
            "iteration":        iteration + 1,
            "timestamp":        time.time(),
            "elapsed_sec":      round(elapsed, 1),
            "results":          results,
            "avg_game_len":     round(avg_game_len, 1),
            "buffer_size":      len(replay_buffer),
            "avg_loss":         round(float(np.mean([m["loss"]         for m in all_metrics])), 4) if all_metrics else None,
            "avg_value_loss":   round(float(np.mean([m["value_loss"]   for m in all_metrics])), 4) if all_metrics else None,
            "avg_policy_loss":  round(float(np.mean([m["policy_loss"]  for m in all_metrics])), 4) if all_metrics else None,
        }
        if all_metrics:
            print(f"Priemerný loss: {log_entry['avg_loss']:.4f} | "
                  f"Čas generácie: {elapsed/60:.1f} min")

        training_log.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        # --- Uloženie checkpointu ---------------------------------------------
        if (iteration + 1) % config["save_every"] == 0:
            ckpt = {
                "iteration": iteration,
                "network":   network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config":    config,
            }
            torch.save(ckpt, ckpt_path)
            named = os.path.join(
                config["checkpoint_dir"], f"iter_{iteration + 1:04d}.pt"
            )
            torch.save(ckpt, named)
            print(f"\nCheckpoint uložený: {named}")

            # Uložíme buffer vedľa checkpointu
            buf_path = os.path.join(config["checkpoint_dir"], "buffer.pkl")
            with open(buf_path, "wb") as f:
                pickle.dump(list(replay_buffer), f)
            print(f"Buffer uložený: {len(replay_buffer)} pozícií")
            show_game(network, device, sims=50)

    print("\n Trénovanie dokončené!")


if __name__ == "__main__":
    train()