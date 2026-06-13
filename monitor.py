# monitor.py - živý monitor trénovania Duck Chess.
# Otvor v samostatnom okne kým beží train.py.

import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from datetime import datetime

LOG_PATH    = "checkpoints/training_log.json"
REFRESH_SEC = 10  # obnova každých 10 sekúnd


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def update(frame, axes, fig):
    log = load_log()
    if not log:
        return

    for ax in axes:
        ax.cla()

    iters       = [e["iteration"]       for e in log]
    losses      = [e["avg_loss"]        for e in log if e["avg_loss"] is not None]
    val_losses  = [e["avg_value_loss"]  for e in log if e["avg_value_loss"] is not None]
    pol_losses  = [e["avg_policy_loss"] for e in log if e["avg_policy_loss"] is not None]
    game_lens   = [e["avg_game_len"]    for e in log]
    wins_w      = [e["results"]["w"]    for e in log]
    wins_b      = [e["results"]["b"]    for e in log]
    draws       = [e["results"]["draw"] for e in log]
    buf_sizes   = [e["buffer_size"]     for e in log]
    times       = [e.get("elapsed_sec", 0) / 60 for e in log]  # v minútach

    loss_iters = iters[:len(losses)]

    # --- Graf 1: Loss ---------------------------------------------
    ax = axes[0]
    if losses:
        ax.plot(loss_iters, losses,     label="Celkový loss",  color="#e74c3c", linewidth=2)
        ax.plot(loss_iters, val_losses, label="Loss hodnoty",  color="#3498db", linewidth=1.5, linestyle="--")
        ax.plot(loss_iters, pol_losses, label="Loss politiky", color="#2ecc71", linewidth=1.5, linestyle="--")
        ax.set_title("Loss (čím menej — tým lepšie)", fontsize=11)
        ax.set_xlabel("Generácia")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        # Anotácia poslednej hodnoty
        ax.annotate(f"{losses[-1]:.3f}",
                    xy=(loss_iters[-1], losses[-1]),
                    fontsize=9, color="#e74c3c",
                    xytext=(5, 5), textcoords="offset points")

    # --- Graf 2: Dĺžka partií ---------------------------------------------
    ax = axes[1]
    ax.plot(iters, game_lens, color="#9b59b6", linewidth=2, marker="o", markersize=3)
    ax.set_title("Priemerná dĺžka partie (ťahov)", fontsize=11)
    ax.set_xlabel("Generácia")
    ax.grid(alpha=0.3)
    ax.annotate(f"{game_lens[-1]:.1f}",
                xy=(iters[-1], game_lens[-1]),
                fontsize=9, color="#9b59b6",
                xytext=(5, 5), textcoords="offset points")
    # Bodkovaná čiara - očakávaná hodnota pre náhodnú hru
    ax.axhline(y=40, color="gray", linestyle=":", alpha=0.5, label="~náhodná hra")
    ax.legend(fontsize=9)

    # --- Graf 3: Výsledky partií ---------------------------------------------
    ax = axes[2]
    ax.stackplot(iters, wins_w, draws, wins_b,
                 labels=["Biele", "Remíza", "Čierne"],
                 colors=["#ecf0f1", "#95a5a6", "#2c3e50"],
                 alpha=0.8)
    ax.set_title(f"Výsledky partií (po {log[0]['results']['w']+log[0]['results']['b']+log[0]['results']['draw']} hier)", fontsize=11)
    ax.set_xlabel("Generácia")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # --- Graf 4: Veľkosť bufferu ---------------------------------------------
    ax = axes[3]
    ax.plot(iters, buf_sizes, color="#e67e22", linewidth=2)
    ax.fill_between(iters, buf_sizes, alpha=0.2, color="#e67e22")
    ax.set_title("Veľkosť replay bufferu (pozícií)", fontsize=11)
    ax.set_xlabel("Generácia")
    ax.grid(alpha=0.3)
    ax.annotate(f"{buf_sizes[-1]:,}",
                xy=(iters[-1], buf_sizes[-1]),
                fontsize=9, color="#e67e22",
                xytext=(5, 5), textcoords="offset points")

    # --- Titulok okna ---------------------------------------------
    last      = log[-1]
    avg_time  = sum(times) / len(times) if times else 0
    eta_iters = 100 - last["iteration"]  # predpokladáme 100 generácií
    eta_min   = eta_iters * avg_time

    fig.suptitle(
        f"Duck Chess — trénovanie  |  Generácia {last['iteration']}  |  "
        f"Loss: {last['avg_loss']}  |  "
        f"~{avg_time:.1f} min/gen  |  "
        f"ETA: {eta_min:.0f} min  |  "
        f"Aktualizované: {datetime.now().strftime('%H:%M:%S')}",
        fontsize=11, y=0.98
    )


def main():
    if not os.path.exists("../checkpoints"):
        print("Priečinok checkpoints nebol nájdený.")
        print("Spusti train.py - monitor zachytí log automaticky.")

    print(f"Monitor spustený. Obnova každých {REFRESH_SEC} sek.")
    print("Zatvor okno pre ukončenie.")

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")

    gs   = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    axes = [
        fig.add_subplot(gs[0, 0]),  # loss
        fig.add_subplot(gs[0, 1]),  # dĺžka partií
        fig.add_subplot(gs[1, 0]),  # výsledky
        fig.add_subplot(gs[1, 1]),  # buffer
    ]

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ani = FuncAnimation(
        fig,
        update,
        fargs=(axes, fig),
        interval=REFRESH_SEC * 1000,
        cache_frame_data=False
    )

    # Prvá obnova ihneď
    update(0, axes, fig)
    plt.show()


if __name__ == "__main__":
    main()