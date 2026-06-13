# Duck Chess — Reinforcement Learning

Ročníkový projekt (RPBP) — FMFI UK Bratislava, 2026  
Autor: Shkaldykov Vladyslav  
Školiteľ: Lacko Peter

Implementácia algoritmu AlphaZero pre hru Duck Chess.

---

## Súbory

| Súbor | Popis |
|-------|-------|
| `chess.py` | Herná logika Duck Chess (pravidlá, ťahy, kačica) |
| `neural_net.py` | Reziduálna neurónová sieť (ResNet, 6 blokov, 128 kanálov) |
| `mcts.py` | Monte Carlo Tree Search s PUCT výberom |
| `train.py` | Trénovací cyklus — self-play → trénovanie → opakovanie |
| `monitor.py` | Živý monitor trénovania (grafy loss, dĺžky partií, výsledky) |
| `play.py` | Hra človek vs. agent (grafické rozhranie, myš) |
| `GUI.py` | Pomocný modul GUI |

---


Trénovanie od nuly cez `train.py` nevyžaduje žiadne checkpointy —  
sieť sa inicializuje náhodne a učí od prvej generácie.

Moje výsledky tam nebudú, pretože vážia viac ako 2 gigabajty

---

## Poznámky

- Model sa počas projektu **neskonvergoval** z dôvodu nedostatku výpočtových zdrojov (CPU, ~15–20 hodín na 100 generácií × 50 partií)
- Pre skutočnú konvergenciu je potrebné GPU a rádovo 10× viac generácií
