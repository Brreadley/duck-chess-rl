# mcts.py — Monte Carlo Tree Search pre Duck Chess.

import math
import numpy as np


class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior        = prior    # P(s,a) z neurónovej siete
        self.visit_count  = 0        # N(s,a)
        self.value_sum    = 0.0      # W(s,a)
        self.children     = {}       # akcia → MCTSNode

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits, c_puct=1.5):
        # Formula PUCT z článku AlphaZero:
        # Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration


class MCTS:
    def __init__(self, network, device, num_simulations=200, c_puct=1.5, dirichlet_alpha=0.3, max_depth=20):
        self.network         = network
        self.device          = device
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.dirichlet_alpha = dirichlet_alpha  # šum pre exploráciu
        self.max_depth       = max_depth        # maximálna hĺbka jednej simulácie

    def run(self, game, temperature=1.0):
        # Spustí num_simulations iterácií MCTS.
        # Vráti vektor pravdepodobností ťahov (policy).
        # temperature=1.0 - viac explorácie (začiatok hry)
        # temperature=0.0 - chamtivý výber (koniec hry)
        root = MCTSNode(prior=0.0)
        self._expand(root, game)
        self._add_dirichlet_noise(root)  # šum v koreni pre exploráciu

        for _ in range(self.num_simulations):
            node        = root
            scratch     = game.clone()
            search_path = [node]

            # KROK 1: VÝBER - ideme nadol stromom (s obmedzením hĺbky)
            depth = 0
            while node.is_expanded() and not scratch.is_terminal() and depth < self.max_depth:
                action, node = self._select(node)
                scratch.step(action)
                search_path.append(node)
                depth += 1

            # KROK 2 & 3: ROZŠÍRENIE + ohodnotenie neurónovou sieťou
            if not scratch.is_terminal():
                value = self._expand(node, scratch)
            else:
                # Skutočný výsledok hry
                result = scratch.winner()
                if result == "draw":
                    value = 0.0
                else:
                    # +1 ak vyhral ten kto práve ťahá, -1 ak prehral
                    value = 1.0 if result == scratch.turn else -1.0

            # KROK 4: SPÄTNÉ ŠÍRENIE
            self._backpropagate(search_path, value, game.turn)

        # Zozbierame štatistiku návštev -> pravdepodobnosti ťahov
        return self._get_policy(root, game.action_size(), temperature)

    def _select(self, node):
        # Vyberieme potomka s maximálnym UCB skóre
        best_score  = -float('inf')
        best_action = None
        best_child  = None
        for action, child in node.children.items():
            score = child.ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score  = score
                best_action = action
                best_child  = child
        return best_action, best_child

    def _expand(self, node, game):
        # Rozšírime uzol: opýtame sa neurónovej siete, vytvoríme potomkov
        legal = game.legal_actions()
        if not legal:
            return 0.0

        policy, value = self.network.predict(
            game.get_observation(), legal, self.device
        )

        for action in legal:
            node.children[action] = MCTSNode(prior=float(policy[action]))

        return value

    def _backpropagate(self, search_path, value, root_turn):
        # Aktualizujeme skóre všetkých uzlov na ceste ku koreňu
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum   += value
            # Zmeníme znamienko - dobrá pozícia pre súpera je zlá pre nás
            value = -value

    def _add_dirichlet_noise(self, root):
        # Pridáme Dirichletov šum do koreňa pre exploráciu (ako v článku)
        if not root.children:
            return
        actions = list(root.children.keys())
        noise   = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        frac    = 0.25  # 25 % šum, 75 % prior zo siete
        for action, n in zip(actions, noise):
            root.children[action].prior = (
                (1 - frac) * root.children[action].prior + frac * n
            )

    def _get_policy(self, root, action_size, temperature):
        # Prevedieme počty návštev na pravdepodobnosti
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature == 0:
            # Chamtivý výber - všetko na najlepší ťah
            best = np.argmax(visits)
            policy = np.zeros(action_size, dtype=np.float32)
            policy[best] = 1.0
            return policy

        # Teplotné vyhladzovanie: visits^(1/T)
        visits = visits ** (1.0 / temperature)
        total  = visits.sum()
        if total > 0:
            return visits / total
        return visits