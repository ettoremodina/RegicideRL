# Regicide AI Training Audit Report

Scope:
- Rules vs implementation review: [rules/RegicideRulesA4.txt](rules/RegicideRulesA4.txt) vs [regicide.py](regicide.py)
- Environment audit: [regicide_gym_env.py](regicide_gym_env.py)
- Policy and training audit: [card_aware_policy.py](card_aware_policy.py), [streamlined_training.py](streamlined_training.py)
- Actionable fixes and recommendations

---

## Executive summary

Main issues affecting learnability and correctness:

1) Rules mismatches in regicide engine
- Hearts healing inserts cards on top of the Tavern deck instead of under/bottom.
- Exact-kill enemy card is placed at bottom instead of top of the Tavern deck.
- Jester is allowed to be paired with other cards (it must be played alone).
- Animal Companion can illegally combine with multi-card combos (rules allow pairing with exactly one other non-Jester card, or with one other Animal Companion).
- Clubs power stacks once per Club in a combo (effectively 3x, 4x damage) — likely should double once per play when any effective Clubs is present.
- Hearts/Diamonds resolution order is not enforced (Hearts should resolve before Diamonds when both apply in the same play).

2) Environment design issues
- Reward shaping focuses almost exclusively on immediate damage (+0.5 per damage) and boss kills, giving no credit for strategic suit powers (Spades shields, Hearts heals, Diamonds draw). This biases against learning rules-based long-term strategies and favors suboptimal short-term damage.
- Valid actions capped at 30; if more exist, they are truncated. This risks silently dropping high-quality actions and confuses learning.
- Jester “choose next player” is auto-advanced (current+1) rather than a choice. This removes a critical tactical lever.
- Test path references a “vector” observation mode that is not implemented (non-critical but confusing).

3) Policy/training shortcomings
- Plain REINFORCE without entropy regularization or a value function baseline leads to high variance and poor exploration.
- Model capacity is small and action scoring is simplistic for the combinatorial nature of actions.
- No exploration schedule and no entropy bonus; no curriculum; no advantage normalization beyond mean; single-episode updates only.

Addressing the rule bugs plus adding minimal exploration and reward shaping for suit powers should significantly improve learning.

---

## 1) Rules vs implementation (regicide.py)

Reviewed file: [regicide.py](regicide.py)

Findings (severity high → low):

- Hearts “under the Tavern deck” placement is inverted (HIGH)
  - Rules: “Shuffle the discard pile then count out a number of cards ... Place them under the Tavern deck.”
  - Current: _hearts_power extends the Tavern deck at the end (top if you draw with pop()).
  - Effect: healed cards are drawn immediately instead of later, changing game dynamics.

- Exact-kill enemy placement is inverted (HIGH)
  - Rules: Exact damage → place enemy face down on top of the Tavern deck.
  - Current: _defeat_enemy places exact-kill enemy at index 0 (bottom), while draw pops from the end (top).
  - Effect: enemy returns much later instead of immediately, affecting difficulty and resource flow.

- Jester must be played alone (HIGH)
  - Rules: “Jester may be played (always on its own).”
  - Current: _is_valid_combo allows A + Jester pairing (invalid).
  - Effect: illegal actions permitted; learned policy may exploit them.

- Animal Companion (A) pairing constraints (HIGH)
  - Rules: A can be played alone or paired with exactly one other non-Jester card, or paired with one other A. A cannot be part of a multi-card “same number” combo.
  - Current: _is_valid_combo allows A(s) with a valid combo of numbered cards (len(other_cards) > 1).
  - Effect: illegal actions permitted; breaks core game mechanics.

- Clubs doubling likely should apply once per play (MED)
  - Rules text: “Double damage: During Step 3, damage dealt by clubs counts for double.” Example shows single doubling with one Club.
  - Current: `_apply_suit_powers` adds `+ total_attack` per Club, which becomes 3x, 4x, ... when multiple Clubs are in a play.
  - Effect: potentially overpowers combos with multiple Clubs. If your intent is strictly by rules, apply a single doubling if any effective Club is present.

- Hearts before Diamonds order not enforced (LOW)
  - Rules: resolve Hearts before Diamonds when both occur in the same play.
  - Current: suit powers loop in card order; may interleave.
  - Effect: subtle sequencing mismatch.

- Jester choose-next-player (LOW in engine, HIGH in env)
  - Engine returns `phase: 'next_player_choice'` correctly, but the environment autoselects next player. Engine is fine; env reduces tactical choice.

- Attack card buffer behavior (OK)
  - Played cards are kept and discarded only once the enemy is defeated. Jester remains “in play”. Correct.

- Spade shields persistence (OK)
  - Spade shield accumulates and persists until enemy defeated. Correct.

Actionable code fixes for the above are provided in the “Patches” section.

---

## 2) Environment audit (regicide_gym_env.py)

Reviewed file: [regicide_gym_env.py](regicide_gym_env.py)

Strengths:
- Well-structured card-aware observation with:
  - hand indices, enemy card index, compact game_state features, discard pile bit-vector, per-action card indices.
- Correct phase handling (attack/defense) and yield/defense flow aligned with engine.
- Terminal handling and info dictionary are consistent.

Issues impacting learning:

- Reward shaping bias (HIGH)
  - Rewards: +0.5 per damage, +20 per boss kill, -0.1 per step. No positive reinforcement for strategic suit powers:
    - No reward for increasing Spade shield (future survival).
    - No reward for Hearts heals (deck sustainability).
    - No reward for Diamonds draws (resource acquisition).
    - No cost for wasteful defense (using too many high-value cards).
  - Consequence: policy learns to chase immediate damage and often ignores critical, rules-aligned tactics. This alone can explain very poor training outcomes.

- Action cap (MED)
  - `max_actions = 30` and truncation if more exist risks dropping good moves; the learner’s “world” randomly omits actions. In combinatorial games this is harmful.
  - Suggest dynamic top-k pruning by heuristic score (e.g., prefer legal single-card plays + best small combos) or raise limit.

- Jester next-player choice lost (MED)
  - Env auto-selects `(current+1) % num_players`. This removes important tactical choice. Expose this branch as a second-step decision or encode it as a small discrete choice on “next player” when Jester is played.

- Minor: test uses “vector” mode which is not implemented (LOW)
  - Cosmetic; switch to “card_aware”.

---

## 3) Policy and training audit

Reviewed files:
- Policy: [card_aware_policy.py](card_aware_policy.py)
- Trainer: [streamlined_training.py](streamlined_training.py)

Strengths:
- Structured, card-aware policy with embeddings for cards and enemy, plus discard pile encoder.
- Per-action card representation via averaged embeddings is a reasonable baseline.
- REINFORCE pipeline is clean; uses baseline subtraction (mean of returns in episode).

Limitations and recommendations:

- No entropy regularization or exploration schedule (HIGH)
  - Add entropy bonus to policy loss with annealed coefficient (e.g., 0.01 → 0.001).
  - Alternatively use temperature on logits with decay.

- No value function / critic (HIGH)
  - REINFORCE has very high variance. Introduce actor-critic (baseline network). Even a small critic on the encoded context helps a lot.

- Reward shaping missing for suit powers (HIGH)
  - Mirror the game’s long-term incentives:
    - +alpha per Spade shield point added (diminishing returns clip).
    - +beta per Hearts heal card returned under Tavern.
    - +gamma per Diamonds card drawn.
    - Small penalty for over-defending (discard value exceeding required damage by a margin).

- Model capacity and action scoring (MED)
  - Hidden size 32-64 is small for nuanced combinatorial scoring.
  - Consider enabling the commented self-attention for hand.
  - Consider deeper action scorer (e.g., 2 layers of 128 with GELU) and/or include simple hand-context features (e.g., suit counts, high card flags).

- Batch updates / variance reduction (MED)
  - Accumulate N episodes (e.g., 8–32) before update to reduce variance.
  - Normalize advantages per batch.

- Truncation of actions at 30 (MED)
  - See environment section.

- Jester next-player decision not represented (MED)
  - If you restore it as a choice, model needs an auxiliary decision head or a short sub-decision step.

- Minor: logging/diagnostics (LOW)
  - Track policy entropy and average action set size. Helps tune exploration and action enumeration.

---

## Patches (rule fixes)

Below are minimally-invasive patches to align with the written rules and improve learnability.

1) Enforce Jester-alone and Animal Companion pairing rules; fix suit power application ordering and Clubs doubling-once; fix Hearts placement; fix exact-kill placement.

````python
// filepath: [regicide.py](http://_vscodecontentref_/0)
# ...existing code...
    def _is_valid_combo(self, cards: List[Card]) -> bool:
        # Jester must be played alone
        if any(c.value == 0 for c in cards):
            return len(cards) == 1

        if len(cards) == 1:
            # Single non-jester card is fine (including single Animal Companion)
            return True

        # Separate Animal Companions from other cards (excluding Jester which is handled above)
        animal_companions = [c for c in cards if c.value == 1]
        other_cards = [c for c in cards if c.value != 1]

        # Animal Companion rules:
        # - AC can be played alone (handled above)
        # - AC can be paired with exactly one other non-Jester card
        # - AC can be paired with exactly one other AC
        # - AC cannot be added to multi-card numbered combos
        if animal_companions:
            # Exactly two cards total
            if len(cards) == 2:
                # Case: AC + AC
                if len(animal_companions) == 2:
                    return True
                # Case: AC + one non-AC (non-jester)
                if len(animal_companions) == 1 and len(other_cards) == 1 and other_cards[0].value != 0:
                    return True
                return False
            # Any other count with AC present is invalid
            return False

        # No Animal Companions: must be regular combo of same number, total value <= 10
        same_value = len(set(card.value for card in cards)) == 1
        if not same_value:
            return False
        total = sum(card.value for card in cards)
        return total <= 10
# ...existing code...
    def _apply_suit_powers(self, cards: List[Card], total_attack: int):
        # Resolve suit powers at total_attack, with proper ordering and clubs doubling-once behavior.
        # Hearts should resolve before Diamonds when both occur.
        # Suit immunity applies per card.
        # For AC pair of same suit, apply that suit power only once (handled by counting distinct suits for AC cases).

        # Classify play
        has_ac = any(c.value == 1 for c in cards) and len(cards) <= 2
        # Determine effective (non-immune) suits present
        effective_cards = [c for c in cards if not self._is_immune(c)]
        effective_suits = [c.suit for c in effective_cards]

        # Special handling: if this is an AC pairing (len<=2 and at least one AC), reduce duplicate suit to one application
        if has_ac:
            suits_to_apply = set(effective_suits)
            hearts_count = 1 if Suit.HEARTS in suits_to_apply else 0
            diamonds_count = 1 if Suit.DIAMONDS in suits_to_apply else 0
            spades_count = 1 if Suit.SPADES in suits_to_apply else 0
            any_clubs = Suit.CLUBS in suits_to_apply
        else:
            # Regular combos: apply per effective card (each card’s suit power at total_attack)
            hearts_count = sum(1 for c in effective_cards if c.suit == Suit.HEARTS)
            diamonds_count = sum(1 for c in effective_cards if c.suit == Suit.DIAMONDS)
            spades_count = sum(1 for c in effective_cards if c.suit == Suit.SPADES)
            any_clubs = any(c.suit == Suit.CLUBS for c in effective_cards)

        # Resolve Hearts before Diamonds
        for _ in range(hearts_count):
            self._hearts_power(total_attack)
        for _ in range(diamonds_count):
            self._diamonds_power(total_attack)

        # Spades: cumulative shield
        for _ in range(spades_count):
            self.current_enemy.spade_protection += total_attack

        # Clubs: double damage once if any effective club present
        if any_clubs:
            self.current_enemy.damage_taken += total_attack
# ...existing code...
    def _hearts_power(self, value: int):
        if not self.discard_pile:
            return

        random.shuffle(self.discard_pile)
        cards_to_heal = min(value, len(self.discard_pile))
        healed_cards = self.discard_pile[:cards_to_heal]
        self.discard_pile = self.discard_pile[cards_to_heal:]

        # Place under the Tavern deck (bottom). Drawing uses pop() from the end (top),
        # so prepend to put them at the bottom.
        self.tavern_deck = healed_cards + self.tavern_deck
# ...existing code...
    def _defeat_enemy(self):
        # When an enemy is defeated, move all attack-played cards to the discard pile
        if self.attack_cards_buffer:
            self.discard_pile.extend(self.attack_cards_buffer)
            self.attack_cards_buffer = []

        # Place enemy in discard pile or on top of tavern deck (exact kill → top)
        if self.current_enemy.damage_taken == self.current_enemy.health:
            # Top of deck corresponds to end of list (since draw uses pop())
            self.tavern_deck.append(self.current_enemy.card)
        else:
            self.discard_pile.append(self.current_enemy.card)

        # Reset for next enemy
        if self.castle_deck:
            self.current_enemy = Enemy(self.castle_deck.pop(0))
            self.jester_immunity_cancelled = False
            self.attack_cards_buffer = []
        else:
            self.current_enemy = None
# ...existing code...