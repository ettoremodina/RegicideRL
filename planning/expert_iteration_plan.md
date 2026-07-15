# Regicide RL Implementation Plan
## AlphaZero-style Expert Iteration with ISMCTS

This document tracks the information extracted from the reference papers necessary to implement the planned features for the Regicide bot.

### Phase 1: Baseline - PIMC (Perfect Information Monte Carlo) ✅ IMPLEMENTED
*   **Concept**: Determinization. At each decision point, sample $N$ possible states (hidden card orderings) from the current information set. 
*   **Execution**: Run a standard deterministic planner (e.g., UCT or a fast heuristic rollout) independently on each sampled state. Average the action values across all determinizations to select the best move.
*   **Reference**: Discussed in *Information Set Monte Carlo Tree Search* (Cowling et al.) and *Lower Bounding Klondike Solitaire with Monte-Carlo Planning* (Bjarnason et al.) as a baseline.
*   **Weakness to Fix Later**: "Strategy fusion" (the agent incorrectly assumes it can make different choices for different determinizations in the future) and "non-locality".
*   **Result**: 3.10/12 enemies defeated avg (d=20), vs 2.35/12 for heuristic baseline.

### Phase 2: Upgrade to ISMCTS (Information Set Monte Carlo Tree Search) ✅ IMPLEMENTED
*   **Concept**: Instead of separate trees for each determinization, build a *single* tree where nodes represent information sets rather than exact game states.
*   **Execution**: 
    1.  At each iteration, sample a single determinization.
    2.  Descend the single ISMCTS tree. At opponent nodes, only branches compatible with the sampled determinization are valid.
    3.  Select actions using UCB, adapted for "subset-armed bandits" (because the available actions depend on the determinization).
    4.  Update node statistics (visits and rewards) in the shared tree.
*   **Reference**: Pseudocode and detailed algorithms (SO-ISMCTS and MO-ISMCTS) are provided in *Information Set Monte Carlo Tree Search* (Cowling et al.).
*   **Result**: 3.15/12 enemies defeated avg (i=200), slightly ahead of PIMC.

### Phase 3: Tune the Search
*   **Heuristic Rollouts**: Use the existing fast heuristic policy instead of random actions during the rollout phase to reduce variance and improve value estimation.
*   **Trade-offs**: Sweep and tune the number of determinizations versus tree depth/simulations per decision.
*   **Tree Reuse**: Retain the relevant subtree after making a real move instead of discarding the entire tree.
*   **Parallelization**: ISMCTS iterations on different determinizations can be highly parallelized.

### Phase 4: Benchmark Standalone Search
*   Evaluate the ISMCTS agent over thousands of games against the Phase 1 PIMC baseline and the pure heuristic baseline to establish statistical significance.

### Phase 5: Expert Iteration (Search Teaches the Network)
*   **Concept**: Use the ISMCTS as an "Expert" (System 2) to train a neural network "Apprentice" (System 1). The network then guides future ISMCTS searches.
*   **Execution (AlphaZero / ExIt loop)**:
    1.  **Self-Play**: Run ISMCTS to play games. Log `(state, action-visit distribution, outcome)`.
    2.  **Training**: Train a dual-headed neural network to predict the action-visit distribution (Policy Head) and the game outcome (Value Head) using the logged data.
    3.  **Bootstrapping**: Replace the heuristic rollout with the neural network. Use the Policy Head's prior probabilities to bias the UCB action selection formula during the tree search (e.g., PUCT formula).
*   **Reference**: Discussed extensively in *Thinking Fast and Slow with Deep Learning and Tree Search* (Anthony et al.) and *Mastering the game of Go with deep neural networks* (Silver et al.). 

### Phase 6: Final Evaluation
*   Conduct large-sample win-rate evaluations (e.g., 1000+ games) with confidence intervals.
*   Perform ablation studies: Search only vs. Network only vs. Combined.
