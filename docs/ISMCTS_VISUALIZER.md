# ISMCTS decision visualizer

An animated, data-driven replay of one real ISMCTS decision. Nothing in the page
is illustrative: every node, counter and reward comes from a search the agent
actually ran, so the page doubles as a debugging tool for the search itself.

The companion prose explanation is [`ismcts_explained.html`](ismcts_explained.html).

## What the animation shows

The page replays the four phases of every iteration, and is built around the
three properties that separate ISMCTS from plain UCT:

1. **Determinization.** The left panel reshuffles the hidden tavern and castle
   cards at the start of each iteration. The tree is *not* reset — that is the
   difference from PIMC, which builds a separate tree per sampled world.
2. **Subset-armed selection.** At each visited node only the children legal in
   the current world compete; the others are greyed out and their counters
   freeze. Deep nodes accumulate children across many worlds, so it is normal to
   see, for example, 8 of 21 known children competing in a given iteration.
3. **Availability counting.** Every node carries `v` (visits) and `a`
   (availability). Availability rises for each legal child whether or not it was
   selected, and it is the denominator of the ISMCTS UCB formula — this is what
   stops rarely-legal actions from looking permanently unexplored.

The bottom panel ranks the root actions, drawing the visit count inside the
availability count. It is the practical payoff view: how quickly, and how far,
the search separates the candidate moves.

The page fits one screen without scrolling and stays nearly wordless: card and
action names, one small caption per box, and a phase card in the top-left corner
naming the current step in one line. That card has a fixed height so switching
phase never reflows the panels below it. Counters are encoded as bar lengths, mean
rewards as a red-amber-green colour, UCB scores as the radius of the pip above
each competing child. Exact numbers are available on hover, as native tooltips.

Each phase claims attention in a different way, so the eye knows where to look
without reading:

| phase | what it does on screen |
| --- | --- |
| determinize | the deck panel is marked up in gold, the tree drops to 50% |
| select | the taken edge draws itself, a marker travels down it, a ring pulses over the winning UCB pip |
| expand | a green ring bursts out of the node just created |
| rollout | a pencil-dashed comet leaves the leaf, ending in a dot sized by playout length and coloured by outcome |
| backpropagate | a green wash rises from the bottom of the board while the path flows upward |

## Palette

The page is printed matter, sharing its inks with the charts under `docs/site`:
paper `#FBF9F4` on a `#E5E0D3` desk, text `#101A24`, with a fractal-noise grain
over both. Colour carries meaning rather than decoration:

- **deep green `#123D3A`** — permanent tree ink: traversed edges, visit bars,
  the live node;
- **slate `#6B8498`** — the pencil of the throwaway playout, and the candidate
  pips;
- **gold `#D0A84C`** — the highlighter over the world being sampled;
- **brick `#B64038`** — losing outcomes and enemy health only.

Node rewards run brick → gold → deep green. Nothing glows: emphasis is carried
by ink weight, hairline keylines and physical shadows, and a child that is
illegal in the current world is drawn *unprinted* — dashed outline, no fill —
rather than faded, which on paper would simply disappear.

## Pipeline

```
ISMCTSAgent ──hooks──▶ ISMCTSTracer ──JSON──▶ trace_ismcts.py ──inline──▶ HTML page
```

| Component | Path | Role |
| --- | --- | --- |
| Tracer | [`agents/ismcts_trace.py`](../agents/ismcts_trace.py) | Records one decision as replayable data |
| Agent hooks | [`agents/ismcts_agent.py`](../agents/ismcts_agent.py) | Six guarded calls, inert when `tracer is None` |
| Driver | [`scripts/trace_ismcts.py`](../scripts/trace_ismcts.py) | Plays to a mid-game position, records, renders |
| Viewer template | [`scripts/templates/ismcts_visualizer.html`](../scripts/templates/ismcts_visualizer.html) | Standalone player, trace inlined at build time |
| Configuration | `ismcts_trace` in `config.yaml` | Seed, filters, budget, recording limits, paths |

The agent is unaffected when no tracer is attached: the hooks are guarded by
`if self.tracer is not None` and the default stays `None`, including for the
benchmark and experimental-report agents.

## Trace format

The trace never stores counters, only the increments that produced them, so a
viewer reproduces `visit_count` and `availability_count` with the same
arithmetic the agent used. `tests/test_ismcts_trace.py` asserts that the replay
matches the agent's own tree exactly.

```jsonc
{
  "meta":     { "seed": 20260803, "n_iterations": 500, "exploration_constant": 1.414, ... },
  "decision": { "hand": ["A♣", ...], "enemy": {...}, "legal_actions": [...], "best_action": 4,
                "root_children": [ { "id": 1, "visits": 45, "availability": 497, ... } ] },
  "nodes":    [ { "id": 1, "parent": 0, "action": 4, "label": "4♥", "depth": 1 } ],
  "iterations": [
    { "i": 0,
      "determinization": { "tavern_top": ["7♦", ...], "castle": ["J♦", ...] },
      "steps": [ { "node": 0, "child": 1, "expanded": true,
                   "available": [ /* node ids whose availability rose */ ],
                   "untried": 12, "legal": 13, "scores": { "1": 17.76 }, "defense": false } ],
      "rollout": { "depth": 24, "reward": -0.1 },
      "reward": -0.1,
      "path": [0, 1] }
  ]
}
```

`scores` holds the UCB1 values actually compared during selection, so the page
never recomputes the agent's arithmetic; it displays it.

## Recording cost and limits

Recording is bounded by `record.max_iterations`, `record.tavern_preview` and
`record.max_depth`. Keep `record.max_iterations` equal to `search.n_iterations`
so the animation and the final visit counts describe the same search.

The search budget drives everything at once — tree size, playback length, page
weight and the quality of the decision. Measured on the reference position
(seed 20260803, 13 legal actions, `max_depth` 5):

| iterations | nodes | descent steps / iteration | phases | root visit spread (C=1.414) | root visit spread (C=14.14) | JSON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 101 | 1.97 | ~500 | 33% | 12% | 68 KB |
| 300 | 301 | 2.69 | ~1 700 | 31% | 8% | 246 KB |
| 500 | 501 | 3.03 | ~3 000 | 33% | 8% | 441 KB |
| 1000 | 983 | 3.54 | ~6 500 | 46% | 6% | 970 KB |
| 2000 | 1843 | 3.91 | ~13 800 | 52% | 7% | 2 071 KB |

Each iteration expands exactly one node, so nodes track iterations until the
tree starts running into terminal states inside itself (hence 1843 rather than
2000). Cost is linear in iterations, roughly 1 KB and 6–7 animation phases each.

Below roughly 500 iterations the decision itself is unreliable: the most visited
root action is not the one with the best mean reward. From 500 up the two
criteria agree, which makes 500 the smallest honest budget to visualize and the
default in `config.yaml`.

The traced decision is chosen by the filters in `config.yaml`: it must come
after `warmup_decisions` heuristic moves and offer between `min_legal_actions`
and `max_legal_actions` legal actions, which keeps the tree drawable. A seed
where no such decision appears raises a clear error instead of producing an
unreadable page.

## Reading the exploration constant

The configured value for the trace is `1.414` (√2). The benchmarked ISMCTS agent
uses `14.14`, and the visualization makes the consequence visible: at 500–3000
iterations the exploration term is over an order of magnitude larger than the
spread in mean rewards, so root visits stay almost uniform and the final choice
by visit count is close to arbitrary. Re-record with `--exploration 14.14` to
see it directly.
