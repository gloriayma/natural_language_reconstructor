# Observations & analysis (embed→unembed round-trip)

Cool findings from the round-trip + layerwise logit-lens data. Everything here is checkable in
the explorer (https://claude.ai/code/artifact/9663731e-8a84-462e-9d85-a80778103ded, soon
gloria.ma/roundtrip.html) or in `results/<model>/results.{json,csv}`. Numbers below use the
**normed** (logit-lens) variant unless noted.

**Data basis: 11 models** (fleet completed 2026-07-06) — 4 tied (gpt2, gpt2-xl, Qwen2.5-0.5B,
gemma-2-2b), 7 untied (pythia-410m/6.9b, Mistral-7B, OLMo-2-7B, gpt-oss-20b, Qwen2.5-7B/72B).
~140–153 token rows per model. Full panel table in the experiment doc. Items written from the
first 2 models and later revised are marked.

## 1. The headline: tied round-trips, untied doesn't — at all

Across the full panel the split is **perfect**: every tied model round-trips (gpt2 95.8%,
gpt2-xl 98.6%, Qwen2.5-0.5B 99.3%, gemma-2-2b 96.2% — the last at a 256k vocab), and every
untied model doesn't (six at exactly 0.0%; Mistral at 0.7% = one surviving word, `▁between`).
Untied medians sit between the 27th and 69th percentile of their vocabs — around random, with
the extreme cases at the literal bottom (obs. 2). And it's tying, not scale: pythia-410m
(untied, 0.4B) fails identically to Qwen2.5-72B; the tied/untied Qwen siblings (0.5B vs 7B)
share a tokenizer and split cleanly.

So "is the embedding the inverse of the unembedding?" — yes-ish when weights are tied (trivially),
and *not even approximately* when untied.

## 2. Untied is not just "no" — it looks actively anti-identity

The worst self-ranks in the 72B are not random-bad, they are **dead last**: `ing` ranks
152,064 of 152,064; `ed` (of "walked") 152,057; `;` 151,913. Common subword continuations are
ranked at the literal bottom of the vocabulary by the direct embed→unembed path. A plausible
reading: the embedding of a frequent token carries a component that *suppresses* immediately
re-emitting that token (repetition suppression baked into the geometry). Untested hypothesis —
would need norms/cosines across the full vocab to pin down (deliberately not done yet).

## 3. No bigram structure in the direct path

The zero-layer-transformer story ("W_E·W_U approximates bigram statistics") does **not**
describe a trained deep model's direct path: `dog` unembeds to `ĠserviceProvider`, `water` to
`ĠoutFile` — junk code tokens, not ` barks`/` bottle`. Once a model has layers, the direct
path apparently stops carrying next-token structure entirely (the layers do all of it).

## 4. Every model has a "resting" token, and they're weird

Mid-network, the logit lens shows one token dominating top-1 across most inputs:

- gpt2 at L6: `,` is top-1 for **71/143** rows; gpt2-xl rests on `Ġand` (78/143).
- Qwen2.5-72B at L53: a **Thai token** (สามาร) for **89/142** rows.
- **Revised with the full panel**: most models rest on *weird rare tokens*, not glue —
  `ascus` (Qwen-0.5B), `enumi` (gemma — LaTeX!), `Ġbast` (pythia-410m), `Ġstrugg` (Qwen-7B),
  `contador` (OLMo), `▁kennis` (Mistral — Dutch!), `ĠBOTH` (gpt-oss), `Ċ` (pythia-6.9b).
  The GPT-2 family's English-glue resting state is the exception, not the rule (and note it
  holds tied or not — gemma is tied and still rests on a LaTeX fragment).

Caveat: mid-network residuals aren't in the output basis (this is the known logit-lens
weakness the tuned-lens literature addresses), so "attractor" describes the lens view, not
necessarily the computation.

## 5. Where the final prediction crystallizes: opposite ends of the network

First probe at which the model's eventual top-1 token enters the lens top-10:

- **gpt2: layer 1** for 91/143 tokens (64%) — the final answer is visible almost immediately.
- **72B: layers 79–80** for 115/142 tokens (81%) — the final answer appears in the last two
  layers, and before that it can be *anti*-ranked: for `crypt`, the eventual top-1 `os` sits
  around rank **149,000** through L77, then rank 3 at L79 and rank 1 at L80.

**Revised with the full panel — this is NOT a tied/untied or size split.** Median
depth-fraction at which the final top-1 enters the lens top-10: gpt2 8%, gpt2-xl 23%,
pythia-6.9b 22% vs pythia-410m 96%, Qwen-0.5B 96%, Mistral 91%, gpt-oss 88%, OLMo/Qwen-7B
100%, Qwen-72B 99% — and gemma-2-2b at 0%. Two confounds explain most of it:
(a) models whose real answer to a bare word is generic glue (gpt2 `,`, gpt2-xl, pythia-6.9b
`-`/`,`/`.`) look "early" because their answer coincides with their lens resting state;
(b) gemma looks "instant" because its real answer *is the input token* 58% of the time
(obs. 9), which a tied model shows at the embedding. Models with diverse final answers
crystallize at 88–100% of depth regardless of size or tying.

## 6. Which tokens fail the round-trip in the tied model: whitespace & control

gpt2's rare identity failures are all low-content tokens: tab (self-rank 106), NUL byte
(105, top-1 is the unrelated `Ġdestro`), newline (10), comma (5). Barely-trained/degenerate
embeddings don't even self-identify in a tied model. Meanwhile the famous **glitch tokens
(` SolidGoldMagikarp`, ` petertodd`) round-trip perfectly at rank 1** — their pathology is
about training-time behavior, not embedding geometry.

## 7. Odds and ends

- **Raw vs normed barely matters for ranking**: RMSNorm (72B) is nearly rank-invariant;
  gpt2's LayerNorm changes little — and raw is slightly *more* identity-preserving than
  normed (98.6% vs 95.8%).
- **Probabilities differ hugely though**: gpt2's normed round-trip is p≈1.0; raw is p≈0.8 for
  content words but p≈0.009 for `the` — frequent function words have small-norm embeddings.
- **72B special tokens don't self-identify either**: its pad/eos `<|endoftext|>` has emb
  self-rank 136,004.
- **Within-word continuation** (multi-token words, e.g. does `crypt` predict `oc`?): at the
  final layer, 72B top-1s the word's true continuation 23/56 times (37/56 top-10); gpt2 12/56
  (22/56). At layer 1: 72B 0/56 — the one shiny gpt2 case (`crypt`→`oc` rank 1 at L1) is
  the exception (1/56), not the rule.

## 8. The unembedding matrix is not normalized — and that explains the attractors

Measured directly from the weights (rows of `lm_head.weight` / gpt2's tied `wte`):

- **gpt2**: row norms 2.45–6.32 (median 3.95). Smallest norms = the most frequent/whitespace
  tokens (`Ġthe` and `Ċ` are the 0th percentile; `,`, tab, NUL the 2nd) — exactly the tokens
  whose round-trips are weak or fail, since a tied token's self-logit is its squared norm.
  ` SolidGoldMagikarp` is an ordinary 53rd percentile (more evidence glitchiness ≠ geometry).
- **Qwen2.5-72B**: row norms 0.40–1.90 (median 0.88). The mid-network Thai attractor token
  (obs. 4) has norm 1.314 = **100th percentile of all 152k rows** — the "resting prediction"
  is literally the vocabulary's loudest unembedding row, winning the dot-product race whenever
  the residual aligns strongly with nothing. The never-trained padded rows (151,665+) sit at
  the floor, ~0.42 (init scale).
- Caveat: the head consumes the final-norm output, whose learned per-dim gain folds into the
  effective per-token gain; raw row norms already account for the observed phenomena though.
- **The capture condition** (user-derived): row j steals token t's round-trip iff
  cos(w_j, w_t) · ‖w_j‖ > ‖w_t‖ — capture basins grow with row norm. Verified in gpt2 weights:
  tab and NUL (norm 3.09, undertrained) are both captured by the same three anomalous
  high-norm rows (`conservancy`, `ModLoader`, `BuyableInstoreAndOnline`; norms 5.2–5.7,
  cos ≈ 0.6 to the whitespace cluster: 0.64×5.19 = 3.32 > 3.09). Trained tokens are safe
  because self-cos = 1.0 is a huge moat (a cos-0.6 rival needs 1.67× your norm; gpt2's whole
  norm range is 2.6×). Mid-network, no token has a large cosine to the residual, so ranking
  degenerates to ~pure norm order — the attractor regime is this argument taken to its limit.

## 9. gemma-2-2b echoes: its true output for a bare word is often the word itself

For 76/131 token rows (58%), gemma-2-2b's actual model output (final_logits top-1) is the
input token again — given `dog` with no context it predicts ` dog`-ish continuations of
itself. No other model in the panel does this (gpt2 1/143, pythia-6.9b 2/145, Qwen-0.5B
5/142 — those predict punctuation/glue instead: gpt2's modal answer is `,` 40/143, Qwen-0.5B's
is `Ġ=` 18/142). Combined with tying, gemma's echo makes its "crystallization depth" 0% — the
answer is visible at the embedding because the answer is the question.

## Caveats

Two models; one run (deterministic); ~12–17 probe layers not all layers; bare-word
tokenization (no leading space); logit-lens basis caveat above. No corpus bigram table was
compared against (obs. 3 is eyeball-level).

## History

- **2026-07-05**: First version, from gpt2 + Qwen2.5-72B v3 data (9 models pending).
- **2026-07-05 (later)**: Added obs. 8 (unembedding row norms measured from weights; attractor
  = max-norm row confirmed), prompted by user questions about invertibility and normalization.
- **2026-07-05 (later still)**: Added the capture condition to obs. 8 after the user pointed
  out high-norm rows should capture more round-trips — confirmed in weights (tab/NUL captured
  by the same three high-norm anomalous rows).
- **2026-07-06**: Fleet complete (11 models). Obs. 1 upgraded to the full-panel perfect split;
  obs. 4 revised (attractors are usually weird rare tokens; GPT-2-family glue is the
  exception); obs. 5 revised (crystallization is not a tied/size split — glue-answer and echo
  confounds); obs. 9 added (gemma echoes bare words 58% of the time).
