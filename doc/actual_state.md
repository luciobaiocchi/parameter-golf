# Parameter Golf Live AI Commentary

*Auto-updated every ~10 minutes. Tracking techniques, trends, idea lineage, and explaining concepts for the community.*


*Last updated: Mar 31, 11:03 PM PT*

---

## The Competition at a Glance

**Goal:** Train the best language model that fits in a **16MB artifact**, training in under **10 minutes on 8xH100s**. Evaluated by **compression** of the FineWeb validation set, measured in **bits per byte (BPB)** — lower is better. Tokenizer-agnostic. Baseline: **1.2244 BPB**.

<details>
<summary><strong>What does "compression" mean here?</strong></summary>

BPB (bits per byte) measures how many bits your model needs to encode each byte of text. A model that perfectly predicts every next character needs zero bits — it already "knows" what comes next. A model with no understanding of language needs the maximum (~8 bits per byte).

A model's cross-entropy loss IS its compression rate. Shannon proved in 1948 that prediction and compression are mathematically equivalent — a model that predicts well compresses well, and vice versa. The competition measures the compression side of that equivalence.

This framing matters because it legitimizes approaches beyond pure language modeling: sliding window eval improves compression by giving more context. Backward-looking TTT adapts to already-scored tokens for better compression. These are valid compression strategies.

There is no separate held-out test set — the FineWeb validation set is the fixed evaluation target. However, val tokens cannot be stored in the artifact (paid prefix ruled out), and pre-eval adaptation on val data is also ruled out. Only backward-looking TTT (adapting on tokens already graded) is permitted.

"Tokenizer-agnostic" means BPB normalizes across tokenizers. A bigger vocabulary uses fewer tokens but more bits per token — BPB cancels that out, measuring compression of raw bytes regardless of how they're tokenized.

</details>

**Record submission requirements:** Artifact ≤16,000,000 bytes (code + compressed model). Training ≤10 min on 8xH100 SXM. Evaluation ≤10 min (separate budget). No network calls. New SOTA records must beat the current best by ≥0.005 nats at p < 0.01 significance (typically 3 seeds). Evaluation methods are unrestricted — any sequence length, sliding window, etc. are fair game. Test-time training is allowed only on already-evaluated tokens (backward-looking); pre-eval adaptation on val data is ruled out.

The competition launched Mar 18. The official SOTA is now **1.1147 BPB** (#1019, @abaybektursun, Mar 25 — merged, AR Self-Gen GPTQ + XSA-all). The best pending submission overall is **#1184 (@icryo, 0.9485 BPB, 3-seed)** using Scylla tokenizer (998-vocab TokenMonster) + Full GPTQ + XSA-all. The best standard-tokenizer submission is **#1176 (@bigbag, 1.0914, 3-seed)** with QK-Gain 4.0 + Muon-TTT + SLOT. A new n-gram backoff cache submission #1185 (@skoustav35, 0.9641) claims proper Laplace normalization. @abaybektursun's #1105 (1.1125, CUTLASS EVT + MLP 3.5x + mixed int5/int6) narrowly misses the 0.005-nat record threshold vs the new SOTA. @dentity007 submitted 7 PRs addressing README wishlist items (JEPA, text diffusion, H-Net, Universal Transformer, SSM, megakernels, random linear maps) — all proof-of-concept stage. The SOTA shift from 1.1194→1.1147 means 4 former record submissions (#728, #1060, #1099, #1130) no longer meet the 0.005-nat threshold.

![Best Pending BPB Over Time](https://quickchart.io/chart?w=800&h=400&bkg=white&c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22datasets%22%3A%5B%7B%22label%22%3A%22Official%20Leaderboard%22%2C%22data%22%3A%5B%7B%22x%22%3A%222026-03-18T08%3A30%3A00%22%2C%22y%22%3A1.2244%7D%2C%7B%22x%22%3A%222026-03-18T10%3A41%3A00%22%2C%22y%22%3A1.2197%7D%2C%7B%22x%22%3A%222026-03-18T10%3A41%3A00%22%2C%22y%22%3A1.2147%7D%2C%7B%22x%22%3A%222026-03-18T13%3A57%3A00%22%2C%22y%22%3A1.206%7D%2C%7B%22x%22%3A%222026-03-18T15%3A36%3A00%22%2C%22y%22%3A1.1925%7D%2C%7B%22x%22%3A%222026-03-19T00%3A15%3A00%22%2C%22y%22%3A1.1502%7D%2C%7B%22x%22%3A%222026-03-19T16%3A55%3A00%22%2C%22y%22%3A1.1428%7D%2C%7B%22x%22%3A%222026-03-20T09%3A25%3A00%22%2C%22y%22%3A1.1307%7D%2C%7B%22x%22%3A%222026-03-20T16%3A10%3A00%22%2C%22y%22%3A1.1271%7D%2C%7B%22x%22%3A%222026-03-21T14%3A15%3A00%22%2C%22y%22%3A1.1248%7D%2C%7B%22x%22%3A%222026-03-22T07%3A43%3A00%22%2C%22y%22%3A1.1228%7D%2C%7B%22x%22%3A%222026-03-23T05%3A00%3A00%22%2C%22y%22%3A1.1194%7D%5D%2C%22borderColor%22%3A%22%232563eb%22%2C%22backgroundColor%22%3A%22rgba%2837%2C99%2C235%2C0.1%29%22%2C%22fill%22%3Afalse%2C%22pointRadius%22%3A4%2C%22pointBackgroundColor%22%3A%22%232563eb%22%2C%22lineTension%22%3A0.2%2C%22borderWidth%22%3A2%7D%2C%7B%22label%22%3A%22Best%20Pending%20%28incl.%20n-gram%20cache%29%22%2C%22data%22%3A%5B%7B%22x%22%3A%222026-03-23T05%3A00%3A00%22%2C%22y%22%3A1.1194%7D%2C%7B%22x%22%3A%222026-03-24T00%3A00%3A00%22%2C%22y%22%3A1.1162%7D%2C%7B%22x%22%3A%222026-03-25T04%3A35%3A00%22%2C%22y%22%3A1.024%7D%2C%7B%22x%22%3A%222026-03-25T07%3A58%3A00%22%2C%22y%22%3A0.9674%7D%2C%7B%22x%22%3A%222026-03-25T10%3A53%3A00%22%2C%22y%22%3A0.9625%7D%2C%7B%22x%22%3A%222026-03-25T14%3A26%3A00%22%2C%22y%22%3A0.937%7D%2C%7B%22x%22%3A%222026-03-25T14%3A58%3A00%22%2C%22y%22%3A0.9258%7D%2C%7B%22x%22%3A%222026-03-25T15%3A26%3A00%22%2C%22y%22%3A0.6683%7D%2C%7B%22x%22%3A%222026-03-25T18%3A41%3A00%22%2C%22y%22%3A0.6567%7D%2C%7B%22x%22%3A%222026-03-25T19%3A01%3A00%22%2C%22y%22%3A0.5466%7D%2C%7B%22x%22%3A%222026-03-25T20%3A33%3A00%22%2C%22y%22%3A0.4416%7D%2C%7B%22x%22%3A%222026-03-25T21%3A20%3A00%22%2C%22y%22%3A0.2952%7D%2C%7B%22x%22%3A%222026-03-26T01%3A14%3A00%22%2C%22y%22%3A0.1663%7D%2C%7B%22x%22%3A%222026-03-26T05%3A20%3A00%22%2C%22y%22%3A0.1434%7D%2C%7B%22x%22%3A%222026-03-26T09%3A45%3A00%22%2C%22y%22%3A0.1181%7D%2C%7B%22x%22%3A%222026-03-26T10%3A05%3A00%22%2C%22y%22%3A0.0935%7D%2C%7B%22x%22%3A%222026-03-27T00%3A12%3A00%22%2C%22y%22%3A0.0887%7D%2C%7B%22x%22%3A%222026-03-27T04%3A21%3A00%22%2C%22y%22%3A0.0498%7D%2C%7B%22x%22%3A%222026-03-27T09%3A03%3A00%22%2C%22y%22%3A0.0165%7D%2C%7B%22x%22%3A%222026-03-29T04%3A51%3A00%22%2C%22y%22%3A0.9693%7D%2C%7B%22x%22%3A%222026-03-29T06%3A15%3A00%22%2C%22y%22%3A1.1123%7D%2C%7B%22x%22%3A%222026-03-29T16%3A03%3A00%22%2C%22y%22%3A0.4961%7D%2C%7B%22x%22%3A%222026-03-29T18%3A10%3A00%22%2C%22y%22%3A1.1086%7D%2C%7B%22x%22%3A%222026-03-29T19%3A58%3A00%22%2C%22y%22%3A0.4027%7D%2C%7B%22x%22%3A%222026-03-30T05%3A00%3A00%22%2C%22y%22%3A1.1099%7D%2C%7B%22x%22%3A%222026-03-30T18%3A55%3A00%22%2C%22y%22%3A1.0806%7D%2C%7B%22x%22%3A%222026-03-31T06%3A22%3A00%22%2C%22y%22%3A1.1015%7D%2C%7B%22x%22%3A%222026-03-31T09%3A48%3A00%22%2C%22y%22%3A1.0962%7D%2C%7B%22x%22%3A%222026-03-31T16%3A01%3A00%22%2C%22y%22%3A0.9485%7D%2C%7B%22x%22%3A%222026-03-31T16%3A48%3A00%22%2C%22y%22%3A0.9641%7D%2C%7B%22x%22%3A%222026-04-01T00%3A46%3A00%22%2C%22y%22%3A1.1063%7D%2C%7B%22x%22%3A%222026-04-01T04%3A48%3A00%22%2C%22y%22%3A1.1064%7D%2C%7B%22x%22%3A%222026-04-01T05%3A42%3A00%22%2C%22y%22%3A1.1108%7D%5D%2C%22borderColor%22%3A%22%2316a34a%22%2C%22backgroundColor%22%3A%22rgba%2822%2C163%2C74%2C0.1%29%22%2C%22fill%22%3Atrue%2C%22pointRadius%22%3A5%2C%22pointBackgroundColor%22%3A%22%2316a34a%22%2C%22lineTension%22%3A0.2%2C%22borderWidth%22%3A2%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22BPB%20Progression%3A%20Official%20vs%20Pending%22%2C%22fontSize%22%3A14%7D%2C%22scales%22%3A%7B%22xAxes%22%3A%5B%7B%22type%22%3A%22time%22%2C%22time%22%3A%7B%22unit%22%3A%22day%22%2C%22displayFormats%22%3A%7B%22day%22%3A%22MMM%20D%22%7D%7D%2C%22gridLines%22%3A%7B%22display%22%3Atrue%7D%7D%5D%2C%22yAxes%22%3A%5B%7B%22ticks%22%3A%7B%22min%22%3A0.9%2C%22max%22%3A1.23%2C%22stepSize%22%3A0.02%7D%2C%22scaleLabel%22%3A%7B%22display%22%3Atrue%2C%22labelString%22%3A%22BPB%20%28lower%20%3D%20better%29%22%7D%7D%5D%7D%2C%22legend%22%3A%7B%22display%22%3Atrue%7D%2C%22annotation%22%3A%7B%22annotations%22%3A%5B%7B%22type%22%3A%22line%22%2C%22mode%22%3A%22horizontal%22%2C%22scaleID%22%3A%22y-axis-0%22%2C%22value%22%3A1.1147%2C%22borderColor%22%3A%22rgba%28239%2C68%2C68%2C0.5%29%22%2C%22borderWidth%22%3A2%2C%22borderDash%22%3A%5B6%2C3%5D%2C%22label%22%3A%7B%22enabled%22%3Atrue%2C%22content%22%3A%22Official%20SOTA%201.1147%22%2C%22position%22%3A%22left%22%2C%22backgroundColor%22%3A%22rgba%28239%2C68%2C68%2C0.8%29%22%2C%22fontSize%22%3A10%7D%7D%5D%7D%7D%7D)

*Blue = official leaderboard (1.2244 → 1.1147). Green = pending frontier (now invalidated n-gram submissions removed). Red dashed = official SOTA (1.1147, #1019).*


---

## Official Leaderboard (Top 5)


| Rank | Score | Author | Key Techniques | PR |
|------|-------|--------|---------------|-----|
| 1 | **1.1147** | @abaybektursun | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 on #549 stack | [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| 2 | **1.1194** | @sanjeevmadhav | LeakyReLU² + Legal Score-First TTT + Parallel Muon on #414 stack | [#549](https://github.com/openai/parameter-golf/pull/549) |
| 3 | **1.1228** | @signalrush | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 | [#414](https://github.com/openai/parameter-golf/pull/414) |
| 4 | **1.1248** | @jfprincz | 11L Partial RoPE + LN Scale + EMA + XSA4 | [#315](https://github.com/openai/parameter-golf/pull/315) |
| 5 | **1.1271** | @jfprincz | 11L XSA4 + EMA + Int6 MLP3x | [#287](https://github.com/openai/parameter-golf/pull/287) |

**New official SOTA: 1.1147** (#1019, merged Mar 25). Best pending: #1184 (0.9485, Scylla tokenizer), #1185 (0.9641, n-gram backoff cache), #753 (0.9625, Podracing II). Best standard tokenizer: #1176 (1.0914, QK-Gain + TTT + SLOT). Four former records (#728, #1060, #1099, #1130) downgraded — no longer clear the 0.005-nat threshold against the new SOTA. Tables below ↓

## Pending: Meets Record Requirements


Record-eligible submissions only. Pre-eval TTT entries excluded per @0hq ruling on [#402](https://github.com/openai/parameter-golf/issues/402) — only backward-looking (score-first, single-pass) TTT is allowed. Official SOTA: **1.1147 BPB** (#1019, @abaybektursun — AR Self-Gen GPTQ + XSA-all + Legal TTT + Parallel Muon, updated Mar 24).


**Top 5 record-eligible** (26 total — full table in collapsible below):


| BPB | Author | Techniques | PR |
|-----|-----|-----|-----|
| **0.4027** | @michaelwinczuk | **Swarm-Designed Causal BackoffNgramMixer.** Orders 2-10, 4M hash buckets, entropy-adaptive alpha, causal sequential chunk scoring (score-first, update-after). Full-vocab mixture distribution. Neural baseline 1.1245. MTP heads=2, LeakyReLU(0.75)², Parallel Muon. Beats #803 (0.4416) by 0.039. No TTT. Std=0.0015. | [#1094](https://github.com/openai/parameter-golf/pull/1094) |
| **0.4416** | @pentxayc | **Complementary Training** — tokens predictable by bigram stats get lower loss weight during training. Model specializes on what n-grams can't predict, enabling higher eval-time n-gram alpha (20-75%). + Backoff N-gram Mixer + VRL + XSA-4. Std=0.0001. | [#803](https://github.com/openai/parameter-golf/pull/803) |
| **0.4961** | @newjordan | **Bandit: ClownCar Crawler + Cubric Ngram9.** ClownCar crawler (4 flat + 1 crawler x4 loops, Frugendorff) + X-WING n-gram oracle (shared tables, 3D Cubric 54-cell warm-start, entropy-adaptive alpha 0.20-0.75, order-9). GPTQ-int6+zstd ~9.3MB. Pure neural baseline (SW BPB): 1.1867. Std=0.0003. | [#1083](https://github.com/openai/parameter-golf/pull/1083) |
| **0.5466** | @travispchen | **Order-Adaptive Entropy Gating + BackoffNgramMixer + Drift-Free TTT.** Builds on #779 with per-order entropy thresholds from #774. Sub-0.55 BPB. Std=0.0010. | [#798](https://github.com/openai/parameter-golf/pull/798) |
| **0.5644** | @newjordan | **X-WING: Shared N-gram Tables** — all 8 GPU ranks update tables with same tokens (full 62M-token view). Cubric per-order adaptive alpha. Std=0.0006. | [#800](https://github.com/openai/parameter-golf/pull/800) |


Also notable: #795 (0.8881) | #788 (0.9059) | #782 (0.9362) | #774 (0.9370) | #1184 (0.9485) + 16 more


## Pending: Not Yet Validated


Submissions with competitive BPB that haven't yet demonstrated statistical significance.


**Top 5 not-yet-validated** (88 total — full table in collapsible below):


| BPB | Author | Techniques | PR |
|-----|-----|-----|-----|
| **0.0180** | @sofiabod | Packed Causal N-gram + Dirichlet Backoff (0.0180). Post-sweep. Normalization status unclear. | [#1056](https://github.com/openai/parameter-golf/pull/1056) |
| **0.0905** | @vimeto | **Seed-Regenerated Random Model + Incremental N-gram Cache.** Model weights generated from seed (not trained) — neural baseline 1.503 BPP. All compression from n-gram cache. 1-seed only, run on MI250X (not H100). Pending H100 validation + 2 more seeds. | [#1095](https://github.com/openai/parameter-golf/pull/1095) |
| **0.1130** | @sofiabod | **Single-Pass Packed N-gram + Dirichlet CTW** (0.1130). Post-sweep submission. Normalization status unclear — Dirichlet CTW may handle it correctly. | [#1030](https://github.com/openai/parameter-golf/pull/1030) |
| **0.4311** | @Naazimsnh02 | Complementary Training + Backoff N-gram Mixer + TTT (0.4311). Post-sweep. | [#1033](https://github.com/openai/parameter-golf/pull/1033) |
| **0.6364** | @Naazimsnh02 | Depth Recurrence + Multi-Order N-gram Backoff (0.6364). | [#808](https://github.com/openai/parameter-golf/pull/808) |


26 record-eligible + 88 unvalidated | Official SOTA: **1.1147** (#1019) | Full tables in collapsibles below ↓

*Note: The full "All Pending Validated" table below contains the pre-n-gram-cache entries. N-gram/Hedge Mixer submissions still open (#702, #715, #706, #700) and #728 is tracked in the Not Yet Validated table (downgraded when SOTA moved to 1.1147).*

## Untried Combinations

Ranked by expected value (likely gain times probability of working), grounded in competition ablation data:

**Tier 1 — Highest expected value (n-gram cache extensions)**

- **N-gram cache + stronger neural base.** The top n-gram submissions (#803 at 0.4416, #798 at 0.5466) show that training-eval co-optimization is the next frontier (XSA-all, VRL, GA). Combining the best neural base (#609's XSA-all + Full GPTQ + Selective Pruning stack, with GPTQ in training budget) with multi-order backoff + entropy-adaptive alpha could push below 0.30. #779's ablation shows neural-only at 1.1363 dropping to 0.6712 with the BackoffNgramMixer alone. A stronger neural base would push even lower.
- **GEPA + n-gram cache.** GEPA's neural-only frontier (#628: 1.0983 on 4xA100) plus n-gram backoff could target sub-0.95. 8xH100 record-eligible GEPA still untried (~1.116-1.120 projected at 7k steps, pre-n-gram).
- **Context Tree Weighting (CTW) instead of heuristic alpha.** The current top n-gram submissions use hand-tuned or entropy-adaptive alpha to mix n-gram orders. CTW (Willems et al.) provides Bayesian-optimal weighting over all context tree models up to a given depth — provably minimax optimal for tree sources. Replaces heuristic with theory. Zero artifact cost. **#1084 tested depth-4 CTW: +0.005 BPB worse, 46 min eval — negative result.** Original estimate questionable. Moderate complexity (tree data structure).
- **Logistic-domain mixing.** Current submissions use linear interpolation: `alpha*p_ngram + (1-alpha)*p_neural`. PAQ-style compressors use log-odds space mixing, which handles extreme probabilities better. A one-line change. **Est. 0.002-0.005 BPB.** Trivial complexity.
- **Adaptive stride (entropy-guided two-pass).** First pass with stride=64 scores all tokens and records per-token entropy. Second pass re-evaluates high-entropy regions with smaller stride (16-32). Targets compute where it helps most. Backward-looking, zero artifact cost. **Est. 0.005-0.015 BPB.** Low-moderate complexity.
- **Fixed-Share Hedge (non-stationary expert tracking).** #700's Hedge algorithm assumes stationary expert quality. Fixed-Share (Herbster & Warmuth) allows "switching" between experts — important because FineWeb contains diverse content types (code, prose, tables). Zero artifact cost. **Est. 0.003-0.008 BPB over standard Hedge.** Low complexity (one parameter: switching rate).

- **PPMII-style escape estimation** (Shkarin 2001). Replace heuristic backoff with principled escape probabilities + information inheritance (new context nodes inherit counts from parent) + full exclusions (symbols already assigned probability at order-k excluded from order-(k-1)). The theoretically optimal version of what the current BackoffNgramMixer approximates. 40 years of compression research refinement. **Est. 0.01-0.03 BPB.** Medium complexity.
- **Match model (longest-match prediction).** Instead of fixed high-order n-grams (8, 9, 10+), find the longest match anywhere in previously-scored data and predict based on what followed that match. Used by LPAQ/PAQ. Captures arbitrarily long repeated contexts without exponential memory cost. Complements rather than replaces multi-order backoff. **Est. 0.005-0.01 BPB.** Medium complexity.
- **Sparse/skip-gram context models.** Use non-contiguous positions (e.g., tokens at -1, -3, -5) as context. Captures patterns with intervening variable content (HTML tags, code indentation, sentence structures). Multiple sparse models with different gap patterns. Zero additional memory per context — just hash different positions. Especially effective on FineWeb's structured web text. **Est. 0.005-0.015 BPB.** Low complexity.
- **Cascaded SSE stages (3-5 chained APMs).** Rather than a single SSE post-processing step, chain multiple Adaptive Probability Map stages with progressively higher-order contexts. Each stage corrects residual biases from the previous one. Used by PAQ/PAQAR. **Est. 0.005-0.015 BPB.** Low complexity.

- **Complementary distillation** — add `-lambda * KL(P_ngram || P_model)` to the training loss. Pushes the neural model to explicitly diverge from the n-gram distribution at every token, not just binary loss reweighting. Based on "N-gram Is Back" (Li et al., EMNLP 2022) residual learning framework. Smooth, differentiable. **Est. 0.01-0.03 BPB over current Complementary Training.** Low complexity.
- **Three-tier token weighting** — extend Complementary Training beyond binary easy/hard. Down-weight tokens predictable by bigrams (n-gram handles them), AND down-weight "noise" tokens (random proper nouns, typos) that neither model will ever predict well. Concentrate gradient on the learnable frontier. Based on Token Weighting for Long-Range LM (NAACL 2025). **Est. 0.005-0.015 BPB.** Low complexity.
- **Higher-order complementary training** — #803 uses bigram statistics for loss reweighting. Using 4-gram or 7-gram statistics (matching the actual eval-time cache) would better align training with eval. Tokens easy for 7-grams but hard for bigrams currently get full training weight but will be handled by the cache at eval time. **Est. 0.005-0.010 BPB.** Low complexity.
- **Adaptive complementary alpha** — instead of fixed COMPLEMENT_ALPHA=0.5, make `w = 1 - alpha * p_ngram`. Tokens with p_ngram=0.99 get near-zero weight; tokens with p_ngram=0.1 get nearly full weight. Smooth weighting instead of hard threshold. **Est. 0.003-0.008 BPB.** Trivial complexity.

**Tier 2 — Top picks for pure neural track** (sorted by expected value)
- **Engram — TRIED as EngramLite** (#1089: bigram+trigram, 2 heads, 8192 buckets, part of 1.1086 record) (DeepSeek, Jun 2025). Multi-head hashing (K=4 heads per N-gram order) + context-aware gating (sigmoid gate suppresses noisy lookups) + tokenizer compression (collapse equivalent IDs, −23% vocab). The competition's BigramHash is a primitive single-head version. Engram's gating could rescue higher-order N-grams (#609 showed TrigramHash hurts without gating: +0.0049). Multi-head reduces hash collisions. Depthwise causal conv for temporal smoothing. Main constraint: embedding tables consume artifact space (2-4MB for full multi-order). **Est. 0.003-0.008 BPB.**
- **Mousse optimizer** (arXiv:2603.09697). Curvature-aware Muon — Shampoo preconditioning before orthogonalization. ~12% more effective training at 3% overhead. Drop-in. **Est. 0.003-0.008 BPB.**
- **OptRot pre-quantization** (arXiv:2512.24124). Rotation matrix redistributes weight outliers before quantizing. Fuses into adjacent layers — zero artifact cost. **Est. 0.001-0.003 BPB** (reduced estimate — Full GPTQ already handles much of the outlier problem; #586 shows rotation "substitutes with GPTQ at int6").
- **Turbo-Muon — TRIED** (#1089: 1.1086 BPP, record). AOL preconditioning + Polar Express coefficients + row_col normalization (4 NS iters). Preconditioned Newton-Schulz — 5-10% faster training. More steps in 600s. Significance test waived for systems-only changes. **Est. 0.002-0.005 BPB.**
- **qTTT — query-only test-time training** (arXiv:2512.13898). Cache K/V once, adapt only Q projection weights. 2-3x more TTT epochs within eval budget. **Est. 0.003-0.010 BPB.** Note: use AdamW with cosine LR, not SGD — #601 shows SGD TTT hurts GPTQ models (+0.030). AdamW TTT works but requires GPTQ calibration within training budget (not eval time).
- **LaCT — Large Chunk TTT** (arXiv:2505.23884, ICLR 2026 Oral). Document-sized chunks → 70% GPU utilization (vs <5% for per-token TTT). Uses Muon as fast-weight optimizer. **Est. 0.002-0.008 BPB** over current TTT approaches.
- **Prune-then-quantize ordering** (arXiv:2603.18426, ICLR 2026). Progressive Intensity Hypothesis: weaker perturbations first, stronger later. #609 currently does quantize-then-prune; reversing the order is a **zero-cost experiment**. Theory + experiments show 0.001-0.003 BPB free gain.
- **SLOT — output-head TTT — TRIED** (#1084: -0.0008 BPB, marginal). Adds a single learnable delta vector (512 dims) at last hidden layer, optimized per-batch during eval. Lighter than LoRA TTT — avoids the GPTQ weight-corruption problem (#601). Compatible with score-first constraint. **Est. 0.002-0.006 BPB.**
- **YAQA adaptive rounding** (arXiv:2505.22988). Drop-in GPTQ replacement: optimizes rounding toward full model's KL divergence (not just per-layer error) via Kronecker-factored Hessian. ~30% less quantization error than GPTQ. Post-training. **Est. 0.001-0.003 BPB.**

<details>
<summary><strong>More Tier 2 ideas</strong> (lower EV or higher complexity)</summary>

| Technique | Est. BPB | Key idea | Complexity |
|-----------|----------|----------|------------|
| **GLU Attention on Values** (arXiv:2507.00022) | 0.002-0.005 | GLU nonlinearity on V projections. Zero parameters, zero overhead. Composable with XSA. | **Very low** |
| **CAGE QAT Gradient** (arXiv:2510.18784, ICLR 2026) | 0.002-0.005 | Curvature-aware STE replacement using Adam's second-moment. W3A3 CAGE matches W4A4 STE. Composes with HESTIA/Soft-Round. Zero artifact cost. | Low-moderate |
| **IFNSO / Iteration-Free NS** (arXiv:2602.02500) | 0.002-0.005 | Collapses Muon's 5-10 NS iterations into one polynomial eval. Systems-only (more steps in 600s). Drop-in. | **Very low** |
| **V:N:M Activation Sparsity** (arXiv:2602.06183) | 0.005-0.015 | Generalizes 2:4 to higher sparsity ratios (1:4+). 6-10x sparse matmul at relu²'s natural >90% sparsity. 1.4-1.7x end-to-end speedup. **Systems-only.** | Moderate-high |
| **Batch Size Warmup** (arXiv:2505.23971) | 0.002-0.005 | Start small (262K), grow to 786K as critical batch size increases. 43% fewer gradient steps for same loss. Resolves the 524K-vs-786K debate. | **Very low** |
| **FlashSigmoid Attention** (Apple, ICLR 2025) | 0.002-0.010 | Replace softmax with sigmoid. Eliminates attention sinks entirely. 17% kernel speedup on H100 (systems-only). | Low-moderate |
| **WSM Checkpoint Merging** (arXiv:2507.17634) | 0.002-0.006 | Replace warmdown with constant-LR training + offline checkpoint merge. More full-LR steps. Theoretically optimal. Compatible with existing EMA. | Low |
| **FoX Forgetting Attention** (arXiv:2503.02130, ICLR 2025) | 0.003-0.008 | Data-dependent forget gate on attention. Eliminates need for positional embeddings. FA3-compatible. | Moderate |
| **DeepCrossAttention** (arXiv:2502.06785, ICML 2025) | 0.003-0.008 | Input-dependent depth routing over all previous layers (replaces simple residuals). 3x convergence speed claim. ~1K params for 11L. | Moderate |
| HybridNorm (arXiv:2503.04598) | 0.002-0.006 | Mixed Pre/Post-Norm for better depth utilization | Very low |
| Differential Attention (arXiv:2410.05258) | 0.005-0.015 | Difference of two softmax maps; reduces outliers | High (arch change) |
| Lattice VQ (arXiv:2603.11021) | 0.005-0.015 | Joint 24-weight Leech lattice encoding; saves 2-4 MB | High (custom kernels) |
| VGA (arXiv:2510.09017) | 0.002-0.005 | Value-gated attention; fixes sliding window sinks | Low-moderate |
| Neural Cache cross-window KV ([#318](https://github.com/openai/parameter-golf/pull/318)) | unknown | Cache K/V from prior windows so new queries attend to 50K+ context; zero artifact cost; untested | Low (FA3 already supports seqlen_k > seqlen_q) |
| Predictive Batch Scheduling (arXiv:2602.17066) | 0.002-0.005 | Loss-aware data ordering (NOT content curriculum); 6-13% faster convergence | Low |
| Late-Stage SAM (arXiv:2410.10373) | 0.002-0.005 | Sharpness-aware minimization last 5-10%; flatter minima complement EMA | Moderate (Muon-SAM) |
| WaveletGPT (arXiv:2409.12924) | 0.003-0.010 | Multi-scale Haar wavelet structure on half of embedding dims; 40-60% faster convergence | Low (zero params) |
| AGGC adaptive gradient clipping (arXiv:2601.11864) | 0.002-0.005 | Per-group adaptive clip thresholds; exploits Q-matrix heterogeneity from #215 | Low (optimizer state) |
| **2:4 Structured Activation Sparsity** (arXiv:2503.16672) | 0.003-0.008 | relu² is already 84-98% sparse; enforce NVIDIA 2:4 pattern for **2× sparse matmul on H100 tensor cores**. ~15-20% more training steps. **Systems-only = significance waived.** | Moderate (custom kernels) |
| In-Place TTT with NTP objective (ICLR 2026 Oral) | 0.003-0.010 | Update MLP final projections during eval using NTP loss (not reconstruction). NTP alignment may explain why naive SGD TTT is neutral at frontier — objective misalignment. MLP-only, last 3 blocks. | Moderate |
| PoPE — Polar Position Embedding (arXiv:2509.10534) | 0.002-0.005 | Decouples content (magnitude) from position (angle) in attention. Principled fix for what Partial RoPE approximates. Strong length extrapolation. OpenAI co-author. | Moderate |
| **Liger-Kernel fused ops** (LinkedIn open-source) | 0.002-0.006 | Fused Triton: RMSNorm (6×), linear+CE (3×), residual+norm. Eliminates kernel launch overhead. 20-43% throughput in benchmarks. pip-installable. **Systems-only.** | Very low |
| Cross-Layer KV Sharing (MLKV/CLA, NAACL 2025) | 0.002-0.006 | Adjacent layer pairs share K/V projections. Saves ~0.5MB artifact for 12L or wider MLP. Unlike depth recurrence, only K/V shared — no quant amplification. | Moderate |
| **Block AttnRes** (arXiv:2603.15031, Kimi, Mar 2026) | 0.003-0.008 | Efficient variant of AttnRes (which failed at 54% overhead in #362). Block partitioning (3 blocks at 11L) reduces overhead to <2%. 1.25× convergence efficiency. | Moderate |
| **QK-Norm** (arXiv:2010.04245, used in Gemma 2/DeepSeek-V3) | 0.001-0.004 | L2-normalize Q and K before dot product + learned per-head temperature. **Prevents attention logit explosion** — the root cause LN Scale patches. Could enable stable 12-13L training. Suppresses #215's Q condition numbers (100M+ → 1). ~4 lines. | **Very low** |
| **Hourglass FFN** (arXiv:2602.06471, Feb 2026) | 0.002-0.006 | Replace wide MLP-3x with stacked narrow-to-narrow sub-MLPs + residuals. **Deeper MLP at fewer params.** Paper: outperforms conventional FFN up to 400M params. Freed params → extra layers or larger BigramHash. | Low-moderate |
| **CERWU** (arXiv:2505.18758) | 0.003-0.008 | Rate-distortion optimal quantization: co-optimizes quant grid + weight updates + entropy coding. GPTQ is special case (λ=0). **Principled upgrade to GPTQ-lite.** Post-training, orthogonal to QAT. | Moderate |
| **Progressive Window Warmup** (modded-nanogpt, proven 2025) | 0.003-0.007 | Start with short local attention (128-384 tokens), grow to full 2048 during training. Faster early steps → more total steps. **Different from blocked seq curriculum** — same input length, just restricted attention span. Systems-only. | Moderate |
| NuMuon (arXiv:2603.03597, Mar 2026) | 0.002-0.006 | Nuclear-norm constraint on Muon updates → lower stable rank → better zstd compression. Pushes compressibility into optimizer itself. Distinct from Mousse/Turbo-Muon (those target speed). | Low-moderate |
| **AdamHD Huber Decay** (arXiv:2511.14721) | 0.002-0.005 | Replace L2 weight decay with Huber regularizer: quadratic below threshold, **linear above**. Specifically suppresses large outlier weights that cause int6 clipping loss. Drop-in for Muon's decoupled WD. Synergizes with GPTQ-lite (fewer outliers = less work). | **Very low** |
| **Layer-Wise Scaling** (arXiv:2509.06518) | 0.002-0.005 | Non-uniform FFN width per layer (e.g., MLP-4x middle, MLP-2x edges). Same total params, better allocation. Crown/Frame/Reverse variants all beat uniform at 180M params. Complements Hourglass FFN (structure vs width). **Zero cost — just per-layer dims.** | **Very low** |
| **Hyper-Connections** (arXiv:2409.19606, ICLR 2025; mHC: 2512.24880, DeepSeek) | 0.003-0.008 | Learned multi-depth residual mixing: replaces `x+f(x)` with a connection matrix (n=2 → 16 params/layer, ~176 total). Richer than Catalytic Residuals or DenseFormer DWA. mHC adds Sinkhorn stability. **Drop-in.** | Low-moderate |
| **HESTIA soft QAT** (arXiv:2601.20745) | 0.002-0.006 | Replaces hard STE with temperature-annealed softmax relaxation + per-tensor Hessian guidance. Enables **earlier QAT** without premature discretization. Synergizes with OptRot. | Moderate |
| **Compute-Optimal QAT** (arXiv:2509.22935, Apple) | 0.001-0.004 | Scaling law for optimal FP→QAT split. **Cooldown+QAT fusion:** activate QAT at warmdown onset, eliminating redundant FP updates. Principled replacement for empirical Late QAT thresholds. | **Very low** |
| **ScaleBITS** (arXiv:2602.17698) | 0.002-0.006 | Automated per-layer bit-width search (which layers get int5 vs int6). Sensitivity analysis + greedy optimization under 16MB constraint. +36% over uniform precision in paper. Replaces manual assignment. | Moderate |
| **CPSVD** (arXiv:2510.19385) | 0.003-0.008 | Column-Preserving SVD: identify weight columns that compress cleanly via low-rank factorization, store rest as int6. **Orthogonal to quantization** — reduces param count, not precision. Freed bytes → capacity. Entirely unexplored in competition. | Moderate |
| **Softpick / Rectified Softmax** (arXiv:2504.20966) | 0.002-0.006 | Replaces softmax with rectified non-sum-to-one variant. **Eliminates attention sinks and massive activations** — directly improves int-N quantization quality (lower kurtosis). 47% sparse attention maps. "Quantized Softpick outperforms quantized softmax at lower bit widths." | Low |
| **Anti-Layer Removal** (arXiv:2603.19348) | 0.002-0.006 | Some layers are "anti-layers" whose removal **improves** performance. Anatomical analysis of 135M model shows 10^7 importance range. If 1-2 middle layers of 11L are anti-layers, removing them frees artifact space for wider MLP or more BigramHash. **Zero-cost ablation pass on existing checkpoint.** | Very low |
| **Deep Delta Learning (DDL)** (arXiv:2601.00417) | 0.003-0.007 | Rank-1 erasure gate on residual: `x + β·proj(x) + f(x)`. Learned gate erases stale features before writing new ones. **3-5 ppl improvement at 124M.** ~5.6K params for 11L. Addresses residual-path interference in quantized models. | **Very low** |
| **Variance-Adaptive Muon (Muon-VS)** (arXiv:2601.14603) | 0.002-0.005 | Variance normalization before NS orthogonalization. Reduces Muon's step-size sensitivity + hyperparameter sensitivity. **Zero extra hyperparameters — direct drop-in.** Lower val loss than standard Muon on GPT-2/LLaMA. | **Very low** |
| **TEON cross-layer Muon** (arXiv:2601.23261) | 0.003-0.007 | Joint tensor orthogonalization across ALL layers (vs Muon's per-layer NS). Captures inter-layer gradient relationships. Consistent ppl improvement 130M-774M. Targets **loss per step** — critical for 600s budget. | Moderate |
| **Seesaw LR+Batch Schedule** (arXiv:2510.14717) | 0.002-0.005 | Multiply LR by 1/sqrt(2) and double batch size simultaneously. ~36% fewer serial steps at equal FLOPs. Principled foundation for the 524K→786K ramp. | **Very low** |
| **1-sqrt Cooldown Shape** (arXiv:2508.01483, TMLR 2025) | 0.001-0.003 | Replace linear warmdown with `1-sqrt((t-T0)/(T+1-T0))`. Outperforms linear, cosine, and other cooldown shapes in WSD schedules. Zero-cost swap. | **Very low** |
| **SSMax (Scalable-Softmax)** (arXiv:2501.19399) | 0.001-0.004 | Scale softmax by input sequence length to prevent attention flattening at seq2048. One scalar multiply. Compatible with FA3. | **Very low** |
| **DCMHA** (arXiv:2405.08553, ICML 2024 Oral) | 0.005-0.015 | Dynamically Composable Multi-Head Attention. Input-dependent transforms on score/weight matrices. Matches 1.7-2x compute models at 405M. Few KB params for 11L. | Moderate-high |
| **VPTQ** (arXiv:2409.17066, EMNLP 2024) | 0.002-0.006 | Vector PTQ guided by second-order Hessian. Beats GPTQ by 0.01-0.34 ppl at 2-3 bits. 10-18x faster than AQLM. Practical within 600s budget. | Moderate |
| **QTIP Trellis Quantization** (arXiv:2406.11235, NeurIPS 2024) | 0.003-0.008 | Trellis coded quantization — stateful sequential coding achieving ultra-high-dimensional VQ. At 3 bits, matches GPTQ at 4 bits. Bitshift trellis for GPU-parallel decoding. | High |
| **Context Tree Switching (CTS)** | 0.002-0.008 | Extension of CTW that handles non-stationary sources (distribution shifts between documents). Same complexity as CTW but mixes over larger model class. | Moderate |

</details>

**Tier 3 — Novel approaches, higher risk**

- **Knowledge distillation** — untried. Train larger teacher ~7 min, distill to 16MB student ~3 min. Est. 0.005-0.010 BPB but tight time budget is the constraint. High complexity.
- **Partial weight sharing + 14L** — share middle-layer pairs with per-layer LoRA adapters. Saves 3-5 MB for extra layers. "Relaxed Recursive Transformers" shows LoRA-adapted shared layers recover most unique-layer quality. Est. +0.005-0.015 BPB. #579 tested 6×2 loops: 1.1478 but GPTQ compounds multiplicatively (see What Doesn't Work).
- **nGPT hypersphere normalization** (arXiv:2410.01131) — constrain Q/K to unit-norm rows, eliminating #215's extreme condition numbers. NVIDIA claims 4-20x convergence speedup. Est. 0.003-0.008 BPB. High complexity, untested at this scale.
- **BitNet b1.58** — #367 reached 1.1770 (68M ternary params). Standard stack breaks on ternary (different optimization regime). Int4 with late QAT is an unexplored middle ground. MoE confirmed dead at this scale (see What Doesn't Work).

### Recently Discovered Techniques (Mar 28 research)

**High relevance — directly applicable:**

| Technique | What It Is | Est. Impact | Difficulty |
|-----------|-----------|-------------|------------|
| **MUD optimizer** ([arXiv:2603.17970](https://arxiv.org/abs/2603.17970)) | Drop-in Muon replacement using triangular whitening instead of Newton-Schulz. 1.3-2.6x faster peak tokens/s on A100. | 0.001-0.003 BPB (via 200-500 extra steps) | Easy |
| **Sigmoid attention + FlashSigmoid** (ICLR 2025) | Replace softmax with element-wise sigmoid. FlashSigmoid kernel: 17% inference / 4% training speedup on H100. Eliminates token competition. | 0.001-0.005 BPP (speed + possible quality gain) | Moderate |
| **Entropy coding for weights** (ANS/Huffman vs zstd) | Specialized entropy coders exploit quantized weight statistics better than general-purpose zstd. **#1089 used Brotli+byte-shuffle instead of zstd on mixed int6/int7.** EntQuant achieves 2-3 bit effective rates from FP8. | 0.005-0.015 BPP (20-40% smaller artifacts = larger model or higher precision) | Moderate |
| **Compute-Optimal QAT scheduling** (Apple 2025) | FP cooldown + QAT fusion — skip separate cooldown phase, do LR decay jointly with QAT. Optimal QAT fraction depends on compute budget. | 0.001-0.003 BPP | Easy |

**Medium relevance — higher implementation cost:**

| Technique | What It Is | Est. Impact | Difficulty |
|-----------|-----------|-------------|------------|
| **QTIP** (NeurIPS 2024) | Trellis coded quantization — Viterbi-optimal paths through 256-dim codebook. Near-FP16 quality at 2-bit. | 0.005-0.015 BPP (fit 3x more params in 16MB) | Hard |
| **Mixture of Recursions** (NeurIPS 2025) | Per-token adaptive depth in recursive transformers. Easy tokens exit early; hard tokens get more passes. 2x inference throughput. | 0.002-0.008 BPP (more TTT iters or longer windows) | Moderate |
| **AQLM + PV-Tuning** (NeurIPS 2024 oral) | Additive multi-codebook quantization, Pareto-optimal below 3 bits. PV-Tuning fixes STE for extreme compression. | 0.01-0.03 BPP (fit ~64M params at 2-bit in 16MB) | Hard |

### New Techniques (Mar 29 research)

| Technique | What It Is | Est. Impact | Difficulty |
|-----------|-----------|-------------|------------|
| **Relaxed Recursive Transformers + LoRA deltas** ([ICLR 2025](https://arxiv.org/abs/2410.20672)) | Share base weights across all layers, add tiny per-layer LoRA deltas (rank-32). Effectively 24L model with ~11L parameter budget. SVD-initialized. MoL variant adds per-token LoRA routing. | 0.01-0.03 BPB (deeper model in 16MB) | Moderate |
| **Mixture of Depths (MoD)** ([arXiv 2024](https://arxiv.org/abs/2404.02258)) | Per-layer router skips "easy" tokens through some layers. Budget parameter B controls skip fraction. Reduces eval compute for given depth. | 0.002-0.008 BPB (faster eval → longer windows) | Low |
| **Soft Quantization via Weight Coupling** ([Jan 2026](https://arxiv.org/abs/2601.21219)) | Physics-inspired coupling regularizer pulls weights toward discrete clusters during training. No STE needed — weights self-discretize. | 0.001-0.003 BPB (smoother QAT alternative) | Low |

### Techniques from Recent Competition PRs (Mar 30)

| Technique | What It Is | Where Used | Est. Impact |
|-----------|-----------|------------|-------------|
| **CUTLASS EVT backward MLP fusion** | Fuses `(grad @ W_down) * act_grad` into GEMM epilogue via Epilogue Visitor Tree. Intermediate never touches HBM. Hopper-only. Complements forward Triton fusion. | #1105 (@abaybektursun): -3.7% step time, +500 steps | 0.001-0.003 BPB |
| **Sigmoid-gated skip connections** | Learnable sigmoid gate on U-Net skip paths: `out = hidden + sigmoid(g) * skip`. Lets model tune encoder-decoder blending per layer. ~5 params for 5 skip paths. | #1089, #1122 | 0.001-0.002 BPB |
| **Brotli-11 + byte-shuffle** | Brotli quality=11 outcompresses LZMA-9 by ~580KB on int6 weights. Byte-shuffle pre-filter groups MSB/LSB bytes for better compression. 580KB = ~93K extra int5 params. | #1089, #1105 | 0.002-0.005 BPB (via larger model) |

### Tokenizer Optimization (Validated by #1143 Scylla — 1.0806 BPP)

The biggest single-technique gain in the competition came from tokenizer choice, not architecture or quantization. #1143 achieved 1.0806 BPP (beating #1089's 1.1086 by 0.028) primarily through a TokenMonster-derived tokenizer.

**Why it matters disproportionately at this scale:** The embedding layer costs V*d parameters. At V=1024, d=384: ~295KB quantized. At V=32K: ~9.2MB — over half the 16MB budget. The parameter golf regime makes vocabulary size a first-order architectural decision.

**TokenMonster vs BPE/Unigram:** TokenMonster uses ungreedy multi-branch search (6 parallel branches scored per position). Produces ~37.5% fewer tokens at equivalent vocab size compared to BPE. BPE's greedy merges are known suboptimal (Bostrom & Durrett, EMNLP 2020).

**Compression is necessary but not sufficient:** Schmidt & Reddy (EMNLP 2024) showed maximum compression does not maximize model performance. The winning tokens are those aligned with what the model can learn, not those that minimize sequence length. #1143's autoresearch found a better vocabulary than pure compression optimization.

**Open directions:**
| Direction | What to Try | Est. Impact |
|-----------|-------------|-------------|
| **More aggressive vocab pruning** | Literature suggests 60%+ tokens removable. #1143 only pruned 2.5% (1024→998). Try 800? 512? | 0.005-0.02 BPB |
| **FineWeb-aligned tokenizer training** | Train tokenizer on FineWeb itself, not generic English. Reduces domain mismatch. | 0.002-0.01 BPP |
| **Byte-level fallback with larger vocab** | TokenMonster + 2048 vocab. More tokens but each carries more information. | Unknown |
| **Combine Scylla tokenizer with #1089 stack** | Turbo-Muon + EngramLite + Scylla. Currently untried. | Potentially <1.07 BPP |

### Techniques from Latest Competition Frontier (Mar 31)

| Technique | What It Is | Where Used | Est. Impact |
|-----------|-----------|------------|-------------|
| **QK-Gain scaling (gain=4.0)** | Learnable per-head scalar on queries after QK-norm. Controls attention sharpness (temperature). Higher gain = sharper attention = more decisive routing. Zero parameter cost. | #1176 (-0.004 BPP from #1125's 45-experiment sweep) | 0.003-0.006 BPB |
| **P2 / Focal Loss (gamma=2)** | Difficulty-aware: `(1-p)^2 * (-log p)` down-weights easy tokens, focuses gradient on hard tokens. Rarely used in LLM training — conventional wisdom says it hurts calibration. But for BPB optimization, hard tokens dominate the loss. | #1180 (1.0577, 1-seed, unvalidated) | Unknown (no ablation) |
| **Conv Token Mixer** | Causal depthwise conv1d for cheap local context mixing. Frees attention capacity for long-range. ~3K params/layer for kernel=4. From ConvMixer/Conformer/Mamba lineage. | #1180 | Unknown (no ablation) |

### The Path Below 0.9 BPP (from #1184's 0.9485 baseline)

**Best-case realistic estimate:** Stacking SLOT + QK-Gain + TTT on #1184 → **~0.915-0.925 BPP**. Sub-0.9 requires at least one breakthrough.

| Technique to Stack on #1184 | Est. Delta | Confidence |
|------------------------------|-----------|------------|
| **SLOT (8 steps, single-layer)** | -0.012 to -0.018 | Medium-high |
| **Aggressive SLOT (32 steps, rank-4, multi-layer)** | -0.018 to -0.028 | Medium |
| **QK-Gain 4.0** | -0.003 to -0.006 | Medium |
| **Muon-TTT (score-first)** | -0.002 to -0.004 | Medium |
| **Brotli + byte-shuffle** | 0 to -0.002 | Medium |
| **Over-Encoding input embeddings** ([ICML 2025](https://arxiv.org/abs/2501.16975)) | -0.005 to -0.015 | Low (untested) |

**New technique: Over-Encoding (OE)** — Keep Scylla's 998 output vocab but use hierarchical n-gram input embeddings (sum of 1-gram + 2-gram + 3-gram tables). Creates exponentially larger effective input vocab with <5% memory overhead. Log-linear loss reduction reported. Could be the key to sub-0.9.

**SLOT scaling insight:** More optimization steps (8→32) are nearly free since features are cached. Rank-4 delta (2048 params) and multi-layer delta (layers 9-11) are unexplored extensions with medium-high upside.

<details>
<summary><strong>What Doesn't Work</strong></summary>

**Three failure patterns.** (1) **Throughput cost exceeds quality gain.** In a 600s budget, anything adding >10% step overhead needs >10% per-step improvement to break even. QAT (#236: 115ms vs 67ms baseline), NorMuon (#236: 110ms), and MTP (#212, #236: 86ms) all fail this test. **Partial reversal:** #1031 uses MTP as auxiliary-only training signal (2-head, weight=0.1, discarded at export) with -0.0037 BPP claimed at zero artifact/eval cost — a different usage pattern (1 seed, unvalidated). (2) **Mechanism redundancy.** Stacking two techniques that extract the same signal yields diminishing returns — TTT+XSA underperforms XSA-alone (#290 vs #265), error-guided TTT doesn't improve over uniform TTT (#296), EMA without XSA hurts (#201). (3) **Regime incompatibility.** Techniques optimized for int6 break under different weight representations — the standard stack (XSA, SmearGate, WD, EMA/SWA, TTT) all fail on ternary (#367), and recurrence amplifies quantization error 900× (#363).

- **12 layers at seq2048 (slower steps cancel extra capacity)** — #219's 12L at seq2048 runs at 107ms/step, fitting only ~5,590 steps. Result: 1.1541 vs 11L's 1.1326. However, #76 shows 12L at **seq1024** (59ms/step, ~9000 steps) reaches 1.1468 — the tradeoff depends on sequence length.
- **Late QAT at 12L is step-budget-dependent.** @saml212's [#332](https://github.com/openai/parameter-golf/pull/332) found that at 12L, Late QAT added ~7ms/step overhead, costing ~770 training steps. At 11L, those steps would cost ~7ms each — but at 12L, each step is already more expensive and step count is already lower, so the overhead-to-gain ratio worsens. Result: Late QAT was dropped from the 12L submission. Takeaway: the same technique's cost-benefit flips depending on step time and total step count. Always re-evaluate overhead techniques when changing layer count.
- **Int5-MLP tradeoff is layer-dependent** — At 11L, #236 found int5 quant penalty (0.029) outweighs artifact savings. But at 10L, #180 used int5 to fund BigramHash(10240) (previous official SOTA at 1.1428; now superseded by #414 at 1.1228). **New data from #469:** all-int5 on a larger model (d=576, 27M params) with early QAT activation (threshold 0.50 = ~1700 adaptation steps) reaches **1.1418** (1-seed) — validating the "train larger, quantize harder" principle.
- **Larger vocabularies + fewer layers** — #123 (vocab 4096, 8L) at 1.1642 and #200 (SP4096, 9L) at 1.2012 both underperform #198's 11L at 1.1326. The embedding matrix gets 4x larger, forcing fewer layers. At current artifact sizes, depth wins over vocab breadth. **New data from #465:** Int6 embedding quantization costs only **+0.0005 BPB** — enabling sp8192 at d=512, but even then sp8192 8L (1.1794) loses to sp1024 10L (1.1508). More layers still dominate.
- **SmearGate without OrthoInit** — hurts BPB by 0.003 (see SmearGate deep dive).
- **SWA with bf16 accumulation** — #212 found catastrophic precision loss when accumulating SWA checkpoints in bf16. Must use fp32. (However, @kellyvv's [#238](https://github.com/openai/parameter-golf/pull/238) found that with enough SWA checkpoints (84), the quant gap actually *reverses* — quantized BPB becomes 0.037 *better* than pre-quant. SWA smoothing eliminates quantization-sensitive outliers.)
- **MTP (multi-token prediction)** — #212's controlled test: no BPB improvement (1.1947 vs 1.1929 control).
- **Curriculum learning (content-based)** — #212 found no effect.
- **LAWA-EMA replacing SWA — context-dependent.** #201 tested EMA alone on #198 base: 1.1551 (0.023 worse than SWA). But #287 uses EMA (decay=0.997) WITH XSA and reaches 1.1280 — beating SWA. EMA needs XSA to work. EMA decay=0.999 was also tried on #287 and hurt BPB — too slow to average (per @jfprincz). The sweet spot is 0.997.
- **cuDNN SDP vs Flash SDP** — #281 found cuDNN is 40% faster per attention op but produces worse BPB (1.1455 vs 1.1418). More steps doesn't help — different internal accumulation precision hurts quality.
- **SwiGLU activation** — worse than relu² on the standard architecture (#340, #344). **However, GEPA's AI-discovered architecture uses SwiGLU successfully** — both with TTT (#462: 1.0672) and **without TTT (#505: 1.1181, GEPA non-TTT)**. **Clarification:** #505's code reveals the "SwiGLU" label is misleading — the actual activation is **Star-ReLU** (relu²+learned affine scale+bias, arXiv:2210.13452), NOT true SwiGLU gating. The MLP uses a single up_proj, not the dual-projection gated structure of SwiGLU. Star-ReLU works when co-optimized with U-Net skip gates and hidden=1792.
- **Step-based LR schedule** — #344 found −0.483 BPB vs wallclock-based warmdown. Catastrophic because the 600s budget varies by hardware; step-count schedules can't adapt.
- **Error-guided TTT** — concentrating TTT on highest-loss tokens doesn't help; they're genuinely unpredictable. Also: **focal loss TTT** (#481: no improvement over CE) and **KL-divergence from pre-quant model** (#481: no improvement). 7 failed TTT objective/targeting variants total.
- **Advanced quantization algorithms at int6 (#756, @abaybektursun, SOTA holder).** Qronos iterative Hessian (+0.0007 worse) and CDQuant coordinate descent (+0.0005 worse) both fail to improve over standard GPTQ at int6. Reason: the int6 quant gap is only +0.0036 BPB — most weights are already at their optimal grid point. **At int6, GPTQ is near-optimal.** Also: **TTT alone is marginal on val-calibrated GPTQ stacks** — 25 total failed TTT attempts across two stacks (full, MLP-down, MLP-all: all +0.0001 or neutral). Val-calibrated GPTQ rounding decisions are disrupted by gradient updates.
- **MoE at small scale** — #480: 2-expert soft-routing MoE = **−0.06 to −0.08 BPB** vs dense baseline on 8xH100. Apple scaling laws (ICML 2025) confirm optimal sparsity = 0 below ~500M params. MoE is definitively unviable at competition scale.
- **Lightweight TTT — 5 variants dead at frontier.** Naive SGD (#338: neutral), MLP-only (#375: neutral), Reptile (#375: +0.008 worse), Self-Distillation (#379: −0.0003), MAML (#384: +0.085 worse). All used conservative settings. **But aggressive TTT works in principle:** backward-looking TTT reached **1.1162** (#606, 3-seed — now closed for eval-time GPTQ). The lightweight variants failed; the aggressive variants (AdamW, cosine LR, selective freezing) succeeded — but need GPTQ within training budget.
- **Multi-epoch TTT — memorization is a gradient, not a threshold (#568, #512, #484).** The PROTEUS series reveals a clear memorization gradient: **3ep→0.9512** (#512), **5ep→0.7853** (#568, −0.166 BPB from just 2 more epochs). At 0.78 BPB the model compresses to <1 bit per byte — near-certain data reproduction. #484's original diagnostic (10+ constant-LR = memorization, 3ep = genuine) doesn't capture this: cosine LoRA TTT memorizes progressively, not at a fixed threshold. The ~0.95 floor was for full-weight TTT; LoRA TTT memorizes differently. Cosine TTT at 100ep (#517: 0.978) stays above the 0.95 floor — suggesting cosine full-weight and cosine LoRA have different memorization dynamics.
- **Systematic frontier negatives (#375, 13 techniques on #315 base, $500/24hrs on 8xH100).** **Reptile meta-TTT: +0.0076 worse** (1.1332, 20% training budget consumed). All 3 TTT variants (low-LR, high-LR MLP-only, Reptile) failed. Also failed: memory tokens (+0.016), Canon layers (48% overhead), MTP (+0.028), gradient-guided quant (noise), cautious WD (breaks torch.compile: 710× slower), label smoothing, L1 reg, 1M batch, full-run QAT. Key positive findings: **EMA > standard SWA by 0.003** (3-seed); **786K > 524K by 0.004** (total tokens > gradient steps at frontier). **Throughput heuristic: each 1ms step overhead ≈ 0.006 BPB** — any technique adding Nms must deliver >N×0.006 to break even. INT4 quant gap: 0.048-0.060 (exponential from int6's 0.006). 3-seed std: 0.0007 BPB.
- **INT4 quantization** — catastrophic. #480's controlled grid: int6→int5 MLP costs +0.007, but int6→int4 MLP costs **+0.065** (10× worse). Full grid: attn6/mlp4 = 1.2111, attn5/mlp4 = 1.2183 vs baseline 1.1456. Int4 is a dead end.
- **MLA (Multi-Head Latent Attention)** — #354's kv_rank=128 MLA on 13L runs at 83ms/step vs ~43ms baseline, halving token throughput (~3.7B vs ~7.2B tokens). Pre-quant 1.2838 — architecture quality may be there but throughput cost makes it infeasible in the 600s budget.
- **Block-wise weight sharing / depth recurrence** — **aggressive recurrence (3+ cycles) fails at 512d**, but shallow recurrence works. #344: 2x slower. #316: 0.09 BPB cost. #319: loop gates collapse. #363: **quant error amplifies ~900× over 3 cycles**. #484 (EBLS): gammas → 0 for MLP (fully shared). **#579:** 6×2 loops — 1.1478 (1-seed). GPTQ compounds multiplicatively: 2 loops survive, 3+ catastrophic (+4.3 BPB). **Exception: #686** (@msisovic) uses **shallow recurrence** (layers 4+5 repeated once each, 11→13 virtual layers) with per-pass learnable block scalars (~2K params). Reaches **1.1182** (3-seed). Recovers ~70% of independent 12L gain at minimal step cost. Key: staying within the "2 loops survive" zone.
- **AttnRes (learned softmax over depth)** — #362: 54% throughput penalty from routing attention over layer outputs. Infeasible in 600s budget.
- **MUD optimizer (#510)** — Triangular Gram preconditioning replacing Muon's Newton-Schulz. 1.1989 BPB at 118ms/step (4.5× slower than Muon's ~26ms). Only 5,087 steps in 600s. Alternative optimizers remain unviable: throughput cost dwarfs quality gains.
- **FTLE per-row precision (#316)** — Dynamical systems-inspired row-level quantization (Lyapunov exponent tracking). Clean negative result: uniform int-N beats FTLE-guided mixed precision at every bit width, because mixing bit widths within a row *increases* quantized value entropy, which defeats zstd compression. Lower RMSE does not imply smaller artifact.
- **#609 frontier ablations (16 techniques on #593 stack).** On the current best non-TTT base: **VRL +0.0012** (conflicts with VE128), **Gated Attention +0.0011** (3% step overhead), **Catalytic Residuals −0.0001** (redundant with existing scaling), **Backout −0.0005** (redundant with U-Net skips), **TrigramHash +0.0049** (hurts compression), **Hadamard rotation −0.0002 but +0.5MB** (net negative for artifact), **Temperature scaling +0.0002** (model well-calibrated at T=1.0), **seq4096 eval catastrophic** (RoPE breaks beyond training length), **lzma at 99.7% Shannon limit** (entropy coding gains capped at 0.05MB).
- **Knowledge Distillation — dead at this budget ([#1029](https://github.com/openai/parameter-golf/pull/1029)).** Hard distillation catastrophic (+0.090 to +0.407 BPP). Soft KL (top-32 cached logits from 105M teacher): +0.003 BPP worse at all alpha values. ~11ms/step I/O overhead costs ~280 steps; knowledge transfer doesn't compensate. Extended 2-hour training shows no crossover — curves track identically. Online distillation (761ms/step) is a non-starter. **At 600s budgets, per-step overhead is fatal.**
- **Compression moonshots — MSE ≠ artifact size ([#1048](https://github.com/openai/parameter-golf/pull/1048), 8 findings).** Procrustes symmetry-transport: 91% MSE reduction but 380% larger artifact (rotation matrices are dense/high-entropy). Low-rank rotation: rank-128 captures only 16.6% of variance. 3% pruning *increases* artifact by 728KB (zeroing hurts zstd-22). **Takeaway: int6 + zstd-22 is near-optimal. Always measure compressed artifact, not RMSE.**
- **MC Dropout ensembling ([#1021](https://github.com/openai/parameter-golf/pull/1021))** — K=16 passes at dropout=0.30: +0.005 BPP. dropout=0.05: +0.002. Sub-networks lack diversity at 17M params. Deterministic single pass strictly better.
- **AdamW TTT at high learning rate ([#1045](https://github.com/openai/parameter-golf/pull/1045))** — AdamW at lr=0.002 degrades from 1.1509 to 1.2804 (+0.13 BPP). Per-document optimizer state resets interact badly with adaptive LRs. **Not universally dead:** lower-LR AdamW TTT works (#490, #731, #1006). The negative is LR-transfer from SGD to AdamW without re-tuning.
- **kNN-LM at eval time ([#1103](https://github.com/openai/parameter-golf/pull/1103), @abaybektursun).** Single-layer (k=64, L2, 2M store): +0.0026 BPP. Multi-layer (11 layers concatenated, cosine): +0.0031. XSA-all already captures the inter-position patterns that kNN-LM tries to exploit. From the SOTA holder, on the #1019 stack.
- **Sliding window logit averaging ([#1103](https://github.com/openai/parameter-golf/pull/1103)).** 32-window average: +0.024 BPP. Destroys sharp predictions. Catastrophically negative.
- **SelfExtend / extended context 4096 ([#1103](https://github.com/openai/parameter-golf/pull/1103)).** +0.48 BPP. The model was trained at seq2048; extending to 4096 at eval time causes massive degradation.
- **Mixed-precision GPTQ int4 attn / int8 MLP ([#1103](https://github.com/openai/parameter-golf/pull/1103)).** +0.047 BPP. Int4 attention weights lose too much information. Hessian sensitivity says MLP is more important, but int8 MLP still can't compensate for int4 attention.

**2:4 Structured Sparsity** (#1105): +0.672 BPB. Definitively dead at competition scale.

**Turbo-Muon on 8xH100** (#1105): +0.0018 BPP worse AND artifact over 16MB. Early convergence advantage at step 500 doesn't hold at 7000+ steps.

**SLOT causality concern** (#1105): @abaybektursun found shared delta optimized over all positions then applied to all positions leaks future tokens. Removed from their submission. However, #1176 and #1172 use SLOT successfully — implementation details may differ.

**1xH100 not a viable proxy** (#1186): 8x fewer optimizer steps means results don't transfer to 8xH100.

**SGD+momentum TTT** (#1186): +0.065 BPP vs AdamW. Use AdamW for TTT.

</details>


---

## The Current Baseline Stack

The foundation that most competitive submissions share. Worth noting: several top submissions diverge from consensus in specific ways that paid off — #180 used int5 (former official SOTA), #236 used 524K batch instead of 786K, #76 dropped QAT and raised LR, #265 added XSA from a recent paper. The meta is a strong starting point, but the data shows room to improve individual components.

**The core five:** Integer quantization (int6-all or int5-MLP/int6-attn) + MLP 3x expansion + sliding window eval (stride=64) + zstd-22 compression + precision passthrough for sensitive layers (usually FP16 tied embedding; #236 uses int8 to fund MLP capacity). Near-universal across all competitive submissions, though quant precision varies — #76, #267, and former SOTA #180 use int5-MLP to fund larger BigramHash or extra layers.

**Near-consensus optimizer settings:** Muon momentum 0.99 (warmup from 0.92 over 1500 steps), halved LRs (matrix=0.02, scalar=0.02, embed=0.03), warmdown 3000 iters, grad clip 0.3. Most top submissions use these. Exceptions: @unixmadtoonslab's [#76](https://github.com/openai/parameter-golf/pull/76) (1.1468) uses higher LRs (0.03) and lower momentum (0.97). @saml212's [#236](https://github.com/openai/parameter-golf/pull/236) (1.1400) used **524K batch** instead of 786K, gaining 0.017 BPB via more gradient updates. **However, #375's systematic study on the #315 frontier base found 786K > 524K by 0.004 BPB (3-seed)** — at the frontier, total tokens matter more than gradient steps. The optimal batch size is stack-dependent: 524K helps Tier 2-3 stacks; 786K helps XSA+EMA frontier stacks.

**Part of the top stack:** SmearGate + BigramHash + OrthoInit — used by most top validated entries. Requires OrthoInit to work (per #212's ablation). 11 layers + WD 0.04 + weight averaging (SWA or EMA). The standard-arch frontier (#414, 1.1228) builds on EMA + XSA4 + GPTQ-lite + Tight SWA + VE128 + Partial RoPE + LN Scale + Late QAT. The overall non-TTT frontier is now **#609 (1.1154**, XSA-all + Full GPTQ + Selective Pruning + Parallel Muon, @saml212).

**Common but not universal:** QAT with STE (~half), SWA (~17/49 validated), NorMuon (~3/49), FA3 (~13/49).

<details>
<summary><strong>The Core Five Explained (for newcomers)</strong></summary>

### 1. Int6 Quantization (instead of Int8)

Standard post-training quantization maps each weight to an 8-bit integer (256 levels). Int6 uses only 6 bits (64 levels, range [-32, 31]) with per-row scale factors, then compresses with zstd (level 22) instead of the baseline's zlib-9. Int6 frees ~25% more artifact space than int8, reinvested in a bigger model. Some submissions keep sensitive layers in fp16 (tied embedding) or int8 (embeddings) to limit compounding precision loss.

**Origin:** @nanlliu introduced int6 mixed precision in [#39](https://github.com/openai/parameter-golf/pull/39).

### 2. MLP 3x Expansion

The baseline uses 2x MLP expansion (hidden dim 1024 for 512-dim model). Top submissions use 3x (1536). Wider MLP = more expressive capacity, funded by int6 artifact savings.

**Origin:** @jfprincz in [#70](https://github.com/openai/parameter-golf/pull/70) (Mar 19 08:57 UTC). @saml212 independently reached the same insight in [#61](https://github.com/openai/parameter-golf/pull/61) later that day.

### 3. Sliding Window Evaluation

Overlapping windows (stride=64, window=2048) give each scored token 1984+ tokens of context vs minimal context with non-overlapping chunks. Purely eval-time. Worth **0.034 BPB** per @samacqua's ablation in [#77](https://github.com/openai/parameter-golf/pull/77).

**Origin:** @mattqlf in [#50](https://github.com/openai/parameter-golf/pull/50). **Stride debate:** stride=256 gives marginally better BPB at 4x less eval time (#114). **Doc isolation hurts at stride=64** — use flat-stream eval (#199).

### 4. FP16 Tied Embedding

The tied embedding matrix (input + output) is uniquely sensitive to quantization — errors compound in both directions. Keeping it in fp16 (~1MB) is the single highest-value precision decision.

**Origin:** @chonchiog in [#42](https://github.com/openai/parameter-golf/pull/42).

### 5. Zstd-22 Compression

Zstandard at level 22 squeezes int6 data significantly tighter than zlib-9 — enough to fit ~1-2M more parameters. Compression happens once after training; decompression is fast. Free lunch.

</details>

---

## The Path Down: What Separates Each Tier

Post-enforcement (Mar 27), the competition has bifurcated into two tracks: **n-gram cache + neural base** (record submissions from 0.44 to 1.05 BPB, though many face compliance scrutiny) and **pure neural frontier** (1.05-1.12 BPB). Official SOTA is now 1.1147 (#1019). Novel eval-time methods like TARA (#1055, closed for causality violation) and DeltaNet Crawler (#1047, 0.8822, causality concerns flagged) explored new directions but have not yet produced compliant results.

### Tier 1: Tweaking the Baseline (1.20–1.22 BPB)

Submissions in this range make one or two changes to the baseline: a longer sequence length, a learning rate sweep, a warmdown adjustment. The approach is **"how do I improve this model?"** — treating the baseline as mostly correct and looking for low-hanging fruit.

This works for the first 0.02 BPB, but hits a wall fast. The constraint isn't hyperparameters — it's the artifact budget. At int8+zlib, you can't fit enough model capacity to go further. Many submissions in this range are also on non-standard hardware (RTX 4090, Apple Silicon, 1xH100), which limits training tokens and disqualifies from the record track.

**What to do if you're here:** Adopt the core five (int6, MLP 3x, sliding window, FP16 embed, zstd-22) as a package. Each technique is well-documented in the deep dives below. Together they're worth ~0.05-0.07 BPB — the single biggest jump available.

### Tier 2: Stacking Known Techniques (1.15–1.18 BPB)

These submissions adopted the core five and are assembling additional techniques: SmearGate, BigramHash, SWA, QAT, NorMuon. The approach is **"what techniques exist and how do I combine them?"** — surveying PRs, identifying high-impact components, and building a combined recipe.

This is effective: the leap from 1.22 to 1.16 is largely a stacking exercise. But submissions in this range often stop at "I added all the techniques" without investigating *interactions*. Common patterns: using SmearGate without OrthoInit (which hurts — per #212's ablation), running QAT from the start (which hurts — late QAT at 70-85% is better), or using SWA without sufficient weight decay (SWA shows no effect below WD=0.04).

**What to do if you're here:** Run ablations. Remove one technique at a time and measure the delta. You'll often find that one "improvement" is actually hurting because of interaction effects. Check your hyperparameters against the consensus (LR=0.02, momentum=0.99, warmdown=3000) but also against divergent successes like #76 (LR=0.03, momentum=0.97). Multi-seed validation (3 seeds) is essential — single-seed scores can be off by 0.002+ BPB.

### Tier 3: Understanding Interactions (~1.120–1.15 BPB)

These submissions adopted the full technique stack and understood *why* each technique works. @jfprincz (#198 at 1.1326) is the canonical example: 11 layers + SmearGate + BigramHash + OrthoInit + WD 0.04 + SWA + FA3 assembled into a coherent system where each piece reinforces the others — WD makes weights compressible AND quantization-friendly, SmearGate+OrthoInit inject bigram context the small model can't learn from attention alone, and SWA smooths the weight landscape during warmdown.

The approach is **"how do these techniques interact, and what's the optimal system?"** Key markers of Tier 3 thinking:
- **Ablation-driven development** — every addition is measured, not assumed helpful
- **Precision budgeting** — spending fp16 only where quantization error hurts most (tied embedding, late-layer keys)
- **Divergent exploration** — #76 found that higher LR + lower momentum + no QAT outperforms consensus settings at 1.1468. #215 discovered that Q matrices are naturally low-rank (100M+ condition numbers) and factoring them saves 22% step time
- **Statistical rigor** — 3+ seeds, significance testing, honest evaluation

**What to do if you're here:** Solidify your baseline with multi-seed validation. The primary path to Tier 4 is adopting **XSA + Full GPTQ + EMA**. The #609 stack (XSA-all + Full GPTQ + Selective Pruning + Parallel Muon + LeakyReLU²) reached 1.1154 but is non-record due to eval-time GPTQ — the techniques are valid if GPTQ calibration is moved into the 600s training budget. The official record SOTA target is **#1019 at 1.1147**. XSA + EMA is the shared infrastructure across all frontier submissions.

### Tier 4: Pure Neural Frontier (<~1.120 BPB)

The official record SOTA is **#1019 at 1.1147** (@abaybektursun). The #609 stack reached 1.1154 (non-record due to eval-time GPTQ) and GEPA #505 reached 1.1181 (artifact >16MB). Both demonstrate what's achievable with compliant implementations.

The key insight at Tier 4: **EMA (0.997) outperforms standard SWA by 0.003 BPB** (#375, 3-seed verified). #315 demonstrates that the XSA+EMA base still had headroom via careful regularization — Partial RoPE, LN Scale, and Late QAT each target a specific weakness.

**What to do if you're here:** Three options. **(a) Beat #549 on pure neural:** Adopt the #609 technique stack with GPTQ calibration inside 600s training budget. Remaining untried: Mousse optimizer, OptRot, systems opts (Liger-Kernel, 2:4 sparsity). **(b) Add n-gram cache (→ Tier 5):** The single biggest lever — 0.07-0.16 BPB from a legal backward-looking n-gram eval cache. **(c) Legal TTT with compliant GPTQ:** All frontier TTT submissions were closed for eval-time GPTQ. The recipe works if GPTQ calibration fits in 600s. GEPA + legal TTT at **1.0983** on 4xA100 (#628, 20k steps) — 8xH100 version untried.

### Tier 5: N-gram Cache (Invalidated Mar 27)

The n-gram cache track was invalidated after discovery that hashed implementations scored only the correct token without full-vocabulary normalization. #978 proved correctly normalized n-gram achieves only 1.51 BPP (worse than neural baseline). 33+ PRs closed by @valerio-oai. Whether a *correctly normalized* eval-time statistical method can improve on pure neural remains an open question.
### Technique Interactions Matter More Than Technique Count

A recurring pattern: techniques that work independently can fail in combination. TTT+XSA actively hurts (#303: +0.016 worse), EMA fails without XSA (#201) but succeeds with it (#287), and 12L fails at seq2048 but works at seq1024 (#219 vs #76). **#474 confirms this extends to newer techniques:** VRL + Gated Attention + Catalytic Residuals stacked on a 12L SWA base (no XSA, no EMA) yielded **1.1690 — worse than the same base without them** (1.1466). Frontier techniques are optimized for the frontier base; applying them to weaker bases produces negative or null returns.

The untried combinations above should be evaluated against your specific model's weaknesses, not applied blindly. **XSA + EMA appears to be a prerequisite for most newer techniques** (VRL, GA). For the pure neural track, the strongest remaining candidates are **systems optimizations** (fused kernels, 2:4 sparsity — throughput gains with significance waived) and **compression innovations** (OptRot, entropy-coded weights). For the overall frontier, **n-gram eval cache** is by far the highest-impact lever available.

<details>
<summary><strong>Val-Data & TTT Rulings (Mar 20-28)</strong></summary>

**Val data ruled out (Mar 20, @0hq):** [Val tokens cannot be in the artifact](https://github.com/openai/parameter-golf/pull/262). Paid prefix (#168), error correction (#108), val-only training all banned for record track. Now in README FAQ.

**TTT ruling (Mar 20, @0hq on [#152](https://github.com/openai/parameter-golf/pull/152)):** Only backward-looking TTT allowed — adapt on tokens *already graded*, not future tokens. Pre-eval adaptation invalid. Causal TTT (#267-style) remains allowed. In README FAQ.

**Mar 22, @cocohearts on [#317](https://github.com/openai/parameter-golf/pull/317):** TTT is "not in the spirit of the challenge." Broader organizer signal — even backward-looking TTT may face scrutiny.

**Mar 23, @0hq on [#402](https://github.com/openai/parameter-golf/issues/402):** Explicit TTT clarification — **token-stream model is correct.** You may use any preceding eval tokens already graded. You may NOT re-order the evaluation set. Invalid TTT PRs (train-on-val-then-measure) will be closed. Auto-review process being built.

**Mar 23, @cocohearts:** #374 rejected for insufficient statistical significance vs new SOTA. #505 needs packaging fixes.

**Mar 24, @valerio-oai — enforcement sweep (15+ PRs closed).** Two categories: **(1) TTT information leakage:** multi-epoch TTT with min-NLL selection, and adapting-then-scoring same tokens, both ruled equivalent to "training on the val set." #593, #576, #573, #568, #596, #605, #614, #620, #518, #548 closed. **(2) Training data at eval time:** GPTQ calibration using training data during eval budget disallowed. #593, #576, #569 closed for this. Calibration must count within training 600s. **#589 ruled valid but closed** — fails 0.005-nat threshold vs #549 SOTA. valerio-oai confirmed: "TTT is a valid approach in theory" but "very easy to unintentionally leak val data into."

**Mar 25, @valerio-oai — second enforcement sweep (issue [#677](https://github.com/openai/parameter-golf/issues/677)).** Comprehensive audit. **(1) Eval-time GPTQ:** Training for full 600s then doing GPTQ calibration afterward (even 3-4s) is "accessing training data at eval time" — disallowed. **#606, #615, #626, #639, #656 closed.** **(2) N-gram eval cache ruling:** The concept is "directionally legal" — building a cache from already-scored tokens is allowed. The specific #659 implementation was illegal (hindsight selection: comparing n-gram vs LM on the true next token). Legal alternatives: fixed-weight blending or entropy-adaptive alpha (using model uncertainty, not ground truth). **(3) #706 flagged:** @valerio-oai told @newjordan that #706's GPTQ calibration still runs after 600s training time — needs fix. **(4) Broad invalid TTT list:** #410, #415, #417, #442, #462, #481, #486, #517, #518, #532, #555, #581, #595 all flagged for adapting on validation before the reported eval pass.

**Mar 27, @valerio-oai — mass n-gram cache closure (33+ PRs, [#677](https://github.com/openai/parameter-golf/issues/677)).** Hashed n-gram caches disallowed: they score only the correct token via hashing without normalizing over the full token distribution, producing invalid probabilities. Two-pass rescoring (score → TTT → rescore) explicitly disallowed as "training on the eval set." PRs closed include #846, #853, #868, #869, #870, #876, #881, #888, #893, #900, #907, #912, #918, #982, and many more. **Only reviewing PRs after #988** for potential merging. Recommended: @NoesisGenesis's formal criteria for valid causal prediction — (a) distribution depends only on artifact + strict prefix, (b) full normalized distribution over token vocabulary required before scoring, (c) score computed from pre-update probability only, (d) single left-to-right pass.

**Mar 28, @valerio-oai on [#728](https://github.com/openai/parameter-golf/pull/728):** Val-calibrated GPTQ "breaks autoregressivity" — disallowed. Self-generated calibration data (as in #1019) is "probably legal." PR left open pending fix.

**Mar 28, @valerio-oai on [#991](https://github.com/openai/parameter-golf/pull/991):** Closed for double-pass TTT (score, train, rescore same tokens).

**Mar 28, @valerio-oai on [#1028](https://github.com/openai/parameter-golf/pull/1028):** GPTQ calibration running after 600s training cap = accessing training data at eval time. Author confirmed bug and resubmitted as #1047 with GPTQ within budget. Under organizer review.

</details>


---

## Technique Deep Dives


<details>
<summary><strong>The Muon Optimizer Family</strong></summary>

**Muon** (MomentUm Orthogonalized by Newton-Schulz) is the optimizer at the heart of this competition's baseline, created by Keller Jordan for the NanoGPT speedrun. It runs standard SGD with Nesterov momentum, then post-processes each 2D parameter's gradient update by replacing it with the nearest orthogonal matrix via Newton-Schulz iteration. Intuitively: compute the gradient direction, then "clean it up" so the update is maximally informative without redundant directions. It's equivalent to steepest descent under the spectral norm, which improves the conditioning of the optimization landscape. ~35% faster training than AdamW on language models.

**NorMuon** extends Muon by adding per-neuron adaptive learning rates from accumulated second-order statistics. Vanilla Muon can produce updates with highly non-uniform norms across neurons, causing some neurons to dominate training. NorMuon normalizes row-wise after orthogonalization, combining Muon's conditioning benefits with Adam-style balanced per-neuron learning. It also improves distributed scaling by avoiding full momentum gathering across GPUs. Used by @mtybadger ([#122](https://github.com/openai/parameter-golf/pull/122)), @vmfunc ([#89](https://github.com/openai/parameter-golf/pull/89)), @abhishekgahlot2 ([#137](https://github.com/openai/parameter-golf/pull/137)), and others.

**Muon Weight Decay** — The competition baseline's Muon optimizer has no weight decay. Decoupled weight decay for Muon (`p.mul_(1 - wd * lr)`) existed in modded-nanogpt since Nov 2025, but wasn't in the baseline. @notapplica was the first to bring it into this competition in [#60](https://github.com/openai/parameter-golf/pull/60), improving BPB from 1.2160 to 1.2094. Weights stay smaller and better-distributed, improving both generalization and compressibility.



**Post-enforcement status (Mar 27):** The Mar 27 enforcement sweep closed 33+ n-gram cache PRs after discovery of the normalization bug — implementations scored only the correct token without full-vocabulary normalization, producing artificially low BPP. #978 proved that properly normalized n-gram achieves only 1.51 BPB (worse than neural baseline). The question of whether a *correctly normalized* eval-time statistical method can improve on pure neural remains open.

</details>

<details>
<summary><strong>Quantization-Aware Training (QAT) with STE</strong></summary>

Instead of training in full precision and quantizing afterward, QAT simulates quantization during training. In the forward pass, weights are rounded to their quantized values. The problem: rounding is non-differentiable, so gradients can't flow through it.

The **Straight-Through Estimator (STE)** solves this by pretending the rounding operation is the identity function during the backward pass. It's mathematically "wrong" but works remarkably well — the model learns weight configurations that are robust to precision loss because it's been "seeing" quantized weights throughout training.

**Late QAT outperforms full-training QAT:** The later, the better. @trovatochris ([#117](https://github.com/openai/parameter-golf/pull/117)) activates at 70%, @mohosy ([#130](https://github.com/openai/parameter-golf/pull/130)) at 75%, @unixmadtoonslab ([#76](https://github.com/openai/parameter-golf/pull/76)) at 85%. #76 even dropped QAT entirely at 12L (1.1468), finding WD=0.04 alone sufficient. @jfprincz's [#315](https://github.com/openai/parameter-golf/pull/315) pushes this to the extreme: STE activates only in the final **4% of training** (lr_scale < 0.1, during low-LR warmdown). This cuts the int6 roundtrip gap to ~0.007 BPB while preserving full-precision convergence. The lesson: QAT activation is a spectrum — later = cleaner convergence, better int6 gap.

**Int8 vs int6 QAT tradeoff:** @mrdavtan's ablation in [#145](https://github.com/openai/parameter-golf/pull/145) shows that **int8 QAT is not worth it** under the 10-min wallclock cap. The `torch.quantile` call for exact percentile matching adds ~20% per-step overhead (64ms → 77ms), costing ~2,000 training steps. Result: 1.2052 BPB with QAT vs 1.1925 without — the lost training tokens hurt more than closing the ~0.007 int8 quantization gap. Int6 QAT, however, likely pays off because its larger ~0.01+ BPB gap justifies the overhead — confirmed by #128 and #137.

</details>

<details>
<summary><strong>SmearGate & Bigram Hash Embedding</strong></summary>

@unnir introduced SmearGate in [#102](https://github.com/openai/parameter-golf/pull/102) and refined it in [#135](https://github.com/openai/parameter-golf/pull/135). This appears to be a novel technique for this competition — no published papers found.

**SmearGate:** A tiny learned gate (~512 params) that blends each token's embedding with the previous token's. This injects bigram (two-token) context directly into the embedding layer before the transformer starts processing. Normally a transformer must discover token pair relationships through self-attention; SmearGate provides this signal for free.

**Bigram Hash:** A hash table (commonly 2048-10240 buckets, dim=128, projected to 512) that maps token pairs to learned embeddings. Together with SmearGate, this gives the model token-pair awareness at nearly zero parameter cost.

@unnir's original combination with orthogonal initialization achieved **1.1539 BPB** in [#135](https://github.com/openai/parameter-golf/pull/135). @jfprincz's #198 (1.1326) extended this with 11L + SWA + FA3 + WD 0.04, and #287 (1.1280) extended further with XSA + EMA.

**OrthoInit appears critical for SmearGate.** @mrdavtan's ablation in [#212](https://github.com/openai/parameter-golf/pull/212) found that adding SmearGate + BigramHash without OrthoInit **hurt** BPB (1.1739 vs 1.1708 without). Every successful SmearGate submission uses OrthoInit — the two techniques may be co-dependent.

</details>


<details>
<summary><strong>Exclusive Self-Attention (XSA)</strong></summary>

XSA ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078), Shuangfei Zhai, 2026) removes self-value bias from attention output via orthogonal projection. In standard attention, each token's value vector contributes to its own output — XSA subtracts this self-component, forcing the model to rely on information from *other* tokens. Applied to the last 3-4 layers only ("Partial XSA"), where self-attention bias is highest.

**Zero parameters, minimal overhead.** @unnir's [#265](https://github.com/openai/parameter-golf/pull/265) GQA-aware implementation reduces XSA overhead from ~7ms/step to ~2ms/step. Near-universal among frontier submissions. Best non-TTT (#609, 1.1154) uses XSA on all 11 layers; official SOTA (#1019, 1.1147) uses XSA-all.

**XSA coverage depth: 4 layers appears near-optimal.** @gowtham0992's [#478](https://github.com/openai/parameter-golf/pull/478) tested XSA on ALL 11 layers: **1.1268** (3-seed) vs XSA-4 at 1.1327 on the same base (−0.006 from XSA-all). But #414 (XSA-4 + VE128 + Partial RoPE + LN Scale) reaches 1.1228 — better than #478's XSA-all(11) at 1.1268. XSA-all adds ~3ms/step overhead (−230 steps), and removing self-value from ALL layers may degrade the model's own-representation capacity. **The progression: 3 layers (#265: 1.1307) → 4 layers (#414: 1.1228) → 11 layers (#478: 1.1268) suggests 4-6 layers is the sweet spot for non-TTT.** However, **#609 (1.1154, best non-TTT) uses XSA-all(11)** and **#606 (1.1162, best legal TTT) also uses XSA-all** — at the current frontier, XSA-all with Full GPTQ overcomes the overhead penalty.

</details>

<details>
<summary><strong>Test-Time Training (TTT)</strong></summary>

@samacqua introduced a creative approach in [#77](https://github.com/openai/parameter-golf/pull/77): adapting the model *during evaluation*.

For each validation document, rank-8 LoRA (Low-Rank Adaptation) adapters are trained on the document's own text using only backward-looking context (no data leakage). The model essentially "studies" each document briefly before being scored on it. LoRA makes this practical by only training tiny low-rank matrices (~1.5% of params) rather than the full model, enabling batched per-document adaptation within the eval time budget.

Original #77 ablation showed TTT itself adds ~0.003 BPB on early baselines (most gain came from doc isolation + sliding window). Full-model SGD TTT (#152) was **ruled invalid by @0hq** — only backward-looking (score-first) TTT is legal. The best legal TTT submissions (#606 at 1.1162, #615 at 1.1169) were later closed for eval-time GPTQ on training data (Mar 25 sweep).

**TTT on XSA+EMA is a spectrum, not a binary.** On SmearGate bases: #254 shows 0.014 BPB gain. Three XSA+EMA data points, sorted by base strength: (1) **#317** (weak base, pre-quant 1.1581, no FA3): TTT **gains 0.024 BPB**. (2) **#338** (@alertcat, #315 base — frontier at 1.1250, Partial RoPE + LN Scale + Late QAT): TTT **neutral ±0.001** (3 seeds). (3) **#303** (@sseanliu, #287 base — 1.1280, without #315's additional regularization): TTT **+0.016 BPB worse**. The pattern suggests TTT interacts with how tightly converged the base model is: under-trained bases benefit from local adaptation; over-regularized frontier bases are disrupted; the current frontier (#315) sits in a neutral zone. #338's neutral result is informative — it means TTT is not a meaningful lever at the frontier.

**Reptile meta-TTT: gains on SmearGate, fails at frontier.** @sseanliu's [#296](https://github.com/openai/parameter-golf/pull/296) shows 0.011 BPB on SmearGate models vs 0.001 naive. But **#375 tested Reptile on #315's XSA+EMA base: +0.0076 worse**, consuming 20% of training budget. The SmearGate gain does not transfer to the frontier. All three TTT variants (naive, MLP-only, Reptile) are now confirmed dead ends at ~1.125. **Error-guided TTT is also negative** — hardest tokens are genuinely unpredictable.

**TTT optimizer recipe matters.** @Christopher-Lee-McClendon's [#461](https://github.com/openai/parameter-golf/pull/461) (non-record, 4xA100) found that **SGD+momentum(0.9), 3 epochs per 32K chunk, freezing first 2 blocks** gets −0.0165 BPB TTT gain — **2.4× better** than AdamW 1-epoch over all params (−0.0068 in their prior #456). Pre-TTT baselines nearly identical, so the entire improvement comes from the TTT recipe. This partially contradicts the #442 narrative (AdamW >> SGD) — the comparison is more nuanced: selective freezing + multi-epoch SGD with momentum can outperform single-epoch full-network AdamW.

**Legal TTT survivors — none remain after Mar 25 sweep.** #606 (1.1162) and #615 (1.1169) were closed — eval-time GPTQ calibration on training data. #576 (1.1164) closed in Mar 24 sweep. #573 (Multi-Pass min(NLL)) ruled invalid. **All frontier TTT submissions used eval-time GPTQ and are now invalid.** TTT optimizer matters for GPTQ: SGD TTT hurts Full GPTQ models (+0.030, #601), but AdamW with cosine LR works. The remaining legal TTT avenue requires GPTQ calibration within the 600s training budget.

**Cosine TTT scheduling is a 3× multiplier.** @mrdavtan's [#481](https://github.com/openai/parameter-golf/pull/481) (3-seed, **1.0970**) introduced two TTT innovations on top of AdamW TTT: (1) **cosine LR decay** over 30 epochs — high LR early to repair quant damage, low LR late to refine; (2) **per-layer LR groups** based on measured quantization error — 3× base LR for MLP output projections (3.4× higher quant error), 0.5× for input projections. Result: TTT gain of **−0.061 BPB** vs #442's −0.019 with flat LR — a **3× improvement from scheduling alone**. Pre-TTT ~1.158 (weaker base, FA2 not FA3). Also tested: focal loss and KL-divergence from pre-quant model — both failed to improve over CE. ⚠️ Pre-eval TTT.

</details>

<details>
<summary><strong>N-gram Eval Cache (the Mar 25 revolution)</strong></summary>

The single biggest BPB lever discovered in the competition. During sliding window evaluation, a backward-looking n-gram cache is built from already-scored tokens and mixed with model predictions. The concept is simple: if the model has already scored "the cat sat on the", and the 5-gram "cat sat on the" was followed by "mat" last time, weight that prediction into the next token's distribution.

**How it works:**
1. After scoring each token, record the preceding n-gram context and the actual next byte
2. For new tokens, look up the current n-gram context in the cache
3. If found, mix the empirical n-gram distribution with the model's distribution: `p_final = (1-α) * p_model + α * p_ngram`
4. Store counts in a hash table (count-min sketch, ~4M buckets)

**Three generations of implementation:**
- **Fixed 5-gram, fixed alpha** (#706, @newjordan): alpha=0.20 always. Simple. Drops BPB by ~0.07 (1.1202→1.0461).
- **Multi-order backoff 2-7** (#702, @lukacf): Try 7-gram first, cascade to 6,5,4,3,2 on miss. Coverage jumps dramatically. Additional −0.02 over fixed 5-gram.
- **Entropy-adaptive alpha** (#727, @Asukabot0): `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`. When the model is uncertain (high entropy), trust n-grams more. Additional −0.02 over fixed alpha. Combined with backoff: 1.1271 neural-only → 0.9674.

**Why it's so effective:** Language has enormous local repetition — names, technical terms, formatting patterns — that a small transformer can't memorize but n-grams capture perfectly. The n-gram cache acts as a lossless "local memory" that costs zero artifact bytes (built on-the-fly from eval data).

**Legality argument:** All implementations claim score-first backward-looking compliance — the cache uses only previously-scored tokens, alpha depends on the model's own entropy (not ground truth), and there's no oracle selection. #702 cites @valerio-oai suggesting entropy-adaptive alpha as a legal alternative in the #659 review. But the technique hasn't been officially ruled on yet.

**#738 adds kNN-LM:** @gowtham0992 stores 512-dim hidden states in a GPU ring buffer and finds k=32 nearest neighbors for uncertain tokens. RBF kernel builds a non-parametric distribution. Additive −0.007 BPB on top of n-gram cache. Based on Khandelwal et al. 2019 (ICLR 2020). Captures semantic patterns that pure n-gram statistics miss.

**Ablation data from #727:**

| Configuration | val_bpb | Delta |
|---|---|---|
| Neural only | 1.1271 | baseline |
| Fixed alpha=0.40, order=7 | 1.0336 | −0.094 |
| Multi-order backoff (2-7) + fixed alpha | 0.9825 | −0.145 |
| Multi-order backoff + entropy-adaptive | 0.9674 | −0.160 |

</details>

<details>
<summary><strong>#315's Techniques: Partial RoPE, LN Scale (Late QAT was inactive)</strong></summary>

@jfprincz's [#315](https://github.com/openai/parameter-golf/pull/315) (1.1250) adds two effective zero-parameter techniques on top of #287's XSA+EMA base, gaining 0.0023 BPB. **Note:** Late QAT was also included in the code, but `torch.compile` constant-folded the `_qat_enabled` flag, making the STE branch dead code — Late QAT never activated (discovered by @152334H, confirmed in #453). **Update:** @wfproc ([#1032](https://github.com/openai/parameter-golf/pull/1032)) confirmed this dead-code bug persists in the current SOTA #549 codebase. A fix via tensor-scale STE actually worsened the int6 gap — suggesting WD+EMA already compensate for what QAT was supposed to do. The 0.0023 gain comes entirely from Partial RoPE + LN Scale.

**Partial RoPE (16 of 64 head dimensions).** Rotary Position Embedding (RoPE) injects position information by rotating query/key vectors. Standard RoPE applies to all head dimensions. Partial RoPE applies to only 25% (16 of 64 dims) — the remaining 48 dims attend without position encoding. Why this helps: the position-free dims learn semantic similarity independent of token distance, improving generalization across different position ranges. The model can learn both "what things are" (position-free) and "where things are" (position-encoded) using different parts of the same head. Zero new parameters.

**LN Scale (output scaled by 1/√(layer_idx+1)).** After each RMSNorm, the output is multiplied by a layer-dependent scale factor that shrinks with depth. Layer 0: ×1.0; Layer 5: ×0.408; Layer 10: ×0.302. This damps the contribution of deeper layers to the residual stream, preventing later layers from "overwriting" early representations. Training is more stable — the model can use depth incrementally rather than being forced to route everything through deep layers. The 1/√(layer+1) schedule is related to the "depth scaling" used in some architecture papers. Zero new parameters.

**Late QAT (STE enabled only when lr_scale < 0.1) — ⚠️ was dead code in #315.** `torch.compile` constant-folded the `_qat_enabled` class attribute, so the STE branch never activated (discovered by @152334H, confirmed in #453). The concept is sound — late activation avoids corrupting Muon's momentum — but #315's actual gains came from Partial RoPE + LN Scale alone. **Working Late QAT:** @unnir (#374, scale<0.1), @signalrush (#414, threshold 0.15), @fbedev (#417). Downstream submissions copying #315's code may also have inactive Late QAT.

The two active techniques (Partial RoPE + LN Scale) gain 0.0023 BPB vs #287 — statistically clear (3-seed variance 0.0005 BPB, t-stat -101.9 vs SOTA, p << 0.01).

</details>

---

<details>
<summary><strong>Notable Non-Record Submissions</strong></summary>

| Author | PR | Highlight |
|--------|-----|-----------|
| @mohosy | [#130](https://github.com/openai/parameter-golf/pull/130) | 7 toggleable improvements; QAT + Muon momentum analysis |
| @MatoTeziTanka | [#95](https://github.com/openai/parameter-golf/pull/95) | PROTEUS EMA — reduces int8 quant loss 0.0072→0.0048 |
| @nglain | [#141](https://github.com/openai/parameter-golf/pull/141) | 33-experiment sweep; found int6 STE + Muon conflict (+0.007) |
| @kellyvv | [#108](https://github.com/openai/parameter-golf/pull/108)/[#232](https://github.com/openai/parameter-golf/pull/232) | **Error Correction Table** — stores model's worst predictions, ~1.05 est. on 8xH100 |
| @mrdavtan | [#145](https://github.com/openai/parameter-golf/pull/145) | Int8 QAT ablation — overhead exceeds recovery |
| @timothywangdev | [#220](https://github.com/openai/parameter-golf/pull/220) | [WIP] First SSM (Linear Recurrent Unit) — non-transformer architecture |
| @mkenney2 | [#599](https://github.com/openai/parameter-golf/pull/599) | **Hymba: Hybrid Attention + Mamba SSM** (first competitive non-transformer). 7L parallel attn+SSM branches with learned mixing. **1.1828 BPB**, 3 seeds, 8xH100. Key: shallow models win (SSM makes each layer more powerful → 7L beats deeper pure transformers at same step budget). |
| @alons23 | [#216](https://github.com/openai/parameter-golf/pull/216) | Ternary Universal Transformer — 68M params, 4×6 depth recurrence |
| @Cwarren15-A | [#283](https://github.com/openai/parameter-golf/pull/283) | **PPM-C context mixer** — classical compression blended with neural (0.015 BPB on baseline) |
| @sseanliu | [#296](https://github.com/openai/parameter-golf/pull/296) | **Reptile meta-TTT** — 0.011 BPB gain on SmearGate models (10x naive TTT). Error-guided TTT negative. |
| @integrate-your-mind | [#289](https://github.com/openai/parameter-golf/pull/289) | 11L seq1024 + U-Net skips (1.1518). TTT LoRA *worse* than sliding window alone on this base. |
| @gowtham0992 | [#295](https://github.com/openai/parameter-golf/pull/295) | **Backout** (learned residual subtraction) + mixed int5/int6 QAT + U-Net skips (1.1477, 1 seed) |
| @JackYoung27 | [#302](https://github.com/openai/parameter-golf/pull/302) | **Online causal TTT + decay prior** (`p += λ(p₀-p)`) + Reptile (last 10%) + XSA3 + Pre-Q/K RMSNorm. TTT gain: **-0.014 BPB** (1.1660→1.1520). Adapts MLP only in last 3 blocks. Int5-MLP/int6-attn + BigramHash(10240). 1 seed. |
| @xuafeng | [#306](https://github.com/openai/parameter-golf/pull/306) | QAT Int5/Int6 on #180 base: **post-training quant outperforms QAT by ~0.002 BPB** — quant noise acts as beneficial regularization that QAT removes (1.14476, 1 seed) |
| @NewyorkDev | [#309](https://github.com/openai/parameter-golf/pull/309) | CLASE-Quant adaptive per-layer quantization: int8 for boundary layers, int6 for middle — saves ~15% vs uniform int8 (1.1914, 3 seeds) |
| @chanwoo-park-official | [#312](https://github.com/openai/parameter-golf/pull/312) | **Canon ACD layers** (Allen-Zhu 2025) on 9L stack — learnable 1D conv (k=3) placed before attention, before MLP, and in MLP hidden stream (avoids QKV=B for cost). 1.1668, 1 seed. Novel architecture technique; interesting if it scales to 11L. |
| @SkywardSyntax | [#316](https://github.com/openai/parameter-golf/pull/316) | 12L Low-Rank Q (r=128) + QAT int7 on 1xH100 (pre-quant 1.2035, awaiting 8xH100). Key negative result: **FTLE per-row precision is a dead end** — uniform int-N beats mixed-row at every bit width due to higher entropy defeating zstd. Layer sharing also abandoned at 512d (costs 0.09 BPB, no space benefit). |
| @aravhawk | [#314](https://github.com/openai/parameter-golf/pull/314) | 11L Int4 MLP QAT on #180 base — int4 MLP saves ~2MB to fund 11th layer vs #180's 10L int5. Awaiting 8xH100 results. Record track aspirant. |
| @Rhodrium | [#331](https://github.com/openai/parameter-golf/pull/331) | 10L MLP3x + BigramHash(2048) + SmearGate + OrthoInit + mixed int5/int6 + SWA + **stride=32** eval. 1.1487 BPB, 3 seeds. Solid consensus stack; above SOTA but clean stride-32 reference on H100s (94/91ms/step). |
| @sheeki03 | [#339](https://github.com/openai/parameter-golf/pull/339) | **Backout ablation**: -0.0071 BPB on #198 base (1.1435→1.1364). First clean measurement. ⚠️ artifact 16.17MB (over limit), 1 seed. Plans int5-MLP fix + XSA/EMA combo. |
| @Ananddna | [#327](https://github.com/openai/parameter-golf/pull/327) | **TrigramHash** (8192 buckets) + Partial RoPE (50%) + **per-head temperature scaling** + stride=32 eval. 1.1450, 2 seeds. Three novel techniques on 10L int5 base. |
| @mahsumaktas | [#333](https://github.com/openai/parameter-golf/pull/333) | **23-run systematic exploration** (1.1565, 3 seeds). Key findings: seq curriculum fails (SWA incompatible across seq lengths), EMA causes 0.14 BPB quant gap on SWA-stack, MLP 2.75x sweet spot at 11L+SmearGate, Late QAT 75% cuts quant gap 0.023→0.006. |
| @sseanliu | [#318](https://github.com/openai/parameter-golf/pull/318) | **Neural Cache** research proposal — maintain per-layer KV cache across sliding windows, extending effective context from 2K to 50K+. Zero artifact cost, backward-looking compliant. Untested (torch.compile state bug). Proposed on #287 base (1.1284). |
| @fbedev | [#348](https://github.com/openai/parameter-golf/pull/348) | QAT + BigramHash(12288) + stride=32 on #180 base. 1.1444, 1 seed. Barely above SOTA — diminishing returns from BigramHash >10240. |
| @sp00mm | [#352](https://github.com/openai/parameter-golf/pull/352) | **Memory Tokens**: 64 learnable embeddings as global context scratchpad. A/B: **-0.014 BPB**. Uses #315 stack + MTP aux heads. 1.1659, 1 seed. |
| @jackopenn | [#336](https://github.com/openai/parameter-golf/pull/336) | **Hypernetwork prototype** — shared-trunk MLP generates full GPT weights from compact conditioning vectors (9.34x compression, 26.5M target params from 2.8M hypernet params, 2.09MB artifact). No BPB result yet. Highest compression-ratio weight-generation approach seen. |
| @mkenney2 | [#362](https://github.com/openai/parameter-golf/pull/362) | 11L SmearGate+BigramHash(4096)+EMA+OrthoInit, WD=0.02, stride=256. 1.1497 (3-seed). Key negatives: AttnRes -54% throughput, seq curriculum compile overhead, depth recurrence, 13L+TTT compression. |
| @shikhar1729 | [#364](https://github.com/openai/parameter-golf/pull/364) | **524K batch on #180 base** — 1.1497 (3-seed). Validates 524K batch benefit: more optimizer steps per wall-clock minute. |
| @charmquark1984 | [#375](https://github.com/openai/parameter-golf/pull/375) | **$500 systematic frontier study.** 13 techniques on #315 base, all failed. Reptile +0.008 worse. EMA>SWA +0.003. 786K>524K +0.004. See What Doesn't Work. |
| @anthony-maio | [#376](https://github.com/openai/parameter-golf/pull/376) | 9L + full stack + custom Triton/CUDA kernels (fused RMSNorm+QKV 1.47×, fused ReLU² MLP 1.26×). 1.1401, 1 seed. 125ms/step (4,782 steps). Kernel pipeline in dev for next submission. |
| @abaybektursun | [#399](https://github.com/openai/parameter-golf/pull/399) | **First Muon systems optimization.** Parameter Banking + Polar Express + Parallel Muon = 82.14ms/step (−3.1% vs #315's 84.76ms, +227 steps). Lossless — identical pre-quant 1.1421. ⚠️ Artifact 20.4MB (packaging issue). Significance waived for systems-only. |
| @anantdgoel | [#384](https://github.com/openai/parameter-golf/pull/384) | 3 research directions: **MAML Meta-TTT** = +0.085 worse (5th dead TTT variant). **Eval stacking** (cache + OGD on vocab bias): −0.003 additive, zero artifact cost. **Tokenizer v8192**: null result — longer tokens harder to predict, offsetting compression. 1xA40, 1.2882. |
| @anantdgoel | [#413](https://github.com/openai/parameter-golf/pull/413) | **Value Residual: −0.015 BPB** (dev). Gated Attention: −0.003. Stack additively (−0.017). PPM-C: +0.002 (negative). 9L dev-scale, 1xRTX3090. |
| @anantdgoel | [#487](https://github.com/openai/parameter-golf/pull/487) | **VRL+GA on 11L production stack** (1xA6000, 14.5hr). **1.1720 BPB**, 19.4MB (over limit). Confirms dev ablation (−0.017 additive). Not 8xH100 — VRL on 8xH100 frontier still untested by originator (#486 by @ndokutovich tested VRL+Cosine TTT at 1.0887). |
| @zachgoldfine44 | [#450](https://github.com/openai/parameter-golf/pull/450) | **12L + Catalytic Residuals** (novel: `x + c*f(x)`, learned per-dim vector c). −0.024 BPB at zero overhead. 3-seed mean 1.1466. Built on #180. |
| @Christopher-Lee-McClendon | [#461](https://github.com/openai/parameter-golf/pull/461) | **High-yield legal TTT**: SGD+momentum(0.9), 3 epochs per 32K chunk, freeze first 2 blocks. TTT gain: **−0.0165** (2.4× better than AdamW 1-epoch). Depth recurrence (11L from 10 cores). 1.14458, 4xA100. |
| @joshuaswarren | [#474](https://github.com/openai/parameter-golf/pull/474) | **First VRL+GA+Catalytic Residuals stack** on 12L + BigramHash(10240) + SWA + Late QAT. 1.1690 — disappointing vs #450's 1.1466 (same base without VRL/GA). Techniques don't stack additively here: no XSA, no EMA → weak base dilutes gains. |
| @leofeasby | [#470](https://github.com/openai/parameter-golf/pull/470) | **Shared-weight transformer** (single block × 9 passes) + U-Net skips + extended warmdown. 1.1454, 2.3hrs 8xH100. Key finding: improvement continues steadily throughout low-LR warmdown — no plateau observed. |
| @LoquiAuris | [#465](https://github.com/openai/parameter-golf/pull/465) | **Int6 embedding quantization**: +0.0005 BPB penalty — essentially free. Systematic tokenizer study: sp8192 d=512 8L (1.1794) vs sp1024 d=512 10L (1.1508) — more layers > tokenizer efficiency. 3-seed std=0.00012. |
| @carlesonielfa | [#457](https://github.com/openai/parameter-golf/pull/457) | 11L + XSA + **VRL (Value Residual Learning)** + SWA + seq4096 + cross-doc TTT. 1.1839 (int8+zlib). Another VRL adopter. |
| @AnirudhRahul | [#511](https://github.com/openai/parameter-golf/pull/511) | **Delayed PPM eval-time bank** on #180 base. Classical n-gram backoff (C trie) with 2048-token delay — only sees tokens outside transformer's window. **−0.00126 BPB (p=0.000041, 3-seed)** — real but below 0.005-nat record bar. Zero artifact cost, composable with any model. First positive classical compression result at frontier. |
| @Robby955 | [#484](https://github.com/openai/parameter-golf/pull/484) | **TTT Memorization Analysis** (updated from EBLS). Diagnostic: 3-epoch TTT adapted weights score **1.0476** via sliding window (genuine adaptation). **At 10 epochs: 0.8566 TTT-loop / 0.9229 sliding — both below ~0.95 theoretical floor = memorization.** Implication: #512's 0.95 seeds are likely memorization artifacts, not real gains. Also: MLP weights are layer-invariant (EBLS gammas → 0). |
| @Christopher-Lee-McClendon | [#598](https://github.com/openai/parameter-golf/pull/598) | **7000-step GEPA** (4xA100). Extended warmdown + mixed int6/int8 + legal TTT. **1.1334 BPB.** |
| @Christopher-Lee-McClendon | [#628](https://github.com/openai/parameter-golf/pull/628) | **Sub-1.10 GEPA** (4xA100, 20k steps). 8k warmdown + int6 GPTQ-lite + legal TTT. **1.0983 BPB.** Scaling law: warmdown is dominant lever. |
| @SPThole | [#623](https://github.com/openai/parameter-golf/pull/623) | **First AWQ in competition** — activation-aware weight scaling (α=0.5) before quant. Closed 63% of quant gap (0.027→0.010). Cyclic Muon Momentum (triangle wave 0.85-0.95). 21+ experiments. **1.1507, 3-seed.** |
| @greqone | [#1044](https://github.com/openai/parameter-golf/pull/1044) | **H-Net:** First learned byte-level tokenization (README wishlist). Differentiable chunking gate discovers ~4-byte segments. 22M params, **1.90 BPB** (4090, 2.8hr). |
| @ikermoel | [#1053](https://github.com/openai/parameter-golf/pull/1053) | **Masked Diffusion (MDLM):** First text diffusion submission (README wishlist). Bidirectional attention, pseudo-log-likelihood eval. **1.3600 BPB**, 3-seed, 12.9MB. High school student's 2nd ML competition. |
| @andrewmouldon | [#1035](https://github.com/openai/parameter-golf/pull/1035) | **ASQU:** Per-channel learned asymmetric activation. Consistent -0.0011 BPB vs LeakyReLU² across 3 seeds. |
| @mrdavtan | [#1048](https://github.com/openai/parameter-golf/pull/1048) | **Compression moonshots:** 8 negative findings. Procrustes (91% MSE reduction but 380% larger artifact), pruning+zstd non-monotonic, selective fp16. Key: int6+zstd is near-optimal for this arch. |
| @himanshudongre | [#1012](https://github.com/openai/parameter-golf/pull/1012) | **JEPA-LM negative result:** -19.5% CE on synthetic Markov chains but only -0.24% on real text. +40% throughput overhead makes it net-negative. Valuable lesson: synthetic benchmarks don't transfer. |
| @wfproc | [#1032](https://github.com/openai/parameter-golf/pull/1032) | **QAT dead-code confirmed in SOTA #549** (torch.compile constant-folds Late QAT). 7 techniques all negative. Heuristic: 1ms overhead = 0.007 BPP at 83ms/step. 1xH100 research. |
| @DbBested | [#1108](https://github.com/openai/parameter-golf/pull/1108) | **nGPT Hypersphere:** Fixed 3 bugs that killed earlier attempt (#831). Normalized transformers viable at 16MB: 1.6915→**1.2714 BPP**. Research contribution. |
| @serdardoesml | [#1088](https://github.com/openai/parameter-golf/pull/1088)/[#1110](https://github.com/openai/parameter-golf/pull/1110) | **Universal Transformer (README wishlist):** Shared recurrent block with depth scheduling. #1088: 1.256. #1110: **1.2249** (near baseline). Iteration embeddings as depth signal. |
| @agalimova | [#1100](https://github.com/openai/parameter-golf/pull/1100) | **LLaDA-MDLM Diffusion:** First discrete diffusion to beat AR baseline (1.2244). **1.1465 BPP**, 512 eval steps, 33M params. Previous best diffusion: 1.625. 1x NVIDIA GB10 (Project DIGITS). |
| @himanshudongre | [#1013](https://github.com/openai/parameter-golf/pull/1013) | **S4D-Lin SSM Hybrid:** First zero-overhead SSM (2 SSM + 9 Transformer layers). 116ms/step matching baseline. Finding: attention > SSM at this scale. **1.1682 BPP.** |
| @CiprianFlorin-Ifrim | [#641](https://github.com/openai/parameter-golf/pull/641)/[#640](https://github.com/openai/parameter-golf/pull/640) | **Binary/Ternary U-Net** — radical compression frontier. Binary (1-bit): **106.2M params in 15.67MB** via bit-packing, 15L 768d, **1.1239 BPB** (non-record, 50k steps). Ternary (1.58-bit): 73.7M params, 10L 768d, **1.1570 BPB** (3-seed, 599s). NeoMuon optimizer, 8192 BPE tokenizer, FP8 QAT, YaRN 2048. 250+ experiments. "Train larger, quantize harder" taken to extreme. |

</details>


---

<details>
<summary><strong>Idea Lineage & Diffusion (63 techniques tracked)</strong></summary>

| Technique | First Appeared | Originator | Adoption |
|-----------|---------------|------------|----------|
| Sliding Window Eval | [#50](https://github.com/openai/parameter-golf/pull/50) | @mattqlf (@mattqlf) | Near-universal (20+) |
| FP16 Tied Embedding | [#42](https://github.com/openai/parameter-golf/pull/42) | @chonchiog (@chonchiog) | ~10+ |
| Int6 Quantization | [#39](https://github.com/openai/parameter-golf/pull/39) | @nanlliu (@nanlliu) | ~15+ |
| MLP 3x Expansion | [#70](https://github.com/openai/parameter-golf/pull/70) | @jfprincz (@jfprincz) | ~12+ |
| Muon Weight Decay | [#60](https://github.com/openai/parameter-golf/pull/60) | @notapplica (@notapplica (from modded-nanogpt)) | Several |
| Overtone Spectral Init | [#60](https://github.com/openai/parameter-golf/pull/60) | @notapplica (@notapplica) | @peytontolbert (#155), @TevBenji (#69) |
| SmearGate / BigramHash | [#65](https://github.com/openai/parameter-golf/pull/65) | @aquariouseworkman (SmearGate + BigramHash — first used in competition by @aquariouseworkman (#65, Mar 19 07:42 UTC). BigramHash separately developed by multiple authors.) | Near-universal (25+). All competitive submissions use SmearGate+BigramHash+OrthoInit. |
| OrthoInit | [#65](https://github.com/openai/parameter-golf/pull/65) | @aquariouseworkman (OrthoInit — first used in competition by @aquariouseworkman (#65, Mar 19 07:42 UTC)) | Near-universal among top SmearGate submissions. Critical co-dependency: SmearGate hurts without OrthoInit (#212 ablation). |
| Test-Time Training | [#77](https://github.com/openai/parameter-golf/pull/77) | @samacqua (@samacqua (LoRA TTT)) | @timowhite88 (#152 SGD, #254 first TTT+SmearGate+11L), @polarizedfortnite-cpu (#81, first TTT+int6), @andrewgcodes (#267 Causal TTT), @charmquark1984 (#281), @ibarrajo (#290, TTT+XSA), @mohosy (#291, pending), @sseanliu (#296, Reptile meta-TTT), @davidpuertolas (#297), @alertcat (#338, TTT on #315 frontier base — neutral), @felipe-parodi (#398, 20-epoch aggressive TTT, 1.1221), @kasimte (#455, SGD TTT on #374 base), @Christopher-Lee-McClendon (#461, high-yield SGD+momentum TTT), **@abaybektursun (#473, legal TTT — 1.1214)**, **@LoquiAuris (#548, batched LoRA TTT — 1.0865)**, **@Sarimsaljook (#573, Multi-Pass TTT — 1.0523 ❌ ruled invalid)** |
| NorMuon | [#0](https://github.com/openai/parameter-golf/pull/0) | @Convergent (Convergent) | @mtybadger, @vmfunc, @dexhunter, others |
| QAT with STE | [#0](https://github.com/openai/parameter-golf/pull/0) | @Convergent (Convergent) | @rsavitt, @yahya010, @trovatochris, others |
| SWA | [#89](https://github.com/openai/parameter-golf/pull/89) | @vmfunc (@vmfunc) | @mtybadger (#122), @dexhunter (#156), @anthony-maio (#376), others |
| Depth Recurrence | [#0](https://github.com/openai/parameter-golf/pull/0) | @Independent (Independent) | @MatthewHRockwell, @koushikkethamakka, @iverbovoy (#148), others |
| Int5 MLP Quantization | [#76](https://github.com/openai/parameter-golf/pull/76) | @unixmadtoonslab (@unixmadtoonslab) | @thwu1 (#180, former SOTA), @alertcat (#219, mixed int5/int6), @Mapika (#349), @Skrisps26 (#354), @signalrush (#369) |
| BigramHash Scaling (4096–16384) | [#180](https://github.com/openai/parameter-golf/pull/180) | @thwu1 (@thwu1 (10240)) | @andrewgcodes (#267, 16384), @simonbissonnette (#466, 12288), @JoeProAI (#462, 8192). Diminishing returns >10240 (#348). |
| Low-Rank Q Factorization | [#215](https://github.com/openai/parameter-golf/pull/215) | @JayCheng113 (@JayCheng113) | Novel — no adopters yet |
| Partial XSA (Exclusive Self-Attention) | [#265](https://github.com/openai/parameter-golf/pull/265) | @unnir (@unnir) | Near-universal at frontier (15+): @jfprincz (#287, #315), @signalrush (#369, #414), @saml212 (#332), @chanwoo-park-official (#400), @fbedev (#417), @sjp611 (#442), @JoeProAI (#462), @kasimte (#455), @ofirkris (#458), @Christopher-Lee-McClendon (#461), others |
| EMA Weight Averaging | [#95](https://github.com/openai/parameter-golf/pull/95) | @MatoTeziTanka (@MatoTeziTanka (PROTEUS EMA)) | Near-universal at frontier (12+): @jfprincz (#287, #315), @signalrush (#369, #414), @sjp611 (#442), @JoeProAI (#462, 0.9985), @ofirkris (#458), @simonbissonnette (#466), @felipe-parodi (#398), @parinzee (#493), others. EMA fails without XSA (#201). |
| Reptile Meta-TTT | [#296](https://github.com/openai/parameter-golf/pull/296) | @sseanliu (@sseanliu) | @JackYoung27 (#302, +causal TTT + decay prior). **#375: failed on #315 base (+0.0076 worse).** |
| BitNet b1.58 | [#126](https://github.com/openai/parameter-golf/pull/126) | @Athenox14 (@Athenox14, @ksang123) | Two independent. #367: standard stack breaks on ternary. |
| Partial RoPE | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz (@jfprincz (25% dims)) | @saml212 (#332), @unnir (#374), @felipe-parodi (#398), @signalrush (#414), @fbedev (#417), @kasimte (#455), @ofirkris (#458), @Christopher-Lee-McClendon (#461), @JoeProAI (#462) |
| LN Scale (1/√layer) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz (@jfprincz) | Near-universal at frontier (10+): @signalrush (#414), @fbedev (#417), @JoeProAI (#462), @sofiabod (#489, calls it "depth damping"), others. Variant: @eb1386 (#449, cosine) |
| Late QAT (last 4% only) | [#315](https://github.com/openai/parameter-golf/pull/315) | @jfprincz (@jfprincz (⚠️ dead code in #315 — torch.compile bug)) | **Working:** @unnir (#374, scale<0.1), @signalrush (#414, threshold 0.15), @fbedev (#417), @JoeProAI (#462). Dropped at 12L (#332). |
| Gradient-Guided Quant | [#332](https://github.com/openai/parameter-golf/pull/332) | @saml212 (@saml212) | @ndokutovich (#486, sensitivity-ranked int7/6/5 — top 10%/70%/20%) |
| TrigramHash | [#327](https://github.com/openai/parameter-golf/pull/327) | @Ananddna (@Ananddna) | @ndokutovich (#486, 4096 buckets + VRL + GradQuant + Cosine TTT, **1.0887**) |
| Per-Head Temperature | [#327](https://github.com/openai/parameter-golf/pull/327) | @Ananddna (@Ananddna) | Novel — each head learns its own temperature scalar |
| Tight SWA (scale<0.2) | [#374](https://github.com/openai/parameter-golf/pull/374) | @unnir (@unnir) | @dannywillowliu-uchi (#379, +GPTQ-lite), @kasimte (#455, +TTT) |
| Shared Value Embedding | [#374](https://github.com/openai/parameter-golf/pull/374) | @unnir (@unnir) | @dannywillowliu-uchi (#379, +GPTQ-lite), @kasimte (#455, +TTT), @Christopher-Lee-McClendon (#461, layers 9-10), **@JoeProAI (#505, GEPA arch, 1.1181)** |
| AdamW TTT | [#442](https://github.com/openai/parameter-golf/pull/442) | @sjp611 (@sjp611 (3-line diff from #398: SGD→AdamW)) | @JoeProAI (#462), @mrdavtan (#481, cosine), @ndokutovich (#486), @sofiabod (#489, 7L), @amaljithkuttamath (#490, +VRL+GA), @ahmettrkck (#491, +DWA), **@EthanYangTW (#503, legal AdamW TTT)**, @ymrohit (#555, closed) |
| GPTQ-lite → Full GPTQ | [#379](https://github.com/openai/parameter-golf/pull/379) | @dannywillowliu (@dannywillowliu-uchi (per-layer clip percentile search)) | @signalrush (#414), @fbedev (#417), @gowtham0992 (#478), @EthanYangTW (#503, **#606 int5 GPTQ**), **@raahilshah (#535)**, **@gowtham0992 (#569)**, @cmcdnd (#576), **@newjordan (#587)**, **@saml212 (#609)**, **@danialht (#615)**. Now standard at frontier. |
| Value Residual Learning | [#413](https://github.com/openai/parameter-golf/pull/413) | @anantdgoel (@anantdgoel (arXiv:2410.17897, −0.015 dev)) | @ndokutovich (#486, **1.0887**+Cosine TTT), **@amaljithkuttamath (#490, VRL+GA+TTT, 1.0891 1-seed!)**, **@gowtham0992 (#569, VRL no-TTT → 1.1175, best non-TTT at time)**, @joshuaswarren (#474, failed on weak base), @carlesonielfa (#457), @yuvrajyadav17 (#471, pending), @ahmettrkck (#491, VRL+DWA+TTT) |
| Catalytic Residuals | [#450](https://github.com/openai/parameter-golf/pull/450) | @zachgoldfine44 (@zachgoldfine44 (`x + c*f(x)`, −0.024 BPB)) | @joshuaswarren (#474, +VRL+GA, 12L — 1.1690, techniques don't stack on weak base) |
| Two-Phase TTT | [#417](https://github.com/openai/parameter-golf/pull/417) | @fbedev (@fbedev (50ep norm-only + 10ep last-3-blocks)) | Novel — no adopters yet |
| Gated Attention | [#413](https://github.com/openai/parameter-golf/pull/413) | @anantdgoel (@anantdgoel (arXiv:2505.06708, −0.003 dev)) | **@amaljithkuttamath (#490, +VRL+TTT, 1.0891)**, @joshuaswarren (#474, failed on weak base), @yuvrajyadav17 (#471, pending) |
| Cosine TTT + Per-Layer LR | [#481](https://github.com/openai/parameter-golf/pull/481) | @mrdavtan (@mrdavtan (cosine LR decay + 3× MLP output proj LR)) | **@sofiabod (#518, cosine+per-layer → 1.0814)**, **@ndokutovich (#486, cosine → 1.0887)**, @Christopher-Lee-McClendon (#537, per-layer LR on legal TTT). ⚠️ Pre-eval TTT (except #537) |
| XSA-All (11 layers) | [#478](https://github.com/openai/parameter-golf/pull/478) | @gowtham0992 (@gowtham0992 (first to test XSA on all layers)) | @EthanYangTW (#503, #606), @cmcdnd (#576), **@newjordan (#587)**, **@saml212 (#609, best non-TTT)**, **@danialht (#615)**. Now standard at frontier. |
| LeakyReLU(0.5)² | [#434](https://github.com/openai/parameter-golf/pull/434) | @parinzee (@parinzee (squared leaky ReLU, 0.5 neg slope)) | **@sofiabod (#518)**, **@raahilshah (#535)**, @Christopher-Lee-McClendon (#537), @abaybektursun (#549), **@gowtham0992 (#569)**, @cmcdnd (#576), @RoyiRa (#589), **@saml212 (#609)**, **@robinojw (#620)**. **10+ adopters — fastest-spreading technique.** |
| Delayed PPM Eval Bank | [#511](https://github.com/openai/parameter-golf/pull/511) | @AnirudhRahul (@AnirudhRahul (classical n-gram backoff with 2048-token delay, on @thwu1's #180 base)) | Novel — −0.00126 BPB at p=0.000041. Zero artifact cost. |
| Post-TTT Temperature Calibration | [#576](https://github.com/openai/parameter-golf/pull/576) | @cmcdnd (@cmcdnd (T=0.98 re-score after legal TTT to correct overconfidence, −0.003 BPB)) | Novel — no adopters yet. Zero-cost technique. |
| Walsh-Hadamard Rotation | [#586](https://github.com/openai/parameter-golf/pull/586) | @EaCognitive (@EaCognitive (pre-quant rotation for outlier redistribution. zstd 1.70x→1.76x, freeing 530KB for VE128)) | Novel — **substitutes with GPTQ at int6** (they address the same outlier problem). Also found Late QAT dead-code bug in CastedLinear. |
| Late Soft-Round QAT | [#589](https://github.com/openai/parameter-golf/pull/589) | @RoyiRa (@RoyiRa (temperature-controlled soft-round surrogate replaces hard STE; bin-aware gradients near int6 boundaries)) | **@EthanYangTW (#606, tanh α1→16, best legal TTT 1.1162)**. Independent discovery likely (~8hr gap, same tanh-alpha approach). |
| Selective Pruning | [#609](https://github.com/openai/parameter-golf/pull/609) | @saml212 (@saml212 (post-GPTQ ±1 magnitude pruning sorted by reconstruction error)) | Novel — no adopters yet. |
| Residual Input Mixing | [#615](https://github.com/openai/parameter-golf/pull/615) | @danialht (@danialht (dense residual: each block sees learned mix of current stream + earlier blocks + x0)) | Novel — no adopters yet. |
| AWQ | [#623](https://github.com/openai/parameter-golf/pull/623) | @SPThole (@SPThole (activation-aware weight scaling α=0.5 before quant, closed 63% quant gap)) | Novel — first use in competition. |
| Cyclic Muon Momentum | [#623](https://github.com/openai/parameter-golf/pull/623) | @SPThole (@SPThole (triangle wave 0.85-0.95, period=50)) | Novel — no adopters yet. |
| N-gram Eval Cache | [#659](https://github.com/openai/parameter-golf/pull/659) | @deanbrr (@deanbrr (concept), @newjordan (5-gram implementation)) | **@lukacf (#702, multi-order backoff)**, **@Asukabot0 (#715, #727, entropy-adaptive)**, **@hypery11 (#724, 7-gram)**, **@gowtham0992 (#738, +kNN-LM)**, **@resouer (#740, 9L+5gram)**, **@andrewbaggio1 (#741, +cosine TTT)**. **8 adopters in <12 hours — fastest spread in competition history.** |
| Multi-Order N-gram Backoff | [#702](https://github.com/openai/parameter-golf/pull/702) | @lukacf (@lukacf (cascade 7→6→5→4→3→2 on miss)) | **@Asukabot0 (#727, orders 2-7 + entropy-adaptive)** |
| Entropy-Adaptive Alpha | [#702](https://github.com/openai/parameter-golf/pull/702) | @lukacf (@lukacf (`alpha = 0.05 + 0.35 * sigmoid(2*(H-4))`)) | **@Asukabot0 (#727, wider range 0.05-0.60)** |
| Hidden-State kNN-LM | [#738](https://github.com/openai/parameter-golf/pull/738) | @gowtham0992 (@gowtham0992 (GPU ring buffer + RBF kernel, Khandelwal et al. 2019)) | Novel — first in competition. |
| Depth Recurrence (with block scalars) | [#686](https://github.com/openai/parameter-golf/pull/686) | @msisovic (@msisovic (layers 4+5 repeated, 11→13 virtual, ~2K params)) | Novel — recovers 70% of independent 12L gain. |
| Hedge Mixer (expert ensemble) | [#688](https://github.com/openai/parameter-golf/pull/688) | @RoyiRa (@RoyiRa (5-expert: neural + unigram + bigram + trigram + entropy, Hedge algorithm eta=0.1. First in #688; improved in #700 with CROWN-Q + MLP3.5x + stride=64)) | **@pentxayc (#731, +VRL+TTT+Polyak EMA)**, **@agalimova (#720, XSA6+BigramHash4K on #700 base)** |
| MiLe Loss | [#703](https://github.com/openai/parameter-golf/pull/703) | @Gusanidas (@Gusanidas (entropy-weighted token loss, γ=1.1 decaying to 0 during warmdown)) | Novel — no adopters yet. |
| DeltaNet / GatedDeltaNet | [#651](https://github.com/openai/parameter-golf/pull/651) | @phulin (Linear attention variant (Gated Delta Rule) from fla library. Enables longer effective context. @newjordan combined with Frugendorff recursive loops for sub-0.9 BPB.) | @shalyhinpavel (#875, 1.0226), @brian386 (#939), @dnldsz (#970), @newjordan (#990, #1028, #1047 — DeltaNet Crawler 0.8822) |
| AR Self-Gen GPTQ Calibration | [#728](https://github.com/openai/parameter-golf/pull/728) | @abaybektursun (Model autoregressively generates calibration tokens for Full Hessian GPTQ. Closes 84% of val-vs-random gap. First appeared in #728, resubmitted as #1019.) |  |
| Multi-Token Prediction (MTP) Auxiliary Loss | [#88](https://github.com/openai/parameter-golf/pull/88) | @seanward (MTP auxiliary head (training-only, discarded at export). Introduced day 1 by @seanward (#88). Disabled in baseline (MTP_NUM_HEADS=0). Re-enabled by @michaelwinczuk (#1031, weight=0.1) with -0.0037 BPP claimed.) |  |
| TARA (Test-Time Activation ReAlignment) | [#1055](https://github.com/openai/parameter-golf/pull/1055) | @sanyalsunny111 (Training-free eval-time method: contrastive adjustment using early-layer hidden states. No gradients, no weight updates. Community flagged causality concern (scatter_ leaks target token).) |  |
| Coprime-Stride Multi-Shard Loader | [#726](https://github.com/openai/parameter-golf/pull/726) | @DeepReinforce (Coprime-stride block sampling across shards for batch diversity. Zero step-time overhead. Full permutation cycle before repetition.) | @dexhunter (#1060, 1.1123 — SOTA-beating) |
| SLOT (Selective Logit Offset Tuning) | [#1084](https://github.com/openai/parameter-golf/pull/1084) | @AnubhavBharadwaaj (Optimize 512-dim delta at last hidden layer during eval (AdamW, 5-8 steps per batch)) | @sahiee-dev (#1150), @dexhunter (#1172), @bigbag (#1176) |
| QK-Gain (Learnable Query Scalar) | [#259](https://github.com/openai/parameter-golf/pull/259) | @outsourc-e (Per-head learnable scalar after QK-norm (concept #259, optimal value 4.0 found by @jainpranjal97 in #1125's 45-experiment sweep)) | @bigbag (#1176) |
| Scylla Tokenizer (998-vocab TokenMonster) | [#1143](https://github.com/openai/parameter-golf/pull/1143) | @simon-marcus (Novel 998-token TokenMonster-derived tokenizer — biggest single-technique BPB breakthrough) | @icryo (#1184, first sub-1.0 BPB at 0.9485) |
| Parallel Residuals (Dual-Lane Routing) | [#1204](https://github.com/openai/parameter-golf/pull/1204) | @msisovic (Separate attention and MLP residual lanes from layer 7 with learned cross-lane routing. Ported from modded-nanogpt PR #230.) |  |
| Window Attention (Training-Time) | [#1212](https://github.com/openai/parameter-golf/pull/1212) | @Gusanidas (Sliding window of 512 on even layers during training, full attention on odd — 21% faster at seq_len 6144. Distinct from sliding window eval.) |  |
| Mixed Seq_Len Training | [#1212](https://github.com/openai/parameter-golf/pull/1212) | @Gusanidas (Different GPUs train at different sequence lengths (5x2048 + 3x6144) for throughput + long-context balance.) |  |

</details>


---

## Predictions & Commentary

1. **Tokenizer optimization is the new frontier.** #1060 broke the 1.1194 ceiling, #1089 pushed to 1.1086, then **#1143 (@simon-marcus, 1.0806, Scylla tokenizer) leapt further** using Turbo-Muon + EngramLite + mixed int6/int7 GPTQ — no TTT, no n-gram cache, 87s eval. #1099 (@Bortlesboat, 1.1136, 3-seed) confirms the Coprime-Stride + Full GPTQ approach is reproducible. This is the cleanest possible record: pure neural with better data loading and quantization. Pending organizer review for official leaderboard merge.

2. **N-gram caches are powerful but face compliance scrutiny.** 11 of 20 record-eligible submissions use n-gram evaluation caches. #1060's SOTA without any n-gram shows well-tuned pure neural can compete. Both frontiers are advancing in parallel: #1094 (@michaelwinczuk, 0.4027, 3-seed) pushed n-gram further with causal sequential scoring, while #1089 pushed pure neural to 1.1086.

3. **TTT is redundant on the Full GPTQ + XSA-all stack.** Both #1060 (no TTT, 1.1123) and #1019 (no TTT, 1.1147) outperform #549 (with TTT, 1.1194). Full Hessian GPTQ + complete XSA coverage captures what TTT was providing, while dropping TTT's eval-time overhead allows more training steps.

4. **New architectures remain exploratory.** DeltaNet Crawler (#1047, 0.8822 but high variance), H-Net (#1044), Causal Oscillator LM (#1061, 1.34) — architecturally novel but not competitive. **Diffusion is emerging:** #1100 (@agalimova, LLaDA-MDLM, 1.1465) is the first discrete diffusion to beat the AR baseline, improving on previous best diffusion (#820, 1.625) by 0.47 BPB. TARA (#1055) was closed for causality violation.

5. **The frontier is now data pipeline + quantization quality.** #1060's key innovation is a better data loader (coprime-stride for batch diversity) and better quantization (Full Hessian GPTQ). Both are systems-level improvements — no architectural novelty. Expect the next wave to optimize training data throughput and quantization precision.
**6. Scylla + SLOT is the untested mega-combination.** #1184 (Scylla, 0.9485) has no eval-time adaptation. Adding SLOT (−0.021 in #1176) could push to ~0.925-0.935. If someone also adds n-gram backoff, sub-0.90 is plausible.

**7. The next official leaderboard merge will be disruptive.** Multiple pending submissions beat the 1.1147 SOTA by large margins. When the next batch is merged, the 0.005-nat threshold will shift dramatically, potentially invalidating many currently-record-eligible submissions.

**8. CUTLASS EVT / fused kernels will become a systems moat.** #1105's fused MLP enables MLP 3.5x within the time budget — 20% faster steps = more training = lower BPP. Hard to replicate without GPU programming expertise.


---

<details>
<summary><strong>Full Official Leaderboard (19 entries)</strong></summary>

Validated against the SOTA at submission time.

| Rank | Score | Author | Key Techniques | PR |
|------|-------|--------|---------------|-----|
| 1 | **1.1147** | @abaybektursun | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 on #549 stack | [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| 2 | **1.1194** | @sanjeevmadhav | LeakyReLU² + Legal Score-First TTT + Parallel Muon on #414 stack | [#549](https://github.com/openai/parameter-golf/pull/549) |
| 3 | **1.1228** | @signalrush | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 | [#414](https://github.com/openai/parameter-golf/pull/414) |
| 4 | **1.1248** | @jfprincz | 11L Partial RoPE + LN Scale + EMA + XSA4 | [#315](https://github.com/openai/parameter-golf/pull/315) |
| 5 | **1.1271** | @jfprincz | 11L XSA4 + EMA + Int6 MLP3x | [#287](https://github.com/openai/parameter-golf/pull/287) |
| 6 | **1.1307** | @unnir | 11L Efficient Partial XSA | [#265](https://github.com/openai/parameter-golf/pull/265) |
| 7 | **1.1458** | @raahilshah | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + MuonWD + SWA | [#162](https://github.com/openai/parameter-golf/pull/162) |
| 8 | **1.1502** | @aruniyer | 11L + Int6 QAT + MLP3x + WD 0.04 + zstd-22 | [#86](https://github.com/openai/parameter-golf/pull/86) |
| 9 | **1.1556** | @aquariouseworkman | SmearGate + OrthoInit + Int6 STE QAT + MLP3x + Sliding Window | [#65](https://github.com/openai/parameter-golf/pull/65) |
| 10 | **1.1586** | @yahya010 | 10L Int6 QAT + Zstd MLP2.6x + Muon 0.99 + Sliding Window | [#63](https://github.com/openai/parameter-golf/pull/63) |
| 11 | **1.163** | @aquariouseworkman | Mixed int6/int8 + MLP3x + Sliding Window | [#65](https://github.com/openai/parameter-golf/pull/65) |
| 12 | **1.1748** | @notapplica | Sliding Window + FP16 Embed + 10L + Muon WD + Spectral Init | [#60](https://github.com/openai/parameter-golf/pull/60) |
| 13 | **1.1925** | @mattqlf | Sliding Window Eval (stride=64) | [#50](https://github.com/openai/parameter-golf/pull/50) |
| 14 | **1.1928** | @samacqua | LoRA Test-Time Training | [#77](https://github.com/openai/parameter-golf/pull/77) |
| 15 | **1.2014** | @spokane-way | 4k seq length + tuned hyperparams | [#52](https://github.com/openai/parameter-golf/pull/52) |
| 16 | **1.206** | @spokane-way | 2048 seq length | [#49](https://github.com/openai/parameter-golf/pull/49) |
| 17 | **1.2147** | @nanlliu | 10 layers, mixed int8/int6 | [#39](https://github.com/openai/parameter-golf/pull/39) |
| 18 | **1.2197** | @chonchiog | FP16 Tied Embedding + LR/Warmdown Tuning | [#42](https://github.com/openai/parameter-golf/pull/42) |
| 19 | **1.2244** | @Baseline | 9L 512dim 1024vocab TiedEmbed 4 KV heads | [#0](https://github.com/openai/parameter-golf/pull/0) |

</details>

<details>
<summary><strong>All Record-Eligible Submissions (26 entries)</strong></summary>

Sorted by BPB ascending.

| BPB | Author | Δ nats | Seeds | Techniques | PR |
|-----|-----|-----|-----|-----|-----|
| **0.4027** | @michaelwinczuk | 1.202 | 3 | **Swarm-Designed Causal BackoffNgramMixer.** Orders 2-10, 4M hash buckets, entropy-adaptive alpha, causal sequential chunk scoring (score-first, update-after). Full-vocab mixture distribution. Neural baseline 1.1245. MTP heads=2, LeakyReLU(0.75)², Parallel Muon. Beats #803 (0.4416) by 0.039. No TTT. Std=0.0015. | [#1094](https://github.com/openai/parameter-golf/pull/1094) |
| **0.4416** | @pentxayc | 1.137 | 3 | **Complementary Training** — tokens predictable by bigram stats get lower loss weight during training. Model specializes on what n-grams can't predict, enabling higher eval-time n-gram alpha (20-75%). + Backoff N-gram Mixer + VRL + XSA-4. Std=0.0001. | [#803](https://github.com/openai/parameter-golf/pull/803) |
| **0.4961** | @newjordan | 1.045 | 3 | **Bandit: ClownCar Crawler + Cubric Ngram9.** ClownCar crawler (4 flat + 1 crawler x4 loops, Frugendorff) + X-WING n-gram oracle (shared tables, 3D Cubric 54-cell warm-start, entropy-adaptive alpha 0.20-0.75, order-9). GPTQ-int6+zstd ~9.3MB. Pure neural baseline (SW BPB): 1.1867. Std=0.0003. | [#1083](https://github.com/openai/parameter-golf/pull/1083) |
| **0.5466** | @travispchen | 0.959 | 3 | **Order-Adaptive Entropy Gating + BackoffNgramMixer + Drift-Free TTT.** Builds on #779 with per-order entropy thresholds from #774. Sub-0.55 BPB. Std=0.0010. | [#798](https://github.com/openai/parameter-golf/pull/798) |
| **0.5644** | @newjordan | 0.929 | 3 | **X-WING: Shared N-gram Tables** — all 8 GPU ranks update tables with same tokens (full 62M-token view). Cubric per-order adaptive alpha. Std=0.0006. | [#800](https://github.com/openai/parameter-golf/pull/800) |
| **0.8881** | @hypery11 | 0.383 | 3 | 11L + **order-adaptive 11-gram backoff** (orders 2-11) + XSA-all + GPTQ-lite. No TTT. 13.99MB. Std=0.0006. Upgrade from #788 (9-gram, 0.9059). | [#795](https://github.com/openai/parameter-golf/pull/795) |
| **0.9059** | @hypery11 | 0.353 | 3 | 11L + **order-adaptive 9-gram backoff** (orders 2-9) + XSA-all + VRL + GA + GPTQ-lite. No TTT. 13.99MB artifact. Std=0.0009. | [#788](https://github.com/openai/parameter-golf/pull/788) |
| **0.9362** | @newjordan | 0.301 | 3 | **Podracing III (Cubric Lite):** 7-gram backoff (2-7) + entropy-adaptive alpha + **per-order adaptive alpha scaling**. 0.026 improvement over Podracing II. Std=0.0004. | [#782](https://github.com/openai/parameter-golf/pull/782) |
| **0.9370** | @travispchen | 0.300 | 3 | **Order-Adaptive Entropy Gating** — different entropy thresholds per n-gram order. Built on #753 + XSA-all. Std=0.0003. | [#774](https://github.com/openai/parameter-golf/pull/774) |
| **0.9485** | @icryo | 0.281 | 3 | **Scylla tokenizer + Full Hessian GPTQ + XSA-all + FA3 + Coprime-Stride Loader.** Combines Scylla tokenizer (998 tokens, from #1143) with #1060 training stack. First sub-1.0 BPP validated submission. No TTT needed. 88ms/step, ~6700 steps. 15.6MB. Beats #1143 (1.0806) by 0.132. Std=0.0008. | [#1184](https://github.com/openai/parameter-golf/pull/1184) |
| **0.9605** | @raahilshah | 0.260 | 3 | 11L Full GPTQ (within 596s budget) + multi-order n-gram backoff + entropy-adaptive alpha. Fixed-alpha variant: 0.9757. Std=0.0003. | [#778](https://github.com/openai/parameter-golf/pull/778) |
| **0.9625** | @newjordan | 0.257 | 3 | **Podracing II:** Multi-order backoff (2-7) + entropy-adaptive alpha on #414 base. GPTQ in training budget. XSA4 + LeakyReLU². No TTT. Std=0.0005. | [#753](https://github.com/openai/parameter-golf/pull/753) |
| **0.9641** | @skoustav35 | 0.254 | 3 | N-gram Backoff Cache (orders 2-9, Laplace smoothing) + Score-First TTT + MTP-2 + XSA-5 | [#1185](https://github.com/openai/parameter-golf/pull/1185) |
| **1.0240** | @lukacf | 0.153 | 3 | **Multi-order n-gram backoff** + entropy-adaptive alpha + XSA-all + VRL + Full GPTQ + 7% prune. No TTT. Std=0.0003. Autonomous research via Goldfish. | [#702](https://github.com/openai/parameter-golf/pull/702) |
| **1.0321** | @dcrow85 | 0.139 | 3 | **Gravity Tokenizer** -- replaces 659/765 BPE merge tokens by ablation-leverage scoring. Vanilla 12L 384d, no standard stack. Std=0.0011. ⚠️ **Tokenizer change -- needs extra scrutiny per README rules** | [#755](https://github.com/openai/parameter-golf/pull/755) |
| **1.0337** | @Asukabot0 | 0.137 | 3 | XSA-all + LeakyReLU² + VRL + GA + **7-gram eval cache** (fixed alpha=0.40). No TTT. Std=0.0010. | [#715](https://github.com/openai/parameter-golf/pull/715) |
| **1.0461** | @newjordan | 0.116 | 3 | **5-gram eval cache** (fixed alpha=0.20, count-min sketch) + LeakyReLU² + XSA4 + GPTQ. No TTT. Std=0.0010. ⚠️ **Eval-time GPTQ flagged by @valerio-oai** | [#706](https://github.com/openai/parameter-golf/pull/706) |
| **1.0541** | @RoyiRa | 0.102 | 3 | **5-expert Hedge Mixer** + CROWN-Q + stride=64. Pre-TTT 1.1254, TTT gain -0.071. High seed variance (std=0.012). | [#700](https://github.com/openai/parameter-golf/pull/700) |
| **1.0806** | @simon-marcus | 0.058 | 3 | **Scylla (novel TokenMonster-derived tokenizer, 998 vocab) + Legal Score-First TTT.** Tokenizer selected via 3-phase autoresearch (SP search → TM sidecar → TM-only optimization). Pruned from english-1024-clean-v1 to 998 tokens. BPB via explicit per-token metadata LUTs, not SentencePiece runtime. Base model already beats SOTA at roundtrip (1.1051). TTT gain: -0.003. 27M params. Dual PR also contains WaterLOO n-gram (0.0990). Tokenizer change faces extra README scrutiny. Std~0.0005. | [#1143](https://github.com/openai/parameter-golf/pull/1143) |
| **1.0909** | @resouer | 0.040 | 3 | **9L** + 5-gram eval cache (fixed alpha=0.20) + XSA-all + LeakyReLU² + int8 quant (14.7MB). No TTT. Std=0.0011. | [#740](https://github.com/openai/parameter-golf/pull/740) |
| **1.0914** | @bigbag | 0.039 | 3 | **QK-Gain 4.0 + XSA-11 + Muon-TTT + SLOT (8 AdamW steps, lr=0.005).** QK_GAIN_INIT=4.0 (from #1125's 45-experiment sweep). XSA all 11 layers. Muon-TTT (score-first, 3 epochs) + SLOT (per-batch delta, 8 steps). Ablation: sliding 1.1155 → +TTT 1.1122 → +SLOT **1.0914**. Built on #1135. 87.2ms/step. Std=0.0003. | [#1176](https://github.com/openai/parameter-golf/pull/1176) |
| **1.1015** | @dexhunter | 0.022 | 3 | **SLOT (lr=0.005, steps=8) + Split-LR + Full GPTQ + XSA-all + Sigmoid-Gated Skips + Soft-Round QAT.** SLOT replaces TTT entirely — most aggressive SLOT hyperparams yet. Split-LR: early layers 0.025, late layers 0.030. BigramHash 2816x160. Brotli+byte-shuffle. Post-EMA 1.1303 → sliding+SLOT 1.1015 (-0.029). Key finding: TTT is neutral on Full GPTQ stack; SLOT alone outperforms TTT+SLOT. 177s eval (vs 569s for TTT+SLOT). Std=0.0011. | [#1172](https://github.com/openai/parameter-golf/pull/1172) |
| **1.1064** | @andrewbaggio1 | 0.014 | 3 | Full Hessian GPTQ + Score-First TTT + SLOT (per-batch delta, 8 AdamW steps). No n-gram cache. | [#1209](https://github.com/openai/parameter-golf/pull/1209) |
| **1.1086** | @mikeapedia | 0.010 | 3 | **Turbo-Muon + EngramLite + GPTQ mixed int6/int7.** Turbo-Muon: AOL preconditioning + Polar Express coefficients + row_col normalization (4 NS iters). EngramLite: multi-head prime-based hash embeddings (bigram+trigram, 2 heads, 8192 buckets). GPTQ mixed-precision with Hessian sensitivity-based bit allocation. Brotli+byte-shuffle compression. MLP 3.5x LeakyReLU(0.3)². Built on #609. No TTT. Std=0.0006. | [#1089](https://github.com/openai/parameter-golf/pull/1089) |
| **1.1099** | @newjordan | 0.008 | 3 | **Rascal: 11L XSA-all + Parallel Muon + Coprime loader + Bigram2048 + RoPE16 + SWA + Late QAT.** No GPTQ — naive int6 embed + 5 layers, zstd. ~15.5MB, 27M params. No TTT. Std~0.0002. | [#1120](https://github.com/openai/parameter-golf/pull/1120) |
| **1.1116** | @barneywohl | 0.005 | 3 | **Fused Triton MLP + Full GPTQ + Coprime Loader + XSA-all + BH2816.** Combines fused forward kernel (#1072-style) with Full Hessian GPTQ + coprime-stride loader. No TTT. Std=0.0005. | [#1135](https://github.com/openai/parameter-golf/pull/1135) |

</details>

<details>
<summary><strong>All Not Yet Validated Submissions (88 entries)</strong></summary>

Competitive submissions that haven't demonstrated statistical significance.

| BPB | Author | Seeds | Techniques | PR |
|-----|-----|-----|-----|-----|
| **0.0180** | @sofiabod | 3 | Packed Causal N-gram + Dirichlet Backoff (0.0180). Post-sweep. Normalization status unclear. | [#1056](https://github.com/openai/parameter-golf/pull/1056) |
| **0.0905** | @vimeto | 1 | **Seed-Regenerated Random Model + Incremental N-gram Cache.** Model weights generated from seed (not trained) — neural baseline 1.503 BPP. All compression from n-gram cache. 1-seed only, run on MI250X (not H100). Pending H100 validation + 2 more seeds. | [#1095](https://github.com/openai/parameter-golf/pull/1095) |
| **0.1130** | @sofiabod | 3 | **Single-Pass Packed N-gram + Dirichlet CTW** (0.1130). Post-sweep submission. Normalization status unclear — Dirichlet CTW may handle it correctly. | [#1030](https://github.com/openai/parameter-golf/pull/1030) |
| **0.4311** | @Naazimsnh02 | 0 | Complementary Training + Backoff N-gram Mixer + TTT (0.4311). Post-sweep. | [#1033](https://github.com/openai/parameter-golf/pull/1033) |
| **0.6364** | @Naazimsnh02 | 0 | Depth Recurrence + Multi-Order N-gram Backoff (0.6364). | [#808](https://github.com/openai/parameter-golf/pull/808) |
| **0.6671** | @hypery11 | 3 | BackoffNgramMixer (0.6671). | [#813](https://github.com/openai/parameter-golf/pull/813) |
| **0.6672** | @minh-stakc | 0 | 11L + Multi-Order N-gram Backoff + Entropy-Adaptive Alpha. Sub-0.7 BPB. | [#770](https://github.com/openai/parameter-golf/pull/770) |
| **0.8822** | @newjordan | 3 | **Medusa S2:** DeltaNet Crawler (4 flat layers + 1 crawler x4 loops, Frugendorff). Loop-aware GPTQ (int6+zstd). EMA. 9.8MB artifact. Std=0.105 — high variance, fails p<0.01. | [#1047](https://github.com/openai/parameter-golf/pull/1047) |
| **0.8960** | @armantsaturian | 0 | 7-gram n-gram cache (0.8960). | [#797](https://github.com/openai/parameter-golf/pull/797) |
| **0.9258** | @agalimova | 2 | Kitchen Sink: 7-gram + XSA6 + BigramHash4K + Cosine TTT on #741 base. 2 seeds (3rd running). | [#776](https://github.com/openai/parameter-golf/pull/776) |
| **0.9984** | @newjordan | 3 | **Medusa (original):** DeltaNet Crawler + Frugendorff. Flagged: GPTQ calibration read training data after 600s wallclock cap. Superseded by #1047. | [#1028](https://github.com/openai/parameter-golf/pull/1028) |
| **1.0226** | @shalyhinpavel | 0 | **Pure Neural GDN** — 1.0226 BPB without n-gram cache. | [#875](https://github.com/openai/parameter-golf/pull/875) |
| **1.0400** | @pentxayc | 1 | Hedge Mixer + VRL + AdamW TTT + Polyak EMA. Freeze 9/11 blocks. ⚠️ Hedge Mixer + n-gram. | [#731](https://github.com/openai/parameter-golf/pull/731) |
| **1.0577** | @estesryan | 1 | **SR-CM-P2Loss: P2 difficulty-aware loss + residual mixing + conv token mixer.** P2 loss ((1-p)^2) weights hard tokens more. Wallclock-aware LR warmdown. Residual mixing + conv token mixer (novel architecture). Int6 + late QAT. 15.06MB. 1 seed only — needs multi-seed validation. | [#1180](https://github.com/openai/parameter-golf/pull/1180) |
| **1.0891** | @amaljithkuttamath | 1 | 11L + Value Residual + Gated Attention + AdamW TTT on #442 base. Pre-quant 1.1545. ⚠️ TTT | [#490](https://github.com/openai/parameter-golf/pull/490) |
| **1.0920** | @Christopher-Lee-McClendon | 1 | GEPA 30k steps + int6 GPTQ-lite + legal SGD TTT. 4xA100 non-record. | [#668](https://github.com/openai/parameter-golf/pull/668) |
| **1.0944** | @Christopher-Lee-McClendon | 1 | GEPA 25k steps (13k warmdown) + int6 GPTQ-lite + legal SGD TTT. 4xA100 non-record. Float base 1.1088. | [#644](https://github.com/openai/parameter-golf/pull/644) |
| **1.0945** | @danielxmed | 0 | N-gram Cache + Entropy-Adaptive Alpha (1.0945). Post-sweep — normalization status unclear. | [#1026](https://github.com/openai/parameter-golf/pull/1026) |
| **1.0983** | @Christopher-Lee-McClendon | 1 | GEPA 20k steps (8k warmdown) + int6 GPTQ-lite + legal SGD TTT (10ep). 4xA100 non-record. Float base 1.1153. | [#628](https://github.com/openai/parameter-golf/pull/628) |
| **1.1063** | @msisovic | 3 | Parallel Residuals (from layer 7, learned routing) + Mini Depth Recurrence (layers 4-5, delayed start step 3000) + Mixed int5/int6 GPTQ + Brotli | [#1204](https://github.com/openai/parameter-golf/pull/1204) |
| **1.1085** | @NewyorkDev | 1 | **JEPA + AdamW Pre-Quant TTT + Full Hessian GPTQ + FA3.** 11L 512d U-Net (5 enc + 6 dec), JEPA auxiliary loss (multi-horizon 1/2/4/8), AdamW TTT before quantization (3 epochs), int6 Full Hessian GPTQ (128-batch), XSA-all, BigramHash(2048), LZMA. Negative: SGD TTT fails on CastedLinear. 1 seed only. | [#1006](https://github.com/openai/parameter-golf/pull/1006) |
| **1.1093** | @aruniyer | 3 | 15L Depth Recurrence + LeakyReLU² + Cosine TTT. Pure neural, below official SOTA. | [#857](https://github.com/openai/parameter-golf/pull/857) |
| **1.1108** | @Gusanidas | 5 | Window Attention (512 on even layers) + Mixed Seq_Len Training (5x2048 + 3x6144) + 12L + Fused Triton MLP + Brotli. No TTT. | [#1212](https://github.com/openai/parameter-golf/pull/1212) |
| **1.1109** | @AnirudhRahul | 4 | **Loader FullGPTQ XSA11 + online n-gram augment.** Builds on #1060, adds warmdown=4000 + single-pass online token agreement evaluator. 4 seeds. | [#1145](https://github.com/openai/parameter-golf/pull/1145) |
| **1.1122** | @dexhunter | 3 | **Coprime-Stride Loader + Full Hessian GPTQ + XSA-all.** Pure neural, no TTT — sliding window only (87s eval). Coprime-stride multi-shard loader (#726-style) for batch diversity. Full Hessian GPTQ with Cholesky error compensation + column reordering (14s within training budget). XSA all 11 layers. BigramHash(2816x112). Built on #549 stack. Beats former SOTA (1.1194) by 0.012 nats. Std=0.0004. | [#1060](https://github.com/openai/parameter-golf/pull/1060) |
| **1.1125** | @abaybektursun | 3 | Fused MLP (CUTLASS EVT + Triton TMA) + MLP 3.5x + Mixed int5/int6 (Hessian) + Brotli-11 + LR floor | [#1105](https://github.com/openai/parameter-golf/pull/1105) |
| **1.1130** | @malc3om | 3 | **Standard 11L SOTA stack reproduction.** LeakyReLU², XSA4, Partial RoPE, LN Scale, VE128, EMA, Late QAT, Legal TTT, GPTQ-lite. Claims 1.1130 but only overall BPP reported, no per-seed breakdown visible. | [#1077](https://github.com/openai/parameter-golf/pull/1077) |
| **1.1133** | @Bortlesboat | 3 | **Coprime-Stride Loader + Full GPTQ + XSA-all** (builds on #1060). GPTQ reserve optimized (14s→10s, +44 extra training steps). FA3/FA2 graceful fallback. No TTT, 85s eval. Std=0.0001. | [#1099](https://github.com/openai/parameter-golf/pull/1099) |
| **1.1140** | @Gusanidas | 12 | **ResidLambdas + Split-LR + Train-Budget GPTQ + Coprime Loader.** Residual lambdas: learnable per-sublayer residual scaling (init sqrt(1.1), 5x scalar LR). Split-LR for different param groups. 12-seed validation (std=0.0005). No TTT. Built on #549. Δ=0.0091 nats vs SOTA, p<0.0001. | [#1130](https://github.com/openai/parameter-golf/pull/1130) |
| **1.1142** | @abaybektursun | 3 | **Val-Calibrated GPTQ** + XSA-all + BigramHash 3072x112. No TTT. Std=0.0001. | [#728](https://github.com/openai/parameter-golf/pull/728) |
| **1.1146** | @unknown | 1 | **EngramLite + Gated Skips + Full GPTQ + FA3.** Combines EngramLite (from #1089) with sigmoid-gated skip connections. 1-seed, 2 pending. | [#1122](https://github.com/openai/parameter-golf/pull/1122) |
| 1.1147 | @abaybektursun | 1 | **Negative results (7 experiments, all negative) on #1019 stack.** kNN-LM single/multi-layer (+0.003), sliding window logit averaging (+0.024), SelfExtend 4096 (+0.48), n-gram log-linear blend (−0.0003 but too slow), mixed-precision GPTQ int4/int8 (+0.047), loss truncation 95th pct (+0.081). From SOTA holder @abaybektursun. | [#1103](https://github.com/openai/parameter-golf/pull/1103) |
| 1.1154 | @AnubhavBharadwaaj | 3 | **SLOT + LeakyReLU² + Legal TTT + Parallel Muon.** Second SLOT submission (after #1084). SLOT delta vector + score-first TTT stacked. Std=0.0002. | [#1128](https://github.com/openai/parameter-golf/pull/1128) |
| 1.1164 | @Asukabot0 | 1 | XSA-all + LeakyReLU² + VRL + GA (no VE128). No TTT. 1xH100 NVL. Pending 8xH100 3-seed. | [#638](https://github.com/openai/parameter-golf/pull/638) |
| 1.1170 | @vimeto | 1 | **Fused Triton MLP kernel + Online Hessian GPTQ.** Custom Triton kernel fuses linear→LeakyReLU(0.5)→square: 70ms/step (vs 87ms), 33% more training steps. Hessian accumulated during training (every 25 steps) — eliminates GPTQ budget tradeoff. XSA-all, BigramHash(4096). No TTT. 1-seed, pending 3-seed. | [#1072](https://github.com/openai/parameter-golf/pull/1072) |
| 1.1171 | @raahilshah | 3 | XSA-all + Full GPTQ + Parallel Muon + Selective Pruning + LZMA. No TTT. Same #609 stack. 0.00394 nats (fails bar). | [#634](https://github.com/openai/parameter-golf/pull/634) |
| 1.1172 | @danialht | 3 | Residual Input Mixing + mixed int6 GPTQ + grouped TTT + MLP 3.5x. GPTQ timing fixed from #615. | [#790](https://github.com/openai/parameter-golf/pull/790) |
| 1.1174 | @unknown | 3 | **CROWN-Q + GPTQ + Legal TTT.** CROWN-Q regularization with standard GPTQ + TTT stack. Std unknown. | [#1129](https://github.com/openai/parameter-golf/pull/1129) |
| 1.1176 | @Gusanidas | 1 | MiLe loss + 8-bit Muon + Cache+Backout on #549. 4xB200 — needs H100. | [#703](https://github.com/openai/parameter-golf/pull/703) |
| 1.1179 | @aamodbhatt | 3 | **Muon TTT (Newton-Schulz orthogonalized, NS=3) + Entropy-Adaptive Epochs** (2/3/4 by chunk entropy, thresholds H=1.75/2.1). 11L, BigramHash(1536), XSA4, score-first TTT (LR=0.002, chunk=32768). No n-gram cache. Pre-quant 1.137, TTT gain -0.019. Std~0.0002. | [#999](https://github.com/openai/parameter-golf/pull/999) |
| 1.1180 | @hypery11 | 3 | 10L Batched LoRA TTT (rank-8, 3 epochs, 64 docs parallel). TTT gain: −0.033. Fails 0.005-nat bar. | [#713](https://github.com/openai/parameter-golf/pull/713) |
| 1.1182 | @msisovic | 3 | Depth Recurrence (layers 4+5 repeated, 11→13 virtual). +TTT. Std=0.0005. Fails bar. | [#686](https://github.com/openai/parameter-golf/pull/686) |
| 1.1184 | @yufengli-oai | 1 | **LeakyReLU² + 4-epoch Legal TTT** (lr=0.0025) on #549 stack. Skips diagnostic pre-TTT evals to keep eval under 10 min. | [#1039](https://github.com/openai/parameter-golf/pull/1039) |
| 1.1185 | @michaelwinczuk | 3 | LeakyReLU(0.75)² + Legal TTT + Parallel Muon (1.1185). 3-seed. Below official SOTA. | [#977](https://github.com/openai/parameter-golf/pull/977) |
| 1.1185 | @michaelwinczuk | 1 | **MTP-2 Funnel:** Multi-Token Prediction (2-head, weight=0.1) as auxiliary training signal on #549 stack. MTP heads discarded at export — zero artifact cost. LeakyReLU(0.75)². | [#1031](https://github.com/openai/parameter-golf/pull/1031) |
| 1.1185 | @AnubhavBharadwaaj | 3 | **First SLOT submission.** Optimizes single delta vector (512 dims) at last hidden layer per batch during eval. Stacks on TTT: -0.0008 BPB over baseline. Also tested CTW — negative result (+0.005 worse). Based on Hu et al. arXiv:2505.12392. Std=0.0003. | [#1084](https://github.com/openai/parameter-golf/pull/1084) |
| 1.1186 | @EthanYangTW | 3 | CROWN-Q + Full GPTQ (within training budget) + SWA/EMA + XSA-all + VRL. No TTT. | [#693](https://github.com/openai/parameter-golf/pull/693) |
| 1.1187 | @Upsalla | 3 | RoPE NTK-Scaling bug fix + BigramHash(3072) + Late QAT@0.57 + Legal TTT. Std=0.00024. | [#714](https://github.com/openai/parameter-golf/pull/714) |
| 1.1190 | @ChaosCodes | 1 | GPTQ int6 + SGD TTT + LeakyReLU² on #414 stack. A800 hardware (non-record). Est. ~1.122 on H100. | [#610](https://github.com/openai/parameter-golf/pull/610) |
| 1.1194 | @Joeavaib | 3 | 9L "Maestro" arch + LeakyReLU² + Legal TTT + Parallel Muon + GPTQ-lite + LZMA. Ties former SOTA (1.1194) (0.00006 nats). | [#625](https://github.com/openai/parameter-golf/pull/625) |
| 1.1198 | @Robby955 | 3 | Full GPTQ + XSA-4 + Score-First TTT. | [#734](https://github.com/openai/parameter-golf/pull/734) |
| 1.1215 | @aryanbhosale | 0 | 11L Parallel Muon + LN Scale + LeakyReLU² + Legal TTT (1.1215). | [#838](https://github.com/openai/parameter-golf/pull/838) |
| 1.1217 | @nothingLiva | 4 | **Adaptive Precision Embedding Quantization:** int8 for top-100 tokens (53% of text), int6 for rest. 15.8MB artifact. Std=0.0005. | [#1042](https://github.com/openai/parameter-golf/pull/1042) |
| 1.1219 | @autocode-rayes | 0 | Full-Training QAT (1.1219). | [#836](https://github.com/openai/parameter-golf/pull/836) |
| 1.1227 | @adityakm24 | 3 | XSA-7 + BigramHash + TrigramHash + ValueResidual + VE128 + Legal TTT (SGD 4-epoch) | [#1182](https://github.com/openai/parameter-golf/pull/1182) |
| 1.1231 | @pattern4bots | 0 | Frequency-Weighted Embedding Quantization (1.1231). Non-record. | [#898](https://github.com/openai/parameter-golf/pull/898) |
| 1.1246 | @unnir | 1 | 11L + Tight SWA (scale<0.2, zero penalty) + Shared VE128 (layers 9-10) + Partial RoPE + LN Scale + Late QAT + XSA4 + SmearGate + FA3 | [#374](https://github.com/openai/parameter-golf/pull/374) |
| 1.1247 | @greqone | 1 | #315 stack + Backout Connection + native FA3 + torch.compile | [#394](https://github.com/openai/parameter-golf/pull/394) |
| 1.1260 | @dannywillowliu-uchi | 1 | #374 stack + GPTQ-lite (per-layer clip percentile search). Self-Distillation TTT: neutral (−0.0003). | [#379](https://github.com/openai/parameter-golf/pull/379) |
| 1.1261 | @okezue | 1 | **Bayesian posterior packets** + selective gating on #549 stack. Conjugate online updating mixes training priors with eval-time counts. +0.0006 over pure neural TTT. | [#1043](https://github.com/openai/parameter-golf/pull/1043) |
| 1.1300 | @jimliu741523 | 0 | Poly5 Softcap + BigramHash(3072) + Wider GPTQ-lite. | [#816](https://github.com/openai/parameter-golf/pull/816) |
| 1.1335 | @Christopher-Lee-McClendon | 4 | E2E TTT-Linear (1.08M params) + 1-step FlowRefiner (flow-matching hidden-state refiner) | [#1166](https://github.com/openai/parameter-golf/pull/1166) |
| 1.1344 | @mrdavtan | 1 | **Compression moonshots (8 negative findings):** Procrustes (91% MSE reduction but 380% larger artifact), selective fp16 embedding, non-monotonic pruning+zstd. Key finding: int6+zstd is near-optimal. | [#1048](https://github.com/openai/parameter-golf/pull/1048) |
| 1.1354 | @ibarrajo | 1 | 11L + Partial XSA (last 3) + TTT + 524K batch + RoPE50K (no FA3) ⚠️ pre-eval TTT | [#290](https://github.com/openai/parameter-golf/pull/290) |
| 1.1354 | @simonbissonnette | 3 | 11L + EMA + BigramHash(12288) + Mixed Int5 + FA3 (fails p<0.01: t=−6.0 vs −7.0) | [#466](https://github.com/openai/parameter-golf/pull/466) |
| 1.1357 | @dennisimoo | 1 | 11L + XSA (last 4) + EMA + SmearGate + BigramHash(2048) + 524K batch + WD 0.04 + torch.compile (SDPA fallback) | [#307](https://github.com/openai/parameter-golf/pull/307) |
| 1.1365 | @ofirkris | 2 | 10L + XSA4 + EMA + Partial RoPE + LN Scale + Int5-MLP/Int6-Attn + 3.2% pruning. No TTT. | [#458](https://github.com/openai/parameter-golf/pull/458) |
| 1.1399 | @Mapika | 3 | 11L + XSA4 + EMA + SmearGate + BigramHash(2048) + Int5-MLP/Int6-Attn/Int8-Embed + 8% pruning (fails 0.005-nat by 0.00004) | [#349](https://github.com/openai/parameter-golf/pull/349) |
| 1.1400 | @aazizyan | 0 | First Viable 3-Loop Recurrence — Birkhoff + Output-LN + Timestep Scaling. | [#855](https://github.com/openai/parameter-golf/pull/855) |
| 1.1412 | @VirajDeshwal | 3 | Unified Attention (single W_unified replaces Q/K/V) + FA3 head-dim padding + Legal TTT | [#1202](https://github.com/openai/parameter-golf/pull/1202) |
| 1.1419 | @chris-buckley | 1 | 11L + XSA4 + EMA + TTT (pre-quant 1.1581; no FA3, SDPA fallback, 5344/9000 steps; seeds 2/3 pending) | [#317](https://github.com/openai/parameter-golf/pull/317) |
| 1.1431 | @SergheiBrinza | 3 | Turbo-Muon + EngramLite(10240x48) + VE(8,9,10) — wider EngramLite forced all-int5 + 20.5% pruning | [#1205](https://github.com/openai/parameter-golf/pull/1205) |
| 1.1448 | @BhatiaUday | 0 | LeakyReLU² + TrigramHashEmbedding. | [#884](https://github.com/openai/parameter-golf/pull/884) |
| 1.1509 | @Hilo-Hilo | 1 | **XSA-all-layers + VRL + bigram3072 + lzma9.** Negative finding: AdamW TTT at lr=0.002 degrades to 1.2804 (SGD better for TTT). 15.3MB artifact. | [#1045](https://github.com/openai/parameter-golf/pull/1045) |
| 1.1520 | @fielding | 0 | Knowledge Distillation — negative result (1.152). Non-record. | [#1029](https://github.com/openai/parameter-golf/pull/1029) |
| 1.1682 | @himanshudongre | 1 | **First zero-throughput-penalty SSM in Parameter Golf.** S4D-Lin hybrid: 2 SSM layers (learned exp-decay kernels via F.conv1d) + 9 Transformer layers (XSA). 116ms/step matching baseline. GPTQ-int5, 13MB artifact. Finding: attention > SSM in lower layers at this scale; GPTQ-int5 degrades SSM weights more. Non-record. | [#1013](https://github.com/openai/parameter-golf/pull/1013) |
| 1.1757 | @newjordan | 2 | Nightcrawler — 5F+1C+5F shared-weight crawler with TAP encoder connections. Extension of Crawler (#1140) | [#1208](https://github.com/openai/parameter-golf/pull/1208) |
| 1.1900 | @MVPandey | 0 | JEPA Self-Distillation with EMA Target Encoder. Novel architecture. Non-record. | [#896](https://github.com/openai/parameter-golf/pull/896) |
| 1.2135 | @nickferrantelive | 3 | 11L 512d Int8+Zlib Baseline. | [#858](https://github.com/openai/parameter-golf/pull/858) |
| 1.2174 | @Jayteare | 1 | **Adaptive Markov mixing:** 1024x1024 unigram transition table + learned per-position gate with confidence thresholding. 11L, 786K batch. No QAT, no EMA. | [#1046](https://github.com/openai/parameter-golf/pull/1046) |
| 1.2300 | @andrewmouldon | 3 | **ASQU (Asymmetric Squared Unit):** Per-channel learned asymmetric activation. Consistent -0.0011 BPB vs LeakyReLU² across 3 seeds. Non-record track. | [#1035](https://github.com/openai/parameter-golf/pull/1035) |
| 1.2560 | @serdardoesml | 1 | **Universal Transformer (README wishlist).** Shared UT-style recurrent block with 2x attention before 3-layer MLP. Depth scheduling (0:2, 2000:6). Bias on pre-norms acts as depth embedding. Noisy QAT. Non-record. | [#1088](https://github.com/openai/parameter-golf/pull/1088) |
| 1.3557 | @raider99k | 1 | Recurrent block_anchor 0011+g2 R768 with grouped anchor gating | [#1203](https://github.com/openai/parameter-golf/pull/1203) |
| 1.3600 | @ikermoel | 3 | **Masked Diffusion (MDLM):** Bidirectional attention training, pseudo-log-likelihood eval (8 passes x 50% mask). 12.9MB artifact. First text diffusion submission (README wishlist). | [#1053](https://github.com/openai/parameter-golf/pull/1053) |
| 1.3646 | @wfproc | 1 | **QAT dead-code bug confirmed in SOTA #549** (torch.compile constant-folds Late QAT). Fix via tensor-scale STE worsens int6 gap — WD+EMA may already compensate. 7 techniques swept (Muon-VS, Deep Delta, Thinking Deeper, etc.), all negative. Key heuristic: 1ms overhead = 0.007 BPB at 83ms/step. 1xH100, non-record research. | [#1032](https://github.com/openai/parameter-golf/pull/1032) |
| 1.3700 | @abaybektursun | 0 | **Negative result: MC Dropout (K=16 ensemble) hurts small LMs.** dropout=0.30: +0.005 BPB, dropout=0.05: +0.002 BPP. Sub-networks lack diversity at 17M params. Deterministic single pass strictly better. | [#1021](https://github.com/openai/parameter-golf/pull/1021) |
| 1.5134 | @AnirudhRahul | 0 | **Corrected full-vocab normalization rerun.** With proper normalization, n-gram path degrades to 1.5134 — WORSE than neural sliding baseline. The n-gram cache 'improvement' was entirely a normalization artifact. | [#978](https://github.com/openai/parameter-golf/pull/978) |
| 1.8989 | @greqone | 1 | **H-Net:** First learned byte-level tokenization (README wishlist). Differentiable chunking gate, 22M params, vocab=260. Non-record unlimited compute track. | [#1044](https://github.com/openai/parameter-golf/pull/1044) |

</details>

<details>
<summary><strong>Glossary</strong></summary>

| Term | Meaning |
|------|---------|
| **BPB** | Bits Per Byte — compression quality. Lower = better |
| **val_bpb** | BPB on FineWeb validation set |
| **Muon** | Optimizer: orthogonalized gradients via Newton-Schulz |
| **QAT/STE** | Quantization-Aware Training / Straight-Through Estimator |
| **Int6/Int8** | 6-bit or 8-bit integer quantization |
| **SWA/EMA** | Stochastic Weight Averaging / Exponential Moving Average |
| **TTT** | Test-Time Training — adapting during evaluation |
| **XSA** | Exclusive Self-Attention — removes self-value bias |
| **FA3** | FlashAttention 3 — optimized H100 attention kernel |
| **LoRA** | Low-Rank Adaptation — tiny trainable matrices |
| **zstd** | Zstandard compression (better than zlib) |

</details>


---

<details>
<summary><strong>Changelog</strong></summary>

| Time | Update |
|------------|--------|
| Mar 31, 11:03 PM | +2 lineage entries (#1212 Window Attn, Mixed SeqLen) |
| Mar 31, 10:51 PM | +#1212 (1.1108, Window Attn + Mixed SeqLen) |
| Mar 31, 10:00 PM | +#1209 (1.1064, GPTQ+TTT+SLOT, record) |
| Mar 31, 9:21 PM | +#1208 (1.1757, Nightcrawler architecture) |
| Mar 31, 6:20 PM | +#1205 (1.1431, wider EngramLite negative result) |
| Mar 31, 6:13 PM | Fix #862 placeholder, typo, stale SOTA refs, +lineage |
| Mar 31, 5:50 PM | +#1204 (1.1063, Parallel Residuals + Mini Depth Recurrence) |
| Mar 31, 4:01 PM | Research cycle: +3 lineage, fix #1180 author, prune changelog, update predictions/negatives |
| Mar 31, 3:52 PM | Fix 7 stale SOTA refs (1.1194→1.1147) |
| Mar 31, 3:35 PM | #741 closed |
| Mar 31, 3:34 PM | #728,#1060,#1099,#1130 → unvalidated (new SOTA) |
| Mar 31, 3:34 PM | +#1202, #1182, #1166, #1203 |
| Mar 31, 3:34 PM | +#1185 (0.9641, n-gram backoff cache) |
| Mar 31, 3:34 PM | +#1105 (1.1125, CUTLASS EVT + MLP 3.5x) |
| Mar 31, 3:34 PM | Official SOTA → 1.1147 (#1019 merged) |
| Mar 31, 2:19 PM | Added sub-0.9 roadmap + Over-Encoding technique |
| Mar 31, 9:11 AM | +#1184 (0.9485, Scylla+GPTQ — first sub-1.0 validated!) |
| Mar 31, 6:21 AM | #1176 updated: 1.0962→1.0914 |
| Mar 31, 6:18 AM | Added QK-Gain, P2/Focal Loss, Conv Mixer to techniques |
| Mar 31, 4:41 AM | +#1180 (1.0577, P2 loss + conv mixer — 1 seed) |
| Mar 31, 2:51 AM | +#1176 (1.0962, QK-Gain+SLOT — new best standard tok!) |
| Mar 31, 2:10 AM | #1172 reopened — reverted to record |
| Mar 31, 12:11 AM | #1172 closed (SLOT+GPTQ 1.1015) |
| Mar 30, 11:33 PM | Enriched #1172 (SLOT replaces TTT — key finding) |
| Mar 30, 11:31 PM | +#1172 (1.1015, SLOT+GPTQ — new best standard tokenizer!) |
| Mar 30, 12:19 PM | Added tokenizer deep-dive from research |
| Mar 30, 12:05 PM | Enriched #1143 (Scylla tokenizer deep-read) |
| Mar 30, 12:01 PM | +#1145 (1.1109, online ngram augment) |
| Mar 30, 12:01 PM | +#1143 (1.0806, Scylla tokenizer — NEW BEST!) |
| Mar 30, 8:41 AM | +#1135 (1.1116, Fused Triton + Full GPTQ) |
| Mar 30, 4:32 AM | #1100 closed, fixed #1060 BPP typo |
| Mar 30, 4:10 AM | +#1130 (1.1140, 12-seed, ResidLambdas) |
| Mar 30, 3:01 AM | Batch: +#1128 +#1129 +#1122 (below-SOTA submissions) |
| Mar 30, 12:17 AM | Added CUTLASS EVT, gated skips, Brotli+shuffle to techniques |
| Mar 29, 10:00 PM | +#1120 (1.1099, Rascal — 2nd best pure neural) |
| Mar 29, 9:01 PM | Added nGPT #1108 and UT #1088/#1110 to non-record |
| Mar 29, 4:41 PM | +#1103 (7 negative results from SOTA holder) |
| Mar 29, 3:50 PM | #1099 updated: 1.1136→1.1133 |
| Mar 29, 3:02 PM | Updated predictions, +#1100 to non-record |
| Mar 29, 2:51 PM | +#1100 (1.1465, first diffusion < AR baseline) |
| Mar 29, 2:51 PM | +#1099 (1.1136, coprime stride record) |
| Mar 29, 1:40 PM | +#1095 (0.0905, seed-regen + n-gram, 1-seed MI250X) |
| Mar 29, 1:00 PM | +#1094 (0.4027, causal BackoffNgramMixer) |
| Mar 29, 11:32 AM | Turbo-Muon + Engram marked tried per #1089 |
| Mar 29, 11:11 AM | +#1088 (1.256, Universal Transformer wishlist) |
| Mar 29, 11:11 AM | +#1089 (1.1086, new best pure neural!) |
| Mar 29, 9:41 AM | SLOT/CTW marked tried, +fused kernel +online GPTQ |
| Mar 29, 9:39 AM | +#1072 (1.117, fused Triton kernel) |
| Mar 29, 9:39 AM | +#1084 (1.1185, first SLOT submission) |
| Mar 29, 9:39 AM | +#1083 (0.4961, Bandit n-gram) |
| Mar 29, 1:40 AM | #1060 updated: 1.1123→1.1122 |
| Mar 29, 12:21 AM | Added 3 new techniques from web research |
| Mar 29, 12:01 AM | Fixed stale TARA present-tense in tier_analysis |
| Mar 28, 11:33 PM | Deep research: rewrote predictions for #1060 era |
| Mar 28, 11:33 PM | #1055 closed (TARA causality violation) |
| Mar 28, 11:20 PM | +#1060 (1.1123, pure neural below SOTA, no TTT) |
| Mar 28, 11:05 PM | Deep research: fixed stale #933 ref, pred #2, +#1012 |
| Mar 28, 10:41 PM | Added #1032 QAT dead-code finding to deep dives |
| Mar 28, 10:35 PM | Deep research: fixed lineage, +#1032 +#1013 |
| Mar 28, 10:31 PM | Updated stale summary and pending_summary prose |
| Mar 28, 10:29 PM | #754 closed |
| Mar 28, 10:29 PM | +#1055 (0.9693, TARA — training-free eval) |
| Mar 28, 10:27 PM | Added 7 new techniques from web research |
| Mar 28, 10:27 PM | Closed 20 PRs from enforcement lists |
| Mar 28, 10:14 PM | Deep research: closed 11 PRs, enriched 5, fixed lineage |
| Mar 28, 10:05 PM | #995 closed |
| Mar 28, 10:05 PM | +#1053 Masked Diffusion: first text diffusion |
| Mar 28, 10:05 PM | +#1044 H-Net: first byte-level tokenization |
| Mar 28, 10:05 PM | +#1042 (1.1217, Adaptive Precision Embedding) |
| Mar 28, 10:05 PM | +#1047 (0.8822, DeltaNet Crawler S2) |
| Mar 28, 10:00 PM | +#1055 (0.97, TARA training-free), +#1056 (0.0180, Dirichlet) |
| Mar 28, 11:26 AM | +#1030 (0.1130, Dirichlet CTW post-sweep), +#1029 (KD negative) |
| Mar 28, 9:34 AM | +#1019 (1.1147, AR Self-Gen GPTQ from SOTA holder), +#1026 (1.0945) |
| Mar 27, 5:10 PM | +#991 (1.1145, pure neural below SOTA) |
| Mar 27, 4:21 PM | Enforcement sweep: @valerio-oai closed 33 PRs (hashed n-gram caches invalidated) |
| Mar 27, 12:50 PM | #978: corrected normalization → n-gram cache degrades to 1.51, WORSE than neural baseline |
| Mar 27, 11:51 AM | N-gram normalization bug: #931/#962 closed. #933/#944 affected. Sub-0.1 scores likely artifacts. |
| Mar 27, 8:30 AM | +#962 (0.0214, low eval-time memory packed n-gram) |
| Mar 27, 2:11 AM | +#944 (0.0165, packed causal memory + Dirichlet — new best validated) |
| Mar 19-26 | Initial build through n-gram cache wave (see git history) |

</details>


---