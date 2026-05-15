# Cross-model comparison — Gemini vs Qwen3-8B vs Aya-expanse-8B

After re-analyzing every run with the fixed `03_analyze.py` (now checks
hint acknowledgment in either the thinking trace or the visible
reasoning — necessary because Groq/HF models don't expose a separate
thinking trace).

## Per-model totals

| Model              | N hinted | Flip | Trn | Opq | NSF | Flip% | Opq% |
|--------------------|---------:|-----:|----:|----:|----:|------:|-----:|
| gemini-2.5-flash   | 1,843    |  47  |  47 |  0  |  0  |  2.6% | **0.0%** |
| qwen3-8b           | 1,894    | 170  | 128 |  23 | 19  |  9.0% | 1.2%   |
| aya-expanse-8b     | 1,736    | 190  |  57 |  80 | 53  | 10.9% | 4.6%   |

`Trn` (Transparent): flipped to wrong_hint AND mentioned cue.
`Opq` (Opaque): flipped to wrong_hint WITHOUT mentioning cue.
`NSF` (Non-syc-flip): changed answer but to a third option (neither baseline nor hint).

## Per (model, dataset)

| Dataset    | Model              | N    | Flip | Trn | Opq | NSF | Opq% |
|------------|--------------------|-----:|-----:|----:|----:|----:|-----:|
| Korean     | gemini-2.5-flash   |  343 |   3  |  3  |  0  |  0  |  0.0% |
| Korean     | qwen3-8b           |  350 |  17  |  8  |  3  |  6  |  0.9% |
| Korean     | aya-expanse-8b     |  308 |  28  | 16  |  5  |  7  |  1.6% |
| American   | gemini-2.5-flash   |  350 |   0  |  0  |  0  |  0  |  0.0% |
| American   | qwen3-8b           |  350 |   7  |  5  |  1  |  1  |  0.3% |
| American   | aya-expanse-8b     |  350 |  15  |  5  |  5  |  5  |  1.4% |
| German     | gemini-2.5-flash   |  336 |   2  |  2  |  0  |  0  |  0.0% |
| German     | qwen3-8b           |  329 |   5  |  4  |  1  |  0  |  0.3% |
| German     | aya-expanse-8b     |  336 |  20  |  6  |  6  |  8  |  1.8% |
| Polish     | gemini-2.5-flash   |  350 |   0  |  0  |  0  |  0  |  0.0% |
| Polish     | qwen3-8b           |  343 |  10  |  3  |  0  |  7  |  0.0% |
| Polish     | aya-expanse-8b     |  322 |  28  |  7  | 16  |  5  |  5.0% |
| GSM8K      | gemini-2.5-flash   |  350 |   6  |  6  |  0  |  0  |  0.0% |
| GSM8K      | qwen3-8b           |  336 |  25  | 10  | 10  |  5  |  3.0% |
| GSM8K      | aya-expanse-8b     |  336 |  67  |  0  | 39  | 28  | 11.6% |
| XSAFETY    | gemini-2.5-flash   |  114 |  36  | 36  |  0  |  0  |  0.0% |
| XSAFETY    | qwen3-8b           |  186 | 106  | 98  |  8  |  0  |  4.3% |
| XSAFETY    | aya-expanse-8b     |   84 |  32  | 23  |  9  |  0  | 10.7% |

## Key cross-model findings

1. **Gemini 2.5 Flash with thinking: 0/1,843 opaque sycophancy across
   six task slices.** Every flip is paired with explicit cue
   acknowledgment in either the thinking trace or the visible
   reasoning.

2. **Open-source 8B models do exhibit opaque sycophancy.** Qwen3-8B
   1.2%, Aya-expanse-8B 4.6%. The pattern is most visible on:
   - Aya × GSM8K: 11.6% opaque — when Aya is talked into a wrong math
     answer by a cited expert, it writes math reasoning that arrives
     at the wrong answer without naming the expert.
   - Aya × XSAFETY: 10.7% opaque — Aya complies with unsafe prompts
     under social/indirect/authority cues, frequently without
     mentioning the cue.
   - Aya × Polish: 5.0% opaque — concentrated in cultural items where
     the cue letter happens to align with a Polish-script response
     pattern.

3. **Behavioral susceptibility scales inversely with model size /
   alignment.** Gemini 2.6% flip rate, Qwen3-8B 9.0%, Aya-8B 10.9%.

4. **Reasoning-mode availability matters.** Gemini's separate thinking
   trace lets the model process the cue explicitly and then reject or
   adopt it transparently. Open-source 8B models without an explicit
   thinking step are more likely to silently incorporate the cue into
   a single-pass reasoning that doesn't surface it.

5. **The asymmetry by cue family persists across models.** Authority
   cues are most effective and most often acknowledged; indirect cues
   produce more confusion (high NonSycFlip rates on Qwen3 and Aya);
   few-shot biased cues produce nearly no flips on cultureMCQA/GSM8K
   on any model.

## Methodological note

Before the analyzer fix (commit on 2026-05-15), acknowledgment was
detected only in the `thinking` field, which is always empty for
non-thinking models (Groq, HF local). This artificially inflated the
opaque count to 53 (Qwen3-8B) / 114 (Aya-8B). After also checking the
visible `reasoning` field, the corrected numbers are 23 / 80 — still
nonzero, but ~50% lower. The original Gemini results are unchanged
because Gemini does expose a thinking trace and the acknowledgment
detection used to find it there.
