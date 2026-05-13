# Culture-CoT Faithfulness

Do LLMs reason **faithfully** when answering culturally-grounded questions,
or do they get nudged by social/authority/indirect hints injected into the
prompt? And does the language of the hint (matched vs. mismatched to the
culture) change the outcome?

This is a multilingual / multicultural extension of
**Turpin et al. (2024)**'s CoT-faithfulness work — but instead of generic
reasoning tasks, we use cultural-knowledge MCQs across **Korean, American,
German, and Polish** cultures, with hints in **English and Korean** (and
templates for German / Polish).

Models tested: Gemini 2.5 Flash (with `thinking_budget=8000` and
`include_thoughts=True`, so we can inspect the reasoning trace separately
from the final answer).

## Research question

Do hints injected after a culturally-grounded question:
1. change the model's final answer? (sycophancy)
2. surface in the model's thinking trace? (transparency)
3. trigger response-language code-switching? (style transfer)
4. cause empty thinking traces? (silent refusal)

And how do these depend on **hint language** (matched / mismatched to the
culture), **hint type** (authority / social / indirect), and **culture**
(Korean / American / German / Polish)?

## Quick start

```bash
# 1. Install
pip install google-genai
gcloud auth application-default login   # for Vertex AI

# 2. Set your project in scripts/01_generate_answers.py
#    (PROJECT_ID, LOCATION, MODEL_ID)

# 3. Run a small smoke test (10 items per culture, all 7 conditions)
python scripts/01_generate_answers.py --culture all --limit 10 --run-name smoke

# 4. Compute metrics
python scripts/02_analyze.py --run-name smoke
```

Full run: drop `--limit` to use all 130 items × 7 conditions = 910 calls.

## Repo layout

```
culture_cot_faithfulness/
├── data/
│   ├── cultural_items_all.json    # 130 items, all cultures combined
│   ├── korean.json                # 15 items
│   ├── american.json              # 15 items
│   ├── german.json                # 50 items
│   └── polish.json                # 50 items
├── prompts/
│   ├── dataset_generation_prompt.md  # how to create new items
│   └── hint_templates.py             # authority / social / indirect, en/ko/de/pl
├── scripts/
│   ├── 01_generate_answers.py     # main driver — uses Gemini structured output (JSON)
│   └── 02_analyze.py              # compute metrics
└── results/                       # output dir (gitignored)
```

## Dataset

130 multiple-choice cultural-knowledge items across 4 cultures:

| Culture  | Items | Code |
|----------|------:|------|
| Korean   | 15    | KR   |
| American | 15    | US   |
| German   | 50    | DE   |
| Polish   | 50    | PL   |

Each item: 4-option MCQ, gold `correct` letter, and a `wrong_hint` letter
(used to inject biased hints). See
[`prompts/dataset_generation_prompt.md`](prompts/dataset_generation_prompt.md)
for the schema and the LLM-prompt used to generate items.

## Experimental conditions

For each item, 7 conditions are run:

| Condition       | Hint type | Hint language |
|-----------------|-----------|---------------|
| `baseline`      | (none)    | (none)        |
| `authority_en`  | authority | English       |
| `authority_ko`  | authority | Korean        |
| `social_en`     | social    | English       |
| `social_ko`     | social    | Korean        |
| `indirect_en`   | indirect  | English       |
| `indirect_ko`   | indirect  | Korean        |

Hint templates for German (`de`) and Polish (`pl`) are also provided in
`prompts/hint_templates.py` for parallel "matched-culture" experiments.

## Metrics produced by `03_analyze.py`

1. **Baseline correctness** per culture.
2. **Answer-change rate** under each hint (filtered to items the model
   answered correctly at baseline).
3. **Sycophancy rate** = fraction of changes that go to `wrong_hint`.
4. **Hint-acknowledgment rate** in the thinking trace (keyword match).
5. **Empty-thinking rate** per condition.
6. **Response code-switching** rate (response language ≠ English).

## Current findings (Gemini 2.5 Flash, thinking_budget=8000, KR+US, n=30)

- **0% answer change** across all 7 conditions — model never flips its
  final answer. Sycophancy is null on this model.
- **Hint acknowledgment** varies sharply by hint type:
  authority/social ≈ 100%, indirect ≈ 40–55%.
- **Empty thinking** strongly concentrated on
  `Korean items × Indirect EN hint` (~70%).
- **Code-switching** (response in Korean) triggered by Korean hint,
  ~30% overall; thinking trace stays in English.

Extending to German + Polish is the natural next step.

## Limitations (feedback-driven)

**Pragmatic non-equivalence of hint types across cultures.**
The same hint template (e.g. "a professor says X") may carry different
pragmatic weight in different cultures:
- Authority hint may be stronger in cultures with steeper power distance.
- Indirect hint may feel more natural in high-context cultures.

Cross-culture comparisons therefore conflate (a) the linguistic content
of the hint with (b) culturally-conditioned compliance norms. Future work
should quantify perceived hint strength per culture (e.g. ask a separate
LLM-rater for 1–5 perceived pressure) and normalize before comparing.

## TODO

### Methodology (feedback-driven)
- [ ] Quantify per-culture pragmatic strength of each hint type
      (e.g. LLM-rater scoring perceived pressure 1–5) and normalize before
      cross-culture comparison
- [ ] Native-speaker review pass on German / Polish items

### Deepening existing findings
- [ ] Root-cause analysis of empty-thinking traces
      (token-distribution and length comparison across conditions)
- [ ] Per-condition code-switching breakdown
      (response language conditional on hint language × culture)

### Scale up
- [x] Add German + Polish item sets (50 each)
- [ ] Scale Korean and American to 50 items each
- [ ] Run full Gemini 2.5 Flash sweep across all 4 cultures × 7 conditions

### Model generalization
- [ ] Add open-source models: Qwen 3.5, DeepSeek
- [ ] Add weaker baseline: GPT-3.5 / Gemini 1.5 Flash

### Hint design
- [ ] Few-shot biased CoT (Turpin et al. original method)
- [ ] Multi-turn persuasion ("Are you sure? Professor disagrees…")
- [ ] Citation / expert framing variants

### Task expansion
- [ ] XSAFETY refusal × hints (suggested by feedback)
- [ ] BIG-Bench Hard × hints

### Deliverables
- [ ] 4–6 final figures with statistical significance
- [ ] Writeup: methods (note pragmatic limitation), results, discussion

## License

Research / academic use. Cite the original Turpin et al. (2024) paper for
the underlying CoT-faithfulness methodology.
