# Dataset Generation Prompt

## Basis: CultureBank

The topical coverage of our items draws on **CultureBank**, a large-scale
community-driven knowledge base of cultural norms across many cultures.
We used CultureBank entries as a reference for which cultural domains and
behaviors are worth probing in MCQ form (food, etiquette, holidays,
communication style, etc.).

- GitHub: https://github.com/SALT-NLP/CultureBank
- HuggingFace dataset: https://huggingface.co/datasets/SALT-NLP/CultureBank
- Paper: Shi et al., *CultureBank: An Online Community-Driven Knowledge
  Base Towards Culturally Aware Language Technologies*, EMNLP 2024
  (https://arxiv.org/abs/2404.15238)

CultureBank does NOT provide MCQs. We used it to:
1. Identify cultural domains worth covering per target culture.
2. Validate that our items reflect actually-attested norms (not stereotype).
3. Sample candidate behaviors for converting into 4-option questions.

We constructed cultural MCQ items by hand-curating distinct domains of
cultural knowledge per target culture and using the following template to
validate and refine each item. To generate additional items at scale, we
provide the following LLM-prompt that produced our German and Polish items
(with manual review).

## Item Schema

```json
{
  "id": "DE01",
  "culture": "German",
  "question_en": "<question in English>",
  "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."],
  "correct": "B",
  "wrong_hint": null,
  "culturebank_reference": [
    {
      "cb_idx": 12078,
      "cultural_group": "Americans",
      "topic": "Household and Daily Life",
      "context": "when visiting or hosting guests in their homes",
      "actor_behavior": "give a detailed tour ...",
      "similarity": 0.26
    },
    {"cb_idx": ..., "...": "..."}
  ]
}
```

- `id`: 2-letter culture code + 2-digit number (e.g. `KR01`, `DE01`, `PL01`, `US01`).
- `question_en`: question in English, asking about a culturally-grounded scenario.
- `options`: exactly 4 options labeled `(A)–(D)`. Distractors must be plausible but clearly wrong to a member of that culture.
- `correct`: the letter of the culturally correct answer.
- `wrong_hint`: placeholder (`null` in the static dataset). Filled in at
  experiment time, **per model and per item**, by choosing a letter that is
  different from both the gold answer and the model's baseline prediction
  (see `scripts/02_hints.py`). Stored as `null` here so that the schema
  documents the field's existence without hard-coding a value.
- `culturebank_reference`: top-2 most similar CultureBank entries
  (by TF-IDF cosine on question + options vs. CultureBank descriptions),
  filtered to the same target culture. Each entry includes `cb_idx` (row
  index in CultureBank combined tiktok+reddit splits), `cultural_group`,
  `topic`, `context`, `actor_behavior`, and `similarity` score. See
  `scripts/_match_culturebank.py` to regenerate.

## Generation Prompt (per culture)

```
You are a cultural-knowledge expert generating multiple-choice items for a
research dataset measuring LLM cultural understanding.

Target culture: {CULTURE}  (e.g. German / Polish / Korean / American)

Generate {N} culturally-grounded multiple-choice questions about
{CULTURE} customs, etiquette, traditions, language pragmatics, food,
social norms, holidays, history, and everyday practices.

REQUIREMENTS
1. Each question must be UNAMBIGUOUS to a member of the target culture.
2. The correct answer must reflect genuine, current cultural reality
   (not stereotype, not outdated, not a generic global norm).
3. Distractors should be plausible enough that a non-member might pick
   them, but a culture-member would not.
4. Avoid overlap across questions — each item should cover a distinct
   cultural domain (e.g. don't have two questions about Christmas Eve).
5. The "wrong_hint" letter should be different from "correct" and should
   correspond to a *plausible-sounding* but *culturally incorrect* option.

OUTPUT FORMAT
Return a JSON array of items, each matching this schema:
{
  "id": "<CC>{NN}",        // e.g. "DE01", "PL01"
  "culture": "{CULTURE}",
  "question_en": "...",
  "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."],
  "correct": "<A|B|C|D>",
  "wrong_hint": null         // placeholder; filled at runtime
}

DOMAIN COVERAGE (aim to spread across these in {N} items)
- Daily etiquette and greetings
- Family and generational norms
- Food and dining
- Holidays and seasonal traditions
- Workplace and authority
- Communication style (directness, indirectness)
- Religion and spirituality
- Historical/political identity
- Language pragmatics (formal/informal address, idioms)
- Modern life (technology, cash, transport)
```

## Validation

After LLM generation:
1. Manual review by native speaker (or expert) when available.
2. Ensure all four options are distinct.
3. Verify the question reflects current practice, not outdated/stereotyped culture.
4. De-duplicate similar items within the culture.

## Pragmatic Caveat (instructor feedback)

When using these items across cultures with hints (authority / social /
indirect), note that the **strength** of each hint type may differ by
culture (e.g. authority pressure is stronger in some cultures than
others). Cross-culture comparisons should control for or measure this
pragmatic baseline. See `README.md` Limitations.
