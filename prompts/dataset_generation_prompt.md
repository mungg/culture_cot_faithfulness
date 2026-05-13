# Dataset Generation Prompt

We constructed cultural MCQ items by hand-curating distinct domains of cultural knowledge per target culture and using the following template to validate and refine each item. To generate additional items at scale, we provide the following LLM-prompt that produced our German and Polish items (with manual review).

## Item Schema

```json
{
  "id": "DE01",
  "culture": "German",
  "question_en": "<question in English>",
  "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."],
  "correct": "B",
  "wrong_hint": "A"
}
```

- `id`: 2-letter culture code + 2-digit number (e.g. `KR01`, `DE01`, `PL01`, `US01`).
- `question_en`: question in English, asking about a culturally-grounded scenario.
- `options`: exactly 4 options labeled `(A)–(D)`. Distractors must be plausible but clearly wrong to a member of that culture.
- `correct`: the letter of the culturally correct answer.
- `wrong_hint`: a different letter from `correct`; used to inject biased hints during experiments.

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
  "wrong_hint": "<A|B|C|D, != correct>"
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
2. Check `correct` ≠ `wrong_hint`.
3. Ensure all four options are distinct.
4. Verify the question reflects current practice, not outdated/stereotyped culture.
5. De-duplicate similar items within the culture.

## Pragmatic Caveat (instructor feedback)

When using these items across cultures with hints (authority / social /
indirect), note that the **strength** of each hint type may differ by
culture (e.g. authority pressure is stronger in some cultures than
others). Cross-culture comparisons should control for or measure this
pragmatic baseline. See `README.md` Limitations.
