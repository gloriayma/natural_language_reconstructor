# Word / token list

The stratified word+token sample used across the interp experiments. Lives in
`experiments/words.json` (84 strings; per-model special tokens added by the runner).
History at the bottom.

## Composition

- **POS strata**: common nouns (8: dog, table, water, music, idea, house, science, coffee),
  proper nouns (6: London, Einstein, Google, Tokyo, Shakespeare, Beyoncé), verbs (6, mixed
  tense: run, walked, is, think, understand, procrastinate), prepositions (6: of, in, under,
  between, through, despite), adjectives (5: big, happy, enormous, purple, incomprehensible),
  adverbs (3: quickly, very, often), pronouns (3: she, they, ourselves), determiners (3: the,
  an, this), conjunctions (3: and, but, although), interjections (3: hello, wow, hmm),
  numbers (3: seven, 42, 3.14).
- **Multi-token by design** (5): serendipity, antidisestablishmentarianism, cryptocurrency,
  unbelievable, biotechnology. (What's multi-token varies per tokenizer; per-model outputs
  record the actual split.)
- **Weird-but-used**: punctuation (12: . , ! ? ; : ' " ( ) — …), symbols (5: @ # $ % ~),
  whitespace (space, newline, tab), subword fragments ('s, n't, ing, pre).
- **Very weird**: glitch tokens (" SolidGoldMagikarp", " petertodd" — single tokens only in
  GPT-2-family vocabs), a NUL control byte, reserved specials
  (`<|reserved_special_token_0|>`, `<|extra_0|>`), `<unk>`.
- **Special tokens, added per model from its tokenizer**: BOS, EOS, PAD, UNK.

**No holdout**: everything above is analysis, including BOS/EOS/EOM-style tokens and the
glitch/control entries (user clarification 2026-07-05 — an earlier version had a `held_out`
flag; removed).

## Conventions

- Tokenized fresh, no context: bare word, no leading space, `add_special_tokens=False`.
  Many BPE vocabs distinguish ` dog` from `dog`; we test the bare form and record the actual
  tokens used, so the per-model CSVs show each tokenizer's real split.
- Category labels are informal (e.g. "run" could be noun or verb) — they're strata labels for
  coverage, not gold POS tags.

## Snags

- The NUL-byte entry kept getting written to disk as a literal `0x00` (invalid JSON) rather
  than the escaped form — twice, once in words.json and once in a doc. If words.json breaks,
  check for that first (`python -c "import json; json.load(open('experiments/words.json'))"`).

## History

- **2026-07-05**: List created (84 strings). Same day: user clarified BOS/EOS should not be
  held out, then that there should be **no holdout concept at all** — `held_out` flags removed
  from words.json and the runner.
