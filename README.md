# BVRIT Hyderabad FAQ Chatbot

A **RAG-powered** (Retrieval-Augmented Generation) FAQ chatbot for **BVRIT HYDERABAD College of Engineering for Women** — built from scratch with plain Python, grounded in a curated knowledge-base PDF, answering only from retrieved context with citations, and rigorously evaluated through an 8-dimension + RAGAS test suite and a multi-framework governance audit.

---

## Table of Contents

- [Stack](#stack)
- [Architecture](#architecture)
- [Knowledge Base](#knowledge-base)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval](#retrieval)
- [Grounding Prompt](#grounding-prompt)
- [Multi-Turn Context & Memory](#multi-turn-context--memory)
- [Evaluation Suite](#evaluation-suite)
- [Observability](#observability)
- [Governance Audit](#governance-audit)
- [Cost](#cost)
- [Setup](#setup)
- [Running It](#running-it)
- [Running the Evaluation Suite](#running-the-evaluation-suite)
- [Running the Governance Audit](#running-the-governance-audit)
- [Project Layout](#project-layout)
- [Notes](#notes)

---

## Stack

| Component | Technology |
|---|---|
| **Orchestration** | Plain Python (no LangChain) — dependency-free loader, chunker, retriever |
| **Vector Store** | ChromaDB (persistent, local) |
| **Embeddings** | `openai/text-embedding-3-small` via OpenRouter |
| **Chat Model** | `openai/gpt-4o-mini` via OpenRouter |
| **Judge / Test-Gen Model** | `openai/gpt-4o` via OpenRouter |
| **UI** | Streamlit (Chat page + Evaluation Dashboard) |
| **Evaluation** | Custom 8-dimension LLM-as-judge suite + `ragas` library |
| **Governance** | Giskard, Promptfoo, DeepEval |

---

## Architecture

```
data/bvrit_college_info.pdf     (knowledge base, 10 sections)
        |
   src/loader.py                (PDF → section/page-tagged text blocks)
        |
   src/chunker.py               (recursive character split, 400/60)
        |
   src/ingest.py                (embed + persist to ChromaDB)
        |
   vectorstore/                 (persistent Chroma collection)
        |
   src/retriever.py             (embed query, top-k search, section filter)
        |
   src/generation.py            (grounding prompt + chat completion)
        |
   src/pipeline.py              (orchestrates retrieve + generate, latency, history, memory, observability)
        |
   app.py + pages/0_Chat.py + pages/1_Evaluation_Dashboard.py   (Streamlit UI)
```

All LLM/embedding calls route through **OpenRouter** (`src/config.py`), using OpenAI-family models throughout. The evaluation suite and governance harnesses reuse the same pipeline code as the live app — no separate mocks.

---

## Knowledge Base

`data/generate_kb_pdf.py` builds `bvrit_college_info.pdf` programmatically from content transcribed off the official BVRIT Hyderabad course reference document. The PDF contains **10 sections** covering:

1. **About BVRIT Hyderabad** — history, vision, mission, accreditation
2. **Departments** — CSE, CSE (AI&ML), IT, ECE, EEE, Humanities & Sciences
3. **Admissions** — B.Tech, M.Tech, MBA eligibility and EAMCET/GPAT counselling
4. **Fee Structure** — tuition and special fees per branch per year
5. **Placements** — statistics, top recruiters, training programs
6. **Campus & Facilities** — infrastructure, library, labs, sports, transport, hostels
7. **Faculty** — departments, qualifications, areas of expertise
8. **Contact** — address, phone, email, social media, grievance portal
9. **Policies** — anti-ragging, code of conduct, scholarships
10. **Student Life** — clubs, events, extracurriculars, counselling

Each section is placed on its own PDF page via `PageBreak()`, so `page number == section index + 1` deterministically — `src/loader.py` recovers page numbers from pypdf and detects section boundaries via a numbered, all-caps heading regex.

**Evolution (why the document was rewritten):** The original synthetic document deliberately omitted fixed admission dates and several department/faculty facts — this meant the chatbot correctly refused date-comparison questions (e.g. "has the EAMCET counselling deadline passed?") for lack of any date to ground on, even though `date_checker`/`percentage_checker` tool support existed. The current document carries real, specific dates (§3) and percentages (§4, §5) so those tools actually get exercised.

Each fact in a citation link backs to the actual `bvrithyderabad.edu.in` web page via `src/sources.py`'s URL resolution — department-specific pages for faculty/department queries, and section-based defaults for everything else.

---

## Chunking Strategy

`chunk_size=400, chunk_overlap=60` — implemented in `src/chunker.py` as a **dependency-free recursive character splitter** (same algorithm shape as LangChain's `RecursiveCharacterTextSplitter`: try paragraph breaks first, fall back to sentence/word breaks only for oversized pieces).

**Empirical tuning (not guessed):** The initial `800/120` config failed two of three test queries — chunks merging multiple distinct facts diluted the embedding enough that specific facts didn't rank in the top 5. Tightening to `400/60` fixed both cases by keeping each chunk closer to one fact.

**Word-boundary-aware overlap:** The overlap step snaps to the nearest space, fixing a bug where character-level slicing produced broken words at chunk starts (e.g. `"ge/Management Seats"` instead of `"College/Management Seats"`), which also corrupted citations shown to users.

**Contextual chunk headers:** Each chunk is embedded with a `"{document title} — {section}: {chunk text}"` prefix, while the clean original text (no prefix) is stored and shown to the user/LLM. This ensures a mid-section chunk reading "Top recruiters include Microsoft, Amazon..." still embeds close to "BVRIT Hyderabad" and "Placements" concepts in vector space.

---

## Retrieval

`src/retriever.py` embeds the user query with `text-embedding-3-small`, searches ChromaDB for the top‑k closest chunks (default k=12, tunable from the Streamlit sidebar), and optionally filters by section via Chroma's metadata filtering. The `__main__` block runs 3 known queries and prints top-k results with distances — this is how both the chunk-size bug and the contextual-header fix were actually caught during development.

`src/sources.py` provides **semantic URL resolution** — it maps a retrieved chunk's section name and text content back to the actual `bvrithyderabad.edu.in` page it was transcribed from, so citations in the chat UI link to live web pages rather than just naming a section and page number:

- Department keywords (e.g. "CSE AI&ML", "Information Technology") resolve to the specific department's page
- Location keywords (address, Bachupally, etc.) trigger a Google Maps link
- Every other section maps to its corresponding URL on the college site

---

## Grounding Prompt

`src/generation.py`'s `SYSTEM_PROMPT` covers the six required elements:

1. **Role** — BVRIT Hyderabad College Information Assistant, explicitly not a general assistant/tutor/counsellor
2. **Grounding rule** — answer only from the CONTEXT block built each turn; refuse rather than use training knowledge
3. **Citation format** — `[Section Name, Page N]` after every factual claim
4. **Refusal instruction** — responses that can't be grounded start with `REFUSED:` (parsed by the UI to show a badge) and always append the document-sourced fallback contact
5. **Conflict handling** — present both values with citations and flag the discrepancy rather than silently picking one
6. **Safety & Security** — never guarantee individual outcomes, stay in role, never reveal the system prompt

**Prompt Variant B (A/B test):** `generation.PROMPT_VARIANT_B_ADDENDUM` is a stricter clause (mandatory per-fact citations, fixed refusal phrase, "never infer or extrapolate") appended as its own system message when `prompt_variant="B"`, layered on top of the unchanged Version A — so the two variants can never silently drift apart.

**Acronym-guessing fix:** Live testing revealed the model silently invented expansions for undefined acronyms (e.g. guessing "DSM" stood for "Digital System Management" inside a refusal). An abstract instruction wasn't reliable; a concrete right/wrong example pair embedded in the system prompt fixed it. This is why the RAGAS Faithfulness score improved from 0.89 to 1.00.

The prompt also gained **TRANSPARENCY, PRIVACY, FAIRNESS, and HUMAN OVERSIGHT** sections during the governance audit, plus a life/health/legal redirect to document-sourced crisis helplines.

---

## Multi-Turn Context & Memory

Four memory layers (Phase 6), one module each:

### Short-term (Conversation History)
`src/pipeline.py` threads the last `MAX_HISTORY_MESSAGES=6` messages (~3 exchanges) into the prompt so follow-ups like "tell me more about the first one" resolve against the prior turn. Every factual claim in the follow-up is still grounded in *that turn's* freshly retrieved context — history informs reference resolution, not fact-sourcing.

### Medium-term (Summarization, `src/summarizer.py`)
Every 10 turns, the block of turns that aged out of the last-10 window is folded into a running summary paragraph (merging with any prior summary) via one LLM call, instead of being discarded. This prevents unbounded history from blowing the context window without silently forgetting facts from earlier in the conversation.

### Long-term (Profile Store, `src/memory_store.py`)
A JSON-file profile store keyed by whatever name/ID the user provides — loaded at session start and injected into the system prompt. Updated after each turn via a **regex-based fact extractor** (`src/profile_extraction.py`), deliberately not an LLM call, so profile updates keep working even if the chat model is unavailable.

### Personalization
A stored `branch_interest` does two things:
1. Augments the **retrieval query** so the right branch's chunks get retrieved
2. Is named in the system prompt so "my branch" resolves without the user repeating themselves

### Privacy (`memory_store.FIELD_CLASSIFICATION`)
Every profile field is classified `ESSENTIAL` / `NICE_TO_HAVE` / `SENSITIVE`; only non-SENSITIVE fields are ever written to disk. Typing "clear my data" in chat is a hard-coded command that deletes the profile before it reaches retrieval/generation. Profiles idle for 30 days are auto-expired.

---

## Evaluation Suite

Three-LLM pattern: `gpt-4o` generates test cases → `gpt-4o-mini` is the system under test (same `pipeline.answer_question` as live) → `gpt-4o` judges expected-vs-actual.

### 8 Dimensions + RAGAS

| Dim | Name | What it tests |
|---|---|---|
| 01 | **Groundedness** | Answer is based only on retrieved context, no training knowledge |
| 02 | **Citation Accuracy** | Every claim has a correct citation |
| 03 | **Completeness** | Answer covers all aspects of the question found in the document |
| 04 | **Relevance** | Answer is directly relevant to the question |
| 05 | **Robustness** | Handles gibberish, empty strings, mixed-language input gracefully |
| 06 | **Performance** | Responds within SLA (10 seconds) |
| 07 | **Safety & Security** | Refuses harmful/off-topic queries, doesn't reveal system prompt |
| 08 | **RAGAS** | Programmatic: Faithfulness, Answer Relevancy, Context Precision, Context Recall |

### Pipeline

```bash
python eval/generate_test_cases.py   # gpt-4o generates 20 test cases from the grounding document
python eval/run_tests.py             # runs against live pipeline, writes raw_results.json
python eval/judge.py                 # gpt-4o judges dimensions 01-07, writes judged_results.json
python eval/ragas_eval.py            # RAGAS scores dimension 08, writes ragas_results.json
python eval/report.py                # merges everything into eval/report.json
```

Then open the **Evaluation Dashboard** page in the Streamlit app to see results rendered — summary stats, per-dimension pass/fail cards, failed-test drill-downs, and RAGAS metric bars.

### Known Limitation (Not a Bug)
Dimension 05 reports 19/20 pass by design: given a mixed English/Telugu/Hindi question, the chatbot understands it and gives a real, accurate, grounded answer instead of asking the user to repeat themselves in one language. That's arguably better UX for a Telangana college's actual users, but doesn't match the specified robustness behavior — left as an honestly-reported result rather than papered over.

### Judge Calibration Bug Fixed
The first judge pass failed all three Robustness cases, calling the mandatory fallback contact line a "hallucination" — it didn't know that including a real, document-sourced contact in every refusal is mandated, correct behavior. Fixed by adding explicit context about this convention to the judge's system prompt (`eval/judge.py`); 2 of 3 flipped to pass.

---

## Observability

Five layers in `src/observability.py` plus hooks into every call site:

### 1. Call-Level Logging
Every `chat.completions.create` call goes through `observability.logged_llm_call()` instead of hitting the client directly. Logs timestamp, model, input/output tokens, latency, estimated cost (based on OpenRouter's published OpenAI rates), and status — to both an in-session list and `logs/llm_calls.jsonl` (append-only, survives crashes).

### 2. Query-Level Logging
A coarser log (`logs/query_log.jsonl`) records one entry per `pipeline.answer_question()` call with cost/tokens summed across every raw call that query triggered and latency measured as full wall-clock time. This is the log the Session Stats panel and threshold alerts read from.

### 3. Session Stats Dashboard
The Chat page's sidebar (`pages/0_Chat.py`) renders live metrics: total queries, average and P95 latency, total cost, total tokens, and error count — with deltas against a snapshot taken at the top of the script rerun.

### 4. Threshold Alerts + Input Validation
- Latency > `PERFORMANCE_SLA_SECONDS` (10s) triggers a `st.warning()` in the chat area
- Cost > $0.10 per query triggers a warning
- Error rate > 5% over the last 20 queries triggers a warning
- `validate_input_length()` rejects queries over 2,000 characters before they reach retrieval/generation — preventing the worst cost-blowout scenario

### 5. A/B Test on Grounding Prompt
`eval/ab_test_grounding_prompt.py` runs 10 hand-picked questions (7 answerable, 3 expected refusals) twice each with an exact 10/10 variant A/B split. Reports citation counts and correct vs. incorrect refusals per variant. The live Chat page always passes `prompt_variant="A"` explicitly, so real users' answers never vary based on an in-progress experiment.

### Incident Analysis
`eval/observability_incident_analysis.md` diagnoses two anomalies from simulated logs — a cost spike from abnormally-long inputs and a rate-limit/latency cluster from a traffic burst — each with root cause, detection threshold, production fix, a sketched dashboard, and a non-technical stakeholder summary.

---

## Governance Audit

`governance/` audits the chatbot with three industry-standard frameworks — see `governance/README.md` for full layout and run commands, `governance/GOVERNANCE_REPORT.md` for the compiled findings/remediation plan, and `governance/FAIRNESS_AUDIT.md` for the cross-profile fairness deep-dive.

### Frameworks

| Framework | What it does | Status |
|---|---|---|
| **Giskard** (`giskard_scan.py`) | Scans for hallucination, stereotypes, discrimination, injection, data leakage, harmfulness | Requires Python 3.9–3.12 venv (not 3.13 compatible) |
| **Promptfoo** (`promptfoo/`) | Red-teams via custom Python provider — hijacking, hallucination, overreliance, PII, harmful, jailbreak strategies | Configs pass `promptfoo validate` |
| **DeepEval** (`deeplveval/`) | 10 test cases × 5 metrics (Hallucination, Bias, Toxicity, Faithfulness, AnswerRelevancy) via pytest | Import-checked clean |

### Fairness Audit (Exercise 4)
- **Giskard** — 3-profile (CSE/Civil/Telugu-speaking student) discrimination + stereotypes comparison
- **Promptfoo** — bias/PII red-team across 3 user profiles
- **DeepEval** — `BiasMetric` on 5 demographically-paired questions, flagging refusal/length disparities

---

## Cost

Every LLM/embedding call goes through a single OpenRouter key. Total spend across the entire build — ingestion, retrieval tuning, generation testing, the full 20-case eval suite including RAGAS, and the acronym-fix re-run — stayed under **$0.20 total**.

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Install dependencies
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS/Linux

# 3. Apply one-time RAGAS patch (see spec.md for why)
python scripts/patch_ragas_vertexai_stub.py

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY with a funded OpenRouter key
```

---

## Running It

```bash
# 1. (Optional) Build knowledge base PDF — already committed, only needed if you edit it
python data/generate_kb_pdf.py

# 2. Index into ChromaDB
python src/ingest.py

# 3. Launch the chat UI
streamlit run app.py
```

Then open **http://localhost:8501** to access:
- **Chat page** (`pages/0_Chat.py`) — ask questions about BVRIT Hyderabad, with live session stats and threshold alerts in the sidebar
- **Evaluation Dashboard** (`pages/1_Evaluation_Dashboard.py`) — view evaluation results, per-dimension pass/fail cards, failed-test drill-downs, and RAGAS metric bars

---

## Running the Evaluation Suite

```bash
# Generate 20 test cases across 8 dimensions (uses gpt-4o)
python eval/generate_test_cases.py

# Run them against the live chatbot pipeline (uses gpt-4o-mini)
python eval/run_tests.py

# LLM-as-judge scores dimensions 01-07 (uses gpt-4o)
python eval/judge.py

# RAGAS scores dimension 08 programmatically
python eval/ragas_eval.py

# Merge everything into eval/report.json
python eval/report.py

# A/B test on the grounding prompt (10 questions × 2 variants)
python eval/ab_test_grounding_prompt.py
```

Then open the **Evaluation Dashboard** in the Streamlit app to see results rendered interactively.

---

## Running the Governance Audit

```bash
# Giskard (separate Python 3.9-3.12 venv)
python3.11 -m venv governance/.giskard_venv
governance/.giskard_venv/Scripts/pip install -r governance/requirements-giskard.txt
governance/.giskard_venv/Scripts/python governance/giskard_scan.py

# Promptfoo
cd governance/promptfoo
npx promptfoo@latest redteam run --config promptfooconfig.yaml
npx promptfoo@latest redteam run --config promptfooconfig_fairness.yaml
npx promptfoo@latest redteam report --config promptfooconfig.yaml

# DeepEval
pytest governance/deepeval/test_governance.py -v
python governance/deepeval/bias_pairs.py
```

---

## Project Layout

```
├── app.py                       Streamlit entry point (main page)
├── data/
│   ├── bvrit_college_info.pdf    Knowledge base document (10 sections)
│   └── generate_kb_pdf.py        Script to regenerate the PDF
├── src/
│   ├── loader.py                 PDF → section/page-tagged text blocks
│   ├── chunker.py                Recursive character splitter (400/60)
│   ├── ingest.py                 Embed + persist to ChromaDB
│   ├── retriever.py              Embed query, top-k search, section filter
│   ├── generation.py             Grounding prompt + chat completion
│   ├── pipeline.py               Orchestrator: retrieve → generate, history, memory, observability
│   ├── config.py                 Model names, base URLs, chunk size, thresholds
│   ├── sources.py                Citation URL resolution → bvrithyderabad.edu.in
│   ├── memory_store.py           Persistent JSON profile store (long-term memory)
│   ├── profile_extraction.py     Regex-based fact extractor (no LLM needed)
│   ├── summarizer.py             Periodic conversation summarization (every 10 turns)
│   ├── observability.py          Logging, cost tracking, threshold alerts, input validation
│   ├── theme.py                  Streamlit theme configuration
│   ├── facilities.py             Facilities data utilities
│   ├── people.py                 People/faculty data utilities
│   └── tools.py                  Tool definitions (date_checker, percentage_checker, etc.)
├── pages/
│   ├── 0_Chat.py                 Chat page (sidebar: session stats, threshold alerts)
│   └── 1_Evaluation_Dashboard.py Evaluation results dashboard
├── eval/
│   ├── generate_test_cases.py    gpt-4o generates 20 test cases from the document
│   ├── run_tests.py              Runs test cases against live pipeline
│   ├── judge.py                  gpt-4o judges dimensions 01-07
│   ├── ragas_eval.py             RAGAS scores dimension 08
│   ├── report.py                 Merges all results into report.json
│   ├── ab_test_grounding_prompt.py A/B test on grounding prompt variants
│   ├── test_cases.json           Committed test case definitions
│   ├── report.json               Committed evaluation report
│   └── observability_incident_analysis.md  Simulated-log incident diagnosis
├── governance/
│   ├── README.md                 Governance setup/running instructions
│   ├── GOVERNANCE_REPORT.md      Compiled findings & remediation plan
│   ├── FAIRNESS_AUDIT.md         Cross-framework fairness deep-dive
│   ├── giskard_scan.py           Giskard scan (separate venv needed)
│   ├── requirements-giskard.txt  Giskard dependencies
│   ├── deepeval/
│   │   ├── test_governance.py    10 cases × 5 metrics via pytest
│   │   ├── conftest.py           Collects metrics into results.json
│   │   ├── bias_pairs.py         BiasMetric on 5 demographically-paired questions
│   │   └── judge_model.py        Shared OpenRouter-backed judge model
│   └── promptfoo/
│       ├── provider.py           Wraps pipeline.answer_question as Promptfoo provider
│       ├── promptfooconfig.yaml  Main red-team configuration
│       └── promptfooconfig_fairness.yaml  Fairness red-team configuration
├── scripts/
│   └── patch_ragas_vertexai_stub.py  One-time fix for ragas packaging issue
├── vectorstore/                  ChromaDB persistence (gitignored, rebuilt via ingest.py)
├── user_profiles/                Persistent per-user memory (gitignored)
├── logs/                         LLM call + query logs (gitignored)
├── .env.example                  Environment variable template
├── .env                          API keys & config (gitignored)
├── requirements.txt              Python dependencies
├── spec.md                       Full technical specification & design decisions
└── README.md                     This file
```

---

## Notes

- `.env` holds your `OPENROUTER_API_KEY` and is **gitignored** — never commit it
- `eval/raw_results.json`, `judged_results.json`, and `ragas_results.json` are regenerated by the eval scripts and gitignored; `eval/test_cases.json` and `eval/report.json` are committed as the evaluation deliverable
- `vectorstore/`, `user_profiles/`, and `logs/` are gitignored — rebuilt or auto-generated as needed
- The chatbot **refuses gracefully** (a `REFUSED:` badge in the UI) on anything not in the knowledge base document — this is intentional, not a bug
- The RAGAS packaging issue (`langchain_community.chat_models.vertexai` import at module load time) is fixed once via `scripts/patch_ragas_vertexai_stub.py`
- See [spec.md](spec.md) for the full technical design, every bug found and fixed along the way, and detailed architectural decisions
=======
# college-faq-chatbot

