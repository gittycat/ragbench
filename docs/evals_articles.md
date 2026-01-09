# RAG Evaluation – Priority Watchlist (YouTube)

Ordered to build **conceptual understanding first**, then **evaluation methodology**, then **operational / benchmark design**, and finally **practical grounding**.

---

## Tier 1 — Must-watch (conceptual foundation)

### 1. RAG Evaluation Metrics Explained: Context Precision, Recall, Relevancy & Faithfulness  
https://www.youtube.com/watch?v=wOoYP55eYF0  

**Why watch first**
- Best single video for understanding *what RAG eval metrics actually measure*
- Clean separation of failure modes:
  - retrieval (precision / recall)
  - grounding (faithfulness)
  - answer quality (relevancy / correctness)
- Tool-agnostic, minimal fluff

**Maps to**
- recall@k, precision@k  
- faithfulness  
- answer_relevancy / answer_correctness

---

### 2. How to Evaluate a RAG System: Methods and Metrics  
https://www.youtube.com/watch?v=qI2qQfOG0Js  

**Why watch second**
- Moves from metrics → **evaluation methodology**
- Explains:
  - offline benchmarks
  - reference vs reference-free (LLM-as-judge)
  - why composite / weighted scores exist

**Maps to**
- multi-dataset evaluation
- weighted scorecards
- judge vs non-judge metrics

---

## Tier 2 — Strongly recommended (operational & benchmark design)

### 3. Evaluating Retrieval-Augmented Generation (RAG) Models (Webinar)  
https://www.youtube.com/watch?v=ia-7WllGoaM  

**Why**
- Deep dive on **LLM-as-judge reliability**
- Covers:
  - judge calibration
  - prompt sensitivity
  - reproducibility and variance

**Maps to**
- faithfulness
- answer correctness
- citation quality (conceptually)
- trust in judge-based scores

---

### 4. Beginner’s Guide to RAG Evaluation (Langfuse / industry webinar)  
https://www.youtube.com/watch?v=Y0X3K9J1hJ8  

**Why**
- Strong operational framing:
  - offline eval vs production monitoring
  - latency / cost trade-offs
  - how to communicate evals to PMs and end users

**Maps to**
- latency p50 / p95
- cost objectives
- reporting & dashboards

---

## Tier 3 — Optional / practical grounding

### 5. Evaluating Your RAG Applications (MongoDB / RAGAS walkthrough)  
https://www.youtube.com/watch?v=mzWwxmyEyWE  

**Why**
- Concrete end-to-end example
- Helps anchor abstract metrics to real pipelines and outputs

**Maps to**
- retrieval + generation metrics
- practical implementation details

---

## Suggested onboarding flow

1. Watch **Tier 1 (#1, #2)** back-to-back  
2. Read the internal RAG evaluation framework doc  
3. Watch **Tier 2 (#3 or #4)** depending on role:
   - infra / ML → #3  
   - product / systems → #4  
4. Watch **Tier 3 (#5)** only if implementation context is needed

This order reliably builds intuition for *what each eval measures*, *why it exists*, and *how to reason about trade-offs*.
