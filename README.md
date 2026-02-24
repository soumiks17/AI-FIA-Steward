---
title: AI FIA Steward
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---

# 🏎️ AI FIA Steward

An AI-powered Formula 1 steward that predicts penalties for on-track incidents by reasoning against real historical FIA decisions.

**Live Demo:** [huggingface.co/spaces/soumiks17/ai-fia-steward](https://huggingface.co/spaces/soumiks17/ai-fia-steward)

---

## Workflow

```
Describe incident → Semantic search → Retrieve precedents → LLM ruling
```

1. User describes an on-track incident.
2. The query is embedded and searched against the vector database.
3. The 3 most semantically similar historical FIA decisions are retrieved.
4. GPT-4o-mini reads the user's query and predicts the penalty that might be imposed on the driver or the team.


---

## Data Scraping

The FIA publishes all steward decisions as PDFs on [fia.com](https://www.fia.com/documents). The site is a JavaScript single-page app, so standard scraping doesn't work. Instead:

- Each season has a list of race weekends, each linked to a **Drupal node ID**
- The scraper hits `/decision-document-list/ajax/{node_id}` — an internal AJAX endpoint that returns raw HTML embedded in a JSON response
- PDF links are extracted from that HTML, filtered by keywords (`decision`, `infringement`, `offence`)
- PDFs are downloaded and stored locally per season/event

Seasons covered: **2019 – 2025**

---

## PDF Parsing

Each downloaded PDF is parsed to extract structured fields:

- **Driver** — who was investigated
- **Breach** — which regulation was allegedly violated
- **Decision** — the penalty or outcome issued
- **Reasoning** — the stewards' full written justification

---

## Embeddings & Vector DB

- The **reasoning text** from each decision is embedded using OpenAI's `text-embedding-3-small` model
- Embeddings are stored in **ChromaDB** with metadata (driver, breach, decision, source PDF)
- At query time, the user's incident description is embedded with the same model and a **cosine similarity search** retrieves the top 3 most relevant precedents
- A **relevance threshold of 1.1** rejects queries that aren't F1-related — unrelated topics score significantly higher and never reach the LLM

---

## Stack

`Gradio` · `LangChain` · `ChromaDB`  · `text-embedding-3-small` · `BeautifulSoup` · `PyMuPDF`