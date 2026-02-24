import os
import re
import gradio as gr
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import snapshot_download

load_dotenv()
if not os.path.exists("./fia_pdfs"):
    print("Downloading PDFs...")
    snapshot_download(
        repo_id="soumiks17/FIA-PDFs",
        repo_type="dataset",
        local_dir="./fia_pdfs"
    )
    print("PDFs ready.")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="./fia_chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

RELEVANCE_THRESHOLD = 1.1

template = """
You are the Chief FIA Steward. Analyze the user's incident based strictly on the provided historical precedents.
State the predicted penalty and your reasoning.

Historical Precedents:
{context}

Incident:
{query}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "query"])
chain = prompt | llm

NO_PRECEDENT_MSG = "⚠  No relevant F1 precedents found. Please describe an incident that could occur in a Formula 1 race."


def format_event_name(filepath):
    match = re.search(r'(\d{4})[\\/]([^\\/]+)', filepath)
    if match:
        year = match.group(1)
        event = match.group(2).replace('_', ' ')
        return f"{year} — {event}"
    return filepath


def is_relevant(docs_with_scores):
    if not docs_with_scores:
        return False
    return any(score <= RELEVANCE_THRESHOLD for _, score in docs_with_scores)


def clean_cell(text):
    return text.replace('\n', ' ').replace('\r', '').strip()


def build_context(docs_with_scores):
    context_for_llm = ""
    pdf_files = []
    html_rows = ""

    S_td   = "padding:0.7rem 0.9rem;vertical-align:top;border-bottom:1px solid #2a2a2a;word-wrap:break-word;overflow-wrap:break-word;white-space:normal;line-height:1.55;font-size:0.82rem;font-family:'DM Sans',sans-serif;"
    S_num  = S_td + "color:#e8002d;font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;white-space:nowrap;width:4%;"
    S_evt  = S_td + "color:#ffffff;font-weight:500;width:17%;"
    S_drv  = S_td + "color:#cccccc;width:12%;"
    S_brch = S_td + "color:#bbbbbb;width:30%;"
    S_dec  = S_td + "color:#a8d8a0;font-weight:600;width:37%;"

    for i, (doc, score) in enumerate(docs_with_scores):
        if score > RELEVANCE_THRESHOLD:
            continue

        driver     = doc.metadata.get('driver',   'Unknown')
        breach     = doc.metadata.get('breach',   'Unknown')
        decision   = doc.metadata.get('decision', 'Unknown')
        raw_source = doc.metadata.get('source',   'Unknown')
        clean_source = format_event_name(raw_source)

        abs_path = os.path.abspath(raw_source)
        if os.path.exists(abs_path):
            pdf_files.append(abs_path)

        context_for_llm += (
            f"\nPrecedent {i+1}:\n"
            f"Event: {clean_source}\n"
            f"Driver: {driver}\n"
            f"Breach: {breach}\n"
            f"Decision: {decision}\n"
            f"Reasoning: {doc.page_content}\n"
        )

        html_rows += f"""
        <tr>
            <td style="{S_num}">#{i+1}</td>
            <td style="{S_evt}">{clean_source}</td>
            <td style="{S_drv}">{driver}</td>
            <td style="{S_brch}">{breach}</td>
            <td style="{S_dec}">{decision}</td>
        </tr>"""

    if html_rows:
        S_th = "padding:0.6rem 0.9rem;text-align:left;font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#e8002d;white-space:nowrap;"
        html_table = f"""
<div style="width:100%;overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed;border-top:2px solid #e8002d;border-bottom:2px solid #e8002d;background:#141414;">
    <thead>
      <tr style="background:#1a1a1a;">
        <th style="{S_th}width:4%;">NO.</th>
        <th style="{S_th}width:17%;">EVENT</th>
        <th style="{S_th}width:12%;">DRIVER</th>
        <th style="{S_th}width:30%;">BREACH</th>
        <th style="{S_th}width:37%;">DECISION</th>
      </tr>
    </thead>
    <tbody>{html_rows}
    </tbody>
  </table>
</div>"""
    else:
        html_table = ""

    return context_for_llm, html_table, pdf_files


def rule_on_incident(incident_description):
    if not incident_description or not incident_description.strip():
        return "Please provide an incident description.", "", []

    docs_with_scores = db.similarity_search_with_score(incident_description, k=3)

    print("\n--- SCORES ---")
    for doc, score in docs_with_scores:
        print(f"  score={score:.4f} | breach={doc.metadata.get('breach', 'N/A')}")
    print(f"  threshold={RELEVANCE_THRESHOLD}")
    print("--------------\n")

    if not is_relevant(docs_with_scores):
        return NO_PRECEDENT_MSG, "", []

    context_for_llm, html_table, pdf_files = build_context(docs_with_scores)

    if not context_for_llm:
        return NO_PRECEDENT_MSG, "", []

    response = chain.invoke({"context": context_for_llm, "query": incident_description})
    return response.content, html_table, pdf_files




class TestRelevance:
    def test_empty_input_returns_prompt(self):
        ruling, precedents, files = rule_on_incident("")
        assert "Please provide" in ruling
        assert precedents == ""
        assert files == []

    def test_whitespace_input_returns_prompt(self):
        ruling, precedents, files = rule_on_incident("   ")
        assert "Please provide" in ruling

    def test_unrelated_topic_cooking(self):
        ruling, precedents, files = rule_on_incident(
            "The chef added too much salt to the pasta and the dish was ruined."
        )
        assert ruling == NO_PRECEDENT_MSG
        assert files == []

    def test_unrelated_topic_football(self):
        ruling, precedents, files = rule_on_incident(
            "A footballer deliberately handled the ball in the penalty area during a corner kick."
        )
        assert ruling == NO_PRECEDENT_MSG
        assert files == []

    def test_unrelated_topic_weather(self):
        ruling, precedents, files = rule_on_incident(
            "A hurricane made landfall causing widespread flooding and damage."
        )
        assert ruling == NO_PRECEDENT_MSG
        assert files == []

    def test_unrelated_topic_finance(self):
        ruling, precedents, files = rule_on_incident(
            "The trader executed a short sell on the stock before insider information was released."
        )
        assert ruling == NO_PRECEDENT_MSG
        assert files == []

    def test_f1_collision_returns_ruling(self):
        ruling, precedents, files = rule_on_incident(
            "Car 1 attempted an overtake at the chicane, made contact with Car 2 and caused a spin."
        )
        assert ruling != NO_PRECEDENT_MSG
        assert len(ruling) > 50

    def test_f1_unsafe_release_returns_ruling(self):
        ruling, precedents, files = rule_on_incident(
            "A car was released from the pit box into the path of another car causing a collision."
        )
        assert ruling != NO_PRECEDENT_MSG
        assert len(ruling) > 50

    def test_f1_track_limits_returns_ruling(self):
        ruling, precedents, files = rule_on_incident(
            "The driver repeatedly exceeded track limits at Turn 4 gaining a lasting advantage."
        )
        assert ruling != NO_PRECEDENT_MSG
        assert len(ruling) > 50

    def test_f1_pit_speeding_returns_ruling(self):
        ruling, precedents, files = rule_on_incident(
            "Car 44 was recorded at 95 km/h in the pit lane, exceeding the 80 km/h speed limit."
        )
        assert ruling != NO_PRECEDENT_MSG
        assert len(ruling) > 50

    def test_f1_false_start_returns_ruling(self):
        ruling, precedents, files = rule_on_incident(
            "The driver moved before the lights went out at the start of the race."
        )
        assert ruling != NO_PRECEDENT_MSG
        assert len(ruling) > 50

    def test_ruling_returns_three_outputs(self):
        ruling, precedents, files = rule_on_incident(
            "Car 33 weaved multiple times under braking to defend position."
        )
        assert isinstance(ruling, str)
        assert isinstance(precedents, str)
        assert isinstance(files, list)

    def test_precedents_contains_ascii_table(self):
        ruling, precedents, files = rule_on_incident(
            "Car 16 caused a collision at the start by braking too late."
        )
        if ruling != NO_PRECEDENT_MSG:
            assert "EVENT" in precedents
            assert "<table" in precedents


class TestFormatEventName:
    def test_windows_path(self):
        result = format_event_name("fia_pdfs\\2024\\Monaco_Grand_Prix")
        assert "2024" in result
        assert "Monaco Grand Prix" in result

    def test_unix_path(self):
        result = format_event_name("fia_pdfs/2024/Monaco_Grand_Prix")
        assert "2024" in result
        assert "Monaco Grand Prix" in result

    def test_no_match_returns_original(self):
        result = format_event_name("unknown_path")
        assert result == "unknown_path"




css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --red: #e8002d;
    --dark: #0a0a0a;
    --mid: #141414;
    --border: #2a2a2a;
    --text: #e8e8e8;
    --muted: #666;
}

body, .gradio-container {
    background: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

#header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}

#header h1 {
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.1em;
    color: #fff !important;
    margin: 0 !important;
}

#header h1 span {
    color: var(--red);
}

#header p {
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 0.5rem 0 0 0;
}

textarea, input {
    background: var(--mid) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 4px !important;
}

textarea:focus, input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 2px rgba(232, 0, 45, 0.15) !important;
}

button.primary {
    background: var(--red) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 2px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
    height: 48px !important;
    max-height: 48px !important;
    flex-shrink: 0 !important;
}

button.primary:hover {
    opacity: 0.85 !important;
}

.label-wrap span {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

.markdown-body, .prose {
    background: transparent !important;
    color: var(--text) !important;
}

.prec-wrap {
    width: 100%;
    overflow-x: auto;
}

.prec-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    font-family: 'DM Sans', sans-serif;
    color: #e8e8e8;
    table-layout: fixed;
    border-bottom: 2px solid #e8002d;
}

.prec-table thead tr {
    background: #1a1a1a;
    border-bottom: 2px solid #e8002d;
}

.prec-table th {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #e8002d;
    padding: 0.6rem 0.8rem;
    text-align: left;
    white-space: nowrap;
}

.prec-table td {
    padding: 0.7rem 0.8rem;
    vertical-align: top;
    border-bottom: 1px solid #2a2a2a;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: normal;
    line-height: 1.5;
}

.prec-table tbody tr:hover {
    background: #1c1c1c;
}

.prec-num {
    color: #e8002d;
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    width: 3%;
    white-space: nowrap;
}

.prec-event  { width: 16%; color: #fff; font-weight: 500; }
.prec-driver { width: 10%; color: #ccc; }
.prec-breach { width: 18%; color: #bbb; }
.prec-decision { width: 14%; color: #a8d8a0; font-weight: 600; }
.prec-reasoning { width: 39%; color: #999; font-style: italic; }

/* Ruling box */
#ruling-box .prose p, #ruling-box .markdown-body p {
    line-height: 1.8 !important;
}

.section-divider {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--red);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-top: 1.8rem;
    margin-bottom: 0.75rem;
}

.ruling-panel {
    background: var(--mid);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    border-radius: 3px;
    padding: 1.25rem 1.5rem;
    line-height: 1.75;
}

.file-preview {
    background: var(--mid) !important;
    border: 1px solid var(--border) !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=css) as demo:

    with gr.Column(elem_id="header"):
        gr.HTML("""
            <h1>AI <span>FIA</span> STEWARD</h1>
            <p>Objective compliance engine — historical racing precedents</p>
        """)

   
    incident_input = gr.Textbox(
        lines=5,
        label="Incident Description",
        placeholder=(
            "Describe the on-track incident in detail.\n\n"
            "Example: Car 16 attempted a late-braking move into Turn 1, "
            "made contact with Car 55's rear tyre, and caused a spin into "
            "the barriers. Car 16 continued with front wing damage."
        )
    )
    submit_btn = gr.Button("Analyze Precedents", variant="primary")

   
    gr.HTML('<div class="section-divider">⚑ &nbsp;Steward\'s Ruling</div>')
    ruling_output = gr.Markdown(elem_classes=["ruling-panel"])

   
    gr.HTML('<div class="section-divider">📋 &nbsp;Cited Precedents</div>')
    precedents_output = gr.HTML()

    gr.HTML('<div class="section-divider">📄 &nbsp;Source Documents</div>')
    documents_output = gr.File(label="")

    submit_btn.click(
        fn=rule_on_incident,
        inputs=incident_input,
        outputs=[ruling_output, precedents_output, documents_output]
    )

if __name__ == "__main__":
    demo.launch()