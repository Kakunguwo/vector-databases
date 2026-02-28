import io
import importlib.util
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

st.set_page_config(page_title="Vector Demo UI", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "vector-demo.py"

SECTIONS = [
    ("Scenario 1: English Baseline", "scenario_1_english_baseline"),
    ("Scenario 2: Shona Direct Embedding", "scenario_2_shona_direct"),
    ("Scenario 3: Shona with Translation", "scenario_3_shona_translated"),
    ("Scenario 4: Bilingual Database", "scenario_4_bilingual"),
    ("Final Report", "generate_report"),
]


@st.cache_resource
def load_demo_module():
    spec = importlib.util.spec_from_file_location("vector_demo_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Unable to load vector-demo.py")
    spec.loader.exec_module(module)
    return module


def capture_output(callable_obj):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        callable_obj()
    return buffer.getvalue().strip()


def capture_output_with_result(callable_obj):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = callable_obj()
    return result, buffer.getvalue().strip()


def initialize_state():
    if "demo" not in st.session_state:
        st.session_state.demo = None
    if "logs" not in st.session_state:
        st.session_state.logs = {}
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "vector_workspace" not in st.session_state:
        st.session_state.vector_workspace = None
    if "sim_sentences" not in st.session_state:
        st.session_state.sim_sentences = []
    if "sim_index" not in st.session_state:
        st.session_state.sim_index = 0
    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_last_workspace" not in st.session_state:
        st.session_state.sim_last_workspace = None


def initialize_demo(module):
    demo_instance, output = capture_output_with_result(module.ProductionDemo)
    st.session_state.demo = demo_instance
    st.session_state.logs = {"Initialization": output}
    st.session_state.current_step = 0


def run_step(section_name, method_name):
    method = getattr(st.session_state.demo, method_name)
    output = capture_output(method)
    st.session_state.logs[section_name] = output


def reset_demo():
    st.session_state.demo = None
    st.session_state.logs = {}
    st.session_state.current_step = 0
    st.session_state.vector_workspace = None
    st.session_state.sim_sentences = []
    st.session_state.sim_index = 0
    st.session_state.sim_history = []
    st.session_state.sim_last_workspace = None


def parse_documents(raw_text: str) -> List[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def split_sentences(raw_text: str) -> List[str]:
    cleaned = " ".join(raw_text.strip().split())
    if not cleaned:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def project_to_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    if vt.shape[0] >= 2:
        basis = vt[:2].T
        return centered @ basis

    x = centered[:, :1]
    y = np.zeros_like(x)
    return np.hstack([x, y])


def cosine_scores(doc_vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denominator = np.maximum(doc_norms * query_norm, 1e-12)
    return (doc_vectors @ query_vector) / denominator


def build_vector_workspace(
    demo,
    documents: List[str],
    query_text: str,
    threshold: float,
    target_cosine: float,
) -> Dict:
    doc_vectors = np.array(demo.embedder.encode_batch(documents))
    query_vector = np.array(demo.embedder.encode(query_text))

    scores = cosine_scores(doc_vectors, query_vector)
    picked_index = int(np.argmin(np.abs(scores - target_cosine)))
    selected_indices = [int(i) for i, score in enumerate(scores) if score >= threshold]

    combined = np.vstack([doc_vectors, query_vector])
    coords = project_to_2d(combined)
    doc_coords = coords[:-1]
    query_coord = coords[-1]

    points = []
    for index, (coord, score, text) in enumerate(zip(doc_coords, scores, documents)):
        if index == picked_index:
            kind = "Picked"
        elif score >= threshold:
            kind = "Selected"
        else:
            kind = "Not selected"

        points.append(
            {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "label": f"Doc {index + 1}",
                "kind": kind,
                "cosine": float(score),
                "text": text,
            }
        )

    points.append(
        {
            "x": float(query_coord[0]),
            "y": float(query_coord[1]),
            "label": "Query",
            "kind": "Query",
            "cosine": 1.0,
            "text": query_text,
        }
    )

    ranked = sorted(
        [
            {
                "Rank": i + 1,
                "Document": documents[idx],
                "Cosine": float(scores[idx]),
            }
            for i, idx in enumerate(np.argsort(scores)[::-1])
        ],
        key=lambda row: row["Cosine"],
        reverse=True,
    )

    return {
        "documents": documents,
        "query_text": query_text,
        "doc_vectors": doc_vectors,
        "query_vector": query_vector,
        "scores": scores,
        "picked_index": picked_index,
        "selected_indices": selected_indices,
        "threshold": threshold,
        "target_cosine": target_cosine,
        "points": points,
        "ranked": ranked,
    }


def start_sentence_simulation(incoming_text: str):
    st.session_state.sim_sentences = split_sentences(incoming_text)
    st.session_state.sim_index = 0
    st.session_state.sim_history = []
    st.session_state.sim_last_workspace = None


def process_sentence_step(demo, documents: List[str], threshold: float, target_cosine: float):
    if st.session_state.sim_index >= len(st.session_state.sim_sentences):
        return

    sentence = st.session_state.sim_sentences[st.session_state.sim_index]
    workspace = build_vector_workspace(
        demo,
        documents,
        sentence,
        threshold,
        target_cosine,
    )

    picked_index = workspace["picked_index"]
    picked_doc = workspace["documents"][picked_index]
    picked_score = float(workspace["scores"][picked_index])

    st.session_state.sim_history.append(
        {
            "step": st.session_state.sim_index + 1,
            "sentence": sentence,
            "picked_doc": picked_doc,
            "picked_score": picked_score,
            "top_matches": workspace["ranked"][:3],
        }
    )

    st.session_state.sim_last_workspace = workspace
    st.session_state.sim_index += 1


initialize_state()
module = load_demo_module()

st.title("Production Vector Database Demo")
st.caption("Interactive UI for step-by-step demonstrations (English vs Shona)")

with st.sidebar:
    st.subheader("Controls")
    if st.button("Start New Demo", use_container_width=True):
        initialize_demo(module)
    if st.button("Reset", use_container_width=True):
        reset_demo()

if st.session_state.demo is None:
    st.info("Click 'Start New Demo' in the sidebar to initialize the demo.")
    st.stop()

st.success("Demo initialized. Use guided mode below to present one section at a time.")

total_steps = len(SECTIONS)
current_step = st.session_state.current_step
progress = min(current_step / total_steps, 1.0)
st.progress(progress, text=f"Progress: {current_step}/{total_steps} sections completed")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Guided Mode")
    if current_step < total_steps:
        section_name, method_name = SECTIONS[current_step]
        st.write(f"Next section: **{section_name}**")
        if st.button(f"Run {section_name}", type="primary", use_container_width=True):
            run_step(section_name, method_name)
            st.session_state.current_step += 1
            st.rerun()
    else:
        st.success("All sections completed.")

with col2:
    st.subheader("Quick Run")
    selected_section = st.selectbox(
        "Run any section",
        options=SECTIONS,
        format_func=lambda item: item[0],
    )
    if st.button("Run Selected Section", use_container_width=True):
        run_step(selected_section[0], selected_section[1])
        st.rerun()

st.divider()
st.subheader("Output")

for key, value in st.session_state.logs.items():
    with st.expander(key, expanded=(key == "Initialization")):
        st.code(value or "No output.", language="text")

st.divider()
st.subheader("Cosine Similarity Visualizer")
st.caption(
    "Explore high-dimensional embeddings (projected to 2D) and pick vectors by cosine similarity."
)

default_docs = """I am feeling very happy and joyful today
I am sad and depressed about the situation
Dogs are loyal and friendly companion animals
I love to eat pizza and pasta for dinner
True happiness comes from within yourself
Ndiri kufara zvikuru nhasi
Ndiri kushushikana pamusoro pemamiriro ezvinhu"""

vector_col1, vector_col2 = st.columns([2, 1])

with vector_col1:
    raw_documents = st.text_area(
        "Document list (one document per line)",
        value=default_docs,
        height=180,
    )
    query_text = st.text_input("Query text", value="happy feelings")

with vector_col2:
    threshold = st.slider(
        "Cosine threshold (selected vectors)",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )
    target_cosine = st.number_input(
        "Target cosine value (pick nearest vector)",
        min_value=-1.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
    )

if st.button("Generate Vector Map", type="primary"):
    docs = parse_documents(raw_documents)
    if not docs:
        st.warning("Please enter at least one document line.")
    elif not query_text.strip():
        st.warning("Please enter a query text.")
    else:
        workspace = build_vector_workspace(
            st.session_state.demo,
            docs,
            query_text.strip(),
            threshold,
            float(target_cosine),
        )
        st.session_state.vector_workspace = workspace

workspace = st.session_state.vector_workspace
if workspace:
    doc_vectors = workspace["doc_vectors"]
    scores = workspace["scores"]
    picked_index = workspace["picked_index"]
    selected_indices = workspace["selected_indices"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Embedding dimensions", f"{doc_vectors.shape[1]}")
    metric_col2.metric("Documents", f"{len(workspace['documents'])}")
    metric_col3.metric("Selected by threshold", f"{len(selected_indices)}")

    picked_text = workspace["documents"][picked_index]
    picked_score = float(scores[picked_index])
    st.info(
        f"Picked by target cosine {workspace['target_cosine']:.2f}: "
        f"Doc {picked_index + 1} (score {picked_score:.3f})"
    )
    st.write(f"Picked document: {picked_text}")

    st.vega_lite_chart(
        {
            "data": {"values": workspace["points"]},
            "mark": {"type": "point", "filled": True, "size": 220},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": "Projection axis 1"},
                "y": {"field": "y", "type": "quantitative", "title": "Projection axis 2"},
                "color": {
                    "field": "kind",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Query", "Picked", "Selected", "Not selected"],
                        "range": ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"],
                    },
                    "title": "Point Type",
                },
                "tooltip": [
                    {"field": "label", "type": "nominal"},
                    {"field": "kind", "type": "nominal"},
                    {"field": "cosine", "type": "quantitative", "format": ".3f"},
                    {"field": "text", "type": "nominal"},
                ],
            },
            "width": "container",
            "height": 420,
            "title": "High-dimensional vectors projected to 2D",
        },
        use_container_width=True,
    )

    st.subheader("Cosine Ranking")
    st.dataframe(workspace["ranked"], use_container_width=True, hide_index=True)

    with st.expander("Show high-dimensional vector values"):
        st.write("Query vector (first 16 dimensions)")
        st.code(np.array2string(workspace["query_vector"][:16], precision=4), language="text")
        st.write("Picked document vector (first 16 dimensions)")
        st.code(
            np.array2string(workspace["doc_vectors"][picked_index][:16], precision=4),
            language="text",
        )

st.divider()
st.subheader("Sentence-by-Sentence Simulation Tool")
st.caption(
    "Simulate incoming text stream: process one sentence at a time and show which vector is picked."
)

simulation_default = (
    "Ndiri kufara today because our team closed the sprint. "
    "Now I am worried about latency in production. "
    "Can we find documents about food and restaurants?"
)

sim_col1, sim_col2 = st.columns([2, 1])

with sim_col1:
    incoming_text = st.text_area(
        "Incoming text stream",
        value=simulation_default,
        height=130,
        key="simulation_input",
    )

with sim_col2:
    sim_threshold = st.slider(
        "Simulation cosine threshold",
        min_value=-1.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        key="simulation_threshold",
    )
    sim_target = st.number_input(
        "Simulation target cosine",
        min_value=-1.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
        key="simulation_target",
    )

docs_for_simulation = parse_documents(raw_documents)

sim_actions_1, sim_actions_2, sim_actions_3 = st.columns(3)
with sim_actions_1:
    if st.button("Start Simulation", type="primary", use_container_width=True):
        if not docs_for_simulation:
            st.warning("Please provide at least one document in the document list above.")
        else:
            start_sentence_simulation(incoming_text)
            st.rerun()

with sim_actions_2:
    if st.button("Process Next Sentence", use_container_width=True):
        if not docs_for_simulation:
            st.warning("Please provide at least one document in the document list above.")
        elif not st.session_state.sim_sentences:
            st.warning("Click 'Start Simulation' first.")
        else:
            process_sentence_step(
                st.session_state.demo,
                docs_for_simulation,
                sim_threshold,
                float(sim_target),
            )
            st.rerun()

with sim_actions_3:
    if st.button("Process All Remaining", use_container_width=True):
        if not docs_for_simulation:
            st.warning("Please provide at least one document in the document list above.")
        elif not st.session_state.sim_sentences:
            st.warning("Click 'Start Simulation' first.")
        else:
            while st.session_state.sim_index < len(st.session_state.sim_sentences):
                process_sentence_step(
                    st.session_state.demo,
                    docs_for_simulation,
                    sim_threshold,
                    float(sim_target),
                )
            st.rerun()

if st.session_state.sim_sentences:
    processed = st.session_state.sim_index
    total = len(st.session_state.sim_sentences)
    st.progress(processed / total, text=f"Simulation progress: {processed}/{total} sentences processed")

    if processed < total:
        st.info(f"Next sentence: {st.session_state.sim_sentences[processed]}")
    else:
        st.success("Simulation complete. All sentences processed.")

if st.session_state.sim_history:
    st.subheader("Simulation Timeline")
    for item in st.session_state.sim_history:
        with st.expander(f"Step {item['step']}: {item['sentence']}", expanded=(item["step"] == len(st.session_state.sim_history))):
            st.write(f"Picked document: {item['picked_doc']}")
            st.write(f"Picked cosine score: {item['picked_score']:.3f}")
            st.dataframe(item["top_matches"], use_container_width=True, hide_index=True)

if st.session_state.sim_last_workspace:
    st.subheader("Current Sentence Vector Map")
    st.vega_lite_chart(
        {
            "data": {"values": st.session_state.sim_last_workspace["points"]},
            "mark": {"type": "point", "filled": True, "size": 220},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": "Projection axis 1"},
                "y": {"field": "y", "type": "quantitative", "title": "Projection axis 2"},
                "color": {
                    "field": "kind",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Query", "Picked", "Selected", "Not selected"],
                        "range": ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"],
                    },
                    "title": "Point Type",
                },
                "tooltip": [
                    {"field": "label", "type": "nominal"},
                    {"field": "kind", "type": "nominal"},
                    {"field": "cosine", "type": "quantitative", "format": ".3f"},
                    {"field": "text", "type": "nominal"},
                ],
            },
            "width": "container",
            "height": 420,
            "title": "Sentence Simulation: projected vector space",
        },
        use_container_width=True,
    )

outputs_dir = BASE_DIR / "outputs"
if outputs_dir.exists():
    st.divider()
    st.subheader("Saved Files")
    json_file = outputs_dir / "production_demo_results.json"
    txt_file = outputs_dir / "production_demo_summary.txt"

    if json_file.exists():
        st.write(f"- {json_file}")
    if txt_file.exists():
        st.write(f"- {txt_file}")
