# Vector Database Demo: Shona vs English (RAG Presentation)

This project is built around one core experiment:

**Why does vector search work well for English, but fail for Shona unless we add translation?**

It demonstrates the failure mode and a practical production-ready fix.

## What This Demo Proves

1. **English baseline works** with standard embedding models.
2. **Direct Shona embedding is unreliable** with English-centric models.
3. **Translation pipeline restores useful semantic retrieval**.
4. **Bilingual retrieval is possible** in one system.

## Files

- `vector-demo.py` - terminal guided scenario demo
- `streamlit_demo.py` - presentation UI with vector-space visualization
- `embed.py` - minimal Ollama embedding example
- `notes/` - supporting notes

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Demo

### Option A: Streamlit UI (best for non-technical audience)

```bash
streamlit run streamlit_demo.py
```

Includes:
- step-by-step scenario control
- cosine similarity visualizer
- high-dimensional vectors projected to 2D dots
- cosine threshold selection
- nearest-vector pick by target cosine value

### Option B: Terminal mode

```bash
python3 vector-demo.py
```

Modes:
- guided mode (press continue between sections)
- auto-run mode

## Scenario Flow (Story Arc)

1. English success
2. Shona failure (without translation)
3. Translation fix
4. Bilingual mixed retrieval
5. Final report

## Output Files

- `outputs/production_demo_results.json`
- `outputs/production_demo_summary.txt`

## Public Release Checklist

- [ ] Confirm the app runs (`streamlit` + terminal)
- [ ] Ensure `venv/` and `outputs/` are not tracked
- [ ] Review docs for final wording
- [ ] Push to GitHub

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial public release: vector DB + RAG presentation demo"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## License

MIT (see [LICENSE](LICENSE)).
