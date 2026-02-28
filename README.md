# Vector Databases + RAG Demo (English vs Shona)

Interactive demo project for presenting:
- Vector embeddings
- Cosine similarity search
- Cross-language retrieval challenges
- Translation-assisted retrieval for low-resource languages
- Streamlit visualization of high-dimensional vectors projected to 2D

## Project Structure

- `vector-demo.py` - terminal guided demo (scenario-by-scenario)
- `streamlit_demo.py` - presentation-friendly web UI
- `embed.py` - simple Ollama embedding example
- `notes/` - supporting presentation notes

## Demo Scenarios

1. English baseline (works well)
2. Shona direct embedding (degraded quality)
3. Shona with translation pipeline (improved retrieval)
4. Bilingual mixed database
5. Final summary report

## Requirements

- Python 3.10+
- Internet access (for model/translator on first use)

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Options

### 1) Streamlit UI (recommended for presentations)

```bash
streamlit run streamlit_demo.py
```

What the UI includes:
- Guided section-by-section execution
- Quick run per section
- Cosine similarity visualizer
- 2D projection of high-dimensional vectors (dots)
- Threshold-based vector selection
- Pick-nearest-vector by target cosine value

### 2) Terminal Guided Demo

```bash
python3 vector-demo.py
```

Choose mode:
- Guided mode (press continue between sections)
- Auto-run all sections

## Outputs

Generated files are saved to:
- `outputs/production_demo_results.json`
- `outputs/production_demo_summary.txt`

## Presentation Script (Short)

- Start with Streamlit UI and run sections one at a time.
- Show that vectors are high-dimensional but projected to 2D for intuition.
- Use cosine threshold slider to explain selection boundaries.
- Use target cosine value to show exactly which vector gets picked.
- Compare direct Shona retrieval vs translation pipeline.

## Public Release Checklist

Before publishing:
- [ ] Remove any private keys or secrets
- [ ] Verify `venv/` is ignored
- [ ] Run the app once to confirm no runtime errors
- [ ] Add your repo URL in this README

## GitHub Publish Commands

```bash
git init
git add .
git commit -m "Initial public release: vector DB + RAG demo"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## License

MIT License (see LICENSE file).
