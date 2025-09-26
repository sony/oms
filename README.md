## Repository: LLM Ad Keyword/Text for EMNLP 2025 Publication

### Overview
- **Self‑contained** local copy of the ad keyword/text generation pipeline.
- **No cloud dependencies**: AWS S3 and Google Ads utilities removed. Search‑volume checks use a local stub.
- **Clustering preserved**: BERT embeddings + lightweight LLM calls for descriptions.

### Folder Structure
- `config/` – runtime configuration (edit `config_non_branded_NuroBiz.ini`)
- `keywords/`
  - `init/` – seed keywords CSV (editable)
  - `gen/` – generated keywords CSV (output)
  - `rule/` – keyword rules/examples
- `src/`
  - `main.py` – entrypoint
  - `ad_agent.py`, `okg/`, `tools.py` – core logic
  - `history_data/` – prepped example data



### Requirements
- **Python** 3.10+
- Common packages: `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `langchain`, `langchain-community`, `langchain-openai`, `faiss-cpu` (or a platform‑appropriate FAISS build)


### Quick Start
1) (Optional) Create venv and install packages
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install pandas numpy scikit-learn transformers torch langchain langchain-community langchain-openai faiss-cpu
   ```
2) Run the pipeline
   ```bash
   cd src
   python -c "from main import main; main()"
   ```


### Input & Output
- **Input seed keywords**: `keywords/init/nurobiz_non_brand_jp.csv` (replace with your own if needed)
- **Generated keywords**: `keywords/gen/nurobiz_non_brand_jp.csv`

### Notes
- The first run may download model weights; subsequent runs are faster.
- If offline, ensure the BERT model is cached or switch to a lighter embedding method.
- Google Ads API and S3 are intentionally removed for publishing, as they require credentials which we cannot provide them in the Github version. If you plan to connect to Google Ads or S3, add those functions separately. This repository does not perform any remote I/O to Google Ads or Amazon S3.

### Key Removals in the Published Version
- **Removed**: all `google_util` and S3 usage. No external file access outside this folder.
- **Search volume**: uses a local stub (fixed values). To re‑enable real checks, integrate your own data source by using Google Ads or data from other SSA platforms.

### Configuration
- Edit `config/config_non_branded_NuroBiz.ini`
  - **[EXE]** `S3_DEPLOY=False` (local only)
  - **[SETTING]** paths under `../keywords/...` must remain within this folder
  - **[KEYS]** placeholders (e.g., `YOUR_OPENAI_...`) are kept for future integrations
  - **[KEYWORD]** `SEARCH_VOLUMN_CHECK=False` by default (uses local stub)




### QA
- Feel free to open an issue in this repository or contact the corresponding authors of this research.