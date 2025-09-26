# OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent
This repo provides a minimal, runnable artifact accompanying the paper.

## Overview
This repository contains the official implementation of the LLM agent system described for ad keyword generation in the paper [**“OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent”**]((https://arxiv.org/abs/2507.02353)) accepted by EMNLP'25, Main Track, written by Bowen Chen, Zhao Wang, Shingo Takamatsu. ***OMS*** proposes a keyword generation framework that is On-the-fly (requires no training data, monitors online performance, and adapts accordingly), Multi-objective (employs agentic reasoning to optimize keywords based on multiple performance metrics), and Self-reflective (agentically evaluates keyword quality

## Overview
- **Self‑contained** local copy of the ad keyword/text generation pipeline.
- **No cloud dependencies**: AWS S3 and Google Ads utilities removed. Search‑volume checks use a local stub.
- **Clustering preserved**: BERT embeddings + lightweight LLM calls for descriptions.

## Folder Structure
- `config/` – runtime configuration (edit `config_non_branded_NuroBiz.ini`)
- `keywords/`
  - `init/` – seed keywords CSV (editable)
  - `gen/` – generated keywords CSV (output)
  - `rule/` – keyword rules/examples
- `src/`
  - `main.py` – entrypoint
  - `ad_agent.py`, `okg/`, `tools.py` – core logic
  - `history_data/` – prepped example data



## Requirements
- **Python** 3.10+
- Common packages: `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `langchain`, `langchain-community`, `langchain-openai`, `faiss-cpu` (or a platform‑appropriate FAISS build)


## Quick Start
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


## Input & Output
- **Input seed keywords**: `keywords/init/nurobiz_non_brand_jp.csv` (replace with your own if needed)
- **Generated keywords**: `keywords/gen/nurobiz_non_brand_jp.csv`

## Notes
- The first run may download model weights; subsequent runs are faster.
- If offline, ensure the BERT model is cached or switch to a lighter embedding method.
- Google Ads API and S3 are intentionally removed for publishing, as they require credentials which we cannot provide them in the Github version. If you plan to connect to Google Ads or S3, add those functions separately. This repository does not perform any remote I/O to Google Ads or Amazon S3.

## Key Removals in the Published Version
- **Removed**: all `google_util` and S3 usage. No external file access outside this folder.
- **Search volume**: uses a local stub (fixed values). To re‑enable real checks, integrate your own data source by using Google Ads or data from other SSA platforms.

## Configuration
- Edit `config/config_non_branded_NuroBiz.ini`
  - **[EXE]** `S3_DEPLOY=False` (local only)
  - **[SETTING]** paths under `../keywords/...` must remain within this folder
  - **[KEYS]** placeholders (e.g., `YOUR_OPENAI_...`) are kept for future integrations
  - **[KEYWORD]** `SEARCH_VOLUMN_CHECK=False` by default (uses local stub)

## Disclaimer
- Keys in the example configuration are only for paper reproduction. Do not commit them to public repositories. After confirming local runs, remove keys or switch to environment variables.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## Contact
For any questions or issues, feel free to reach out: Zhao.Wang@sony.com or this github repo for any information.

## Cite
If you use or reference ***TalkHier***, please cite us with the following BibTeX entry:
```bibtex
@misc{ChenWangTakamatsu_2025_OMS,
  title        = {OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent},
  author       = {Chen, Bowen and Wang, Zhao and Takamatsu, Shingo},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  url          = {https://arxiv.org/abs/2507.02353}
}
```
