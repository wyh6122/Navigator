# Navigator: Knowledge-Guided Medical Dialogue Reasoning with Large Language Models

This is the official codebase of the **Navigator** framework for evidence-grounded medical dialogue reasoning with knowledge graph navigation and large language models.

Navigator is developed from the classic graph-of-thought prompting pipeline and follows a similar deployment and invocation workflow. It further introduces medical evidence filtering, graph navigation, and doctor-agent checking to improve the relevance of retrieved knowledge and the reliability of diagnosis-oriented generation.

## Overview

We present **Navigator**, a plug-and-play prompting framework that enables LLMs to navigate medical knowledge graphs, filter noisy evidence, and generate evidence-grounded medical responses. Given a patient question, Navigator extracts medical entities, matches them to knowledge graph nodes, retrieves path-based and neighbor-based evidence, and then generates the final answer with an interpretable reasoning trace.

![Navigator framework overview](navigator_framework.jpg)

The framework currently supports experiments on the `chatdoctor5k` dataset and includes scripts and data folders for CMCQA-based preprocessing and evaluation.

## Run Navigator

As the `chatdoctor5k` dataset for example. First, create a Blank Sandbox on [Neo4j Sandbox](https://sandbox.neo4j.com/), click "connect via drivers", and find your URL, username, and password. Then replace the following parts in `Navigator.py`:

```python
YOUR_OPENAI_KEY = "YOUR_OPENAI_KEY"

uri = "YOUR_URL"
username = "YOUR_USER"
password = "YOUR_PASSWORD"
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run Navigator:

```bash
python Navigator.py
```

The script will build the Neo4j knowledge graph from `data/chatdoctor5k/train.txt`, load entity and keyword embeddings, retrieve graph evidence, call LLMs for diagnosis-oriented generation, and save results to:

```text
output/chatdoctor5k/output.csv
```

Note that the program uses the legacy OpenAI Python SDK style and the dependency file pins `openai==0.27.7` and `langchain==0.0.181`. We recommend using a fresh Python environment when reproducing the experiments.
