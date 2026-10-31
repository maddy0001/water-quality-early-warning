"""Local LLM incident reports (layer 7, Segment 7). Reads alarms; never writes them.

    records       alarm records (the facts a report may use)
    guidance.yaml WHO and project guidance snippets
    reporters     template and ollama report writers; guidance retrieval
    faithfulness  automatic fact check of a report against its record
    run           the S7 experiment
"""

from ewsv2.llm import reporters  # noqa: F401
