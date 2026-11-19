"""rag_halluc: sentence-level hallucination detection for RAG with small LMs.

This package sits beside the original LLM-Check code (common_utils.py,
utils_*.py, run_detection_combined.py), which is kept untouched as the
reference implementation. Everything new for the project lives here.

Layout (filled in phase by phase, see docs/ROADMAP.md):
    models.py          reader-model registry, loading, chat-template encoding
    data/ragtruth.py   RAGTruth download + join (Phase 0), sentence labels (Phase 1)
    eval/metrics.py    AUROC and related metrics
    repro/             Phase 0 reproduction of LLM-Check on small readers
"""

__version__ = "0.1.0"
