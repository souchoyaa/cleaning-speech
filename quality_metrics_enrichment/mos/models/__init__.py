"""MOS metric model builders.

Each module exposes a single ``build_<metric>(cfg, device)`` factory
returning the loaded model. Adding a new metric is a new file here plus
one switch in ``mos.worker.MosAssessmentWorker.__init__``.
"""
