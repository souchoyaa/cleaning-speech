"""Report building blocks for the dedup + quality pipeline.

A file-driven menu of selectable evaluation components (figures / tables /
sentences), one per pipeline stage plus end-to-end and multilingual views.

    from report.loader import PipelineReport
    from report import blocks, build

CLI:  python3 build.py --list
      python3 build.py --run <run_dir|config.yaml> --all --out report_out
      python3 build.py --run <...> --blocks overview_summary,asr_model_wer --out report_out
"""
