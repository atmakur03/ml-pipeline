"""
pipeline.py
Orchestrator: runs the full ML pipeline end-to-end.
Stages: Ingest -> Features -> Train -> Predict
"""

import time
from src import ingest, features, train, predict


def run_pipeline():
    print("="*50)
    print(" ML PIPELINE STARTING")
    print("="*50)

    stages = [
        ("1. Ingestion",        ingest.run),
        ("2. Feature Engineering", features.run),
        ("3. Model Training",   train.run),
        ("4. Batch Inference",  predict.run),
    ]

    results = {}
    total_start = time.time()

    for name, fn in stages:
        print(f"\n--- {name} ---")
        stage_start = time.time()
        result = fn()
        elapsed = round(time.time() - stage_start, 2)
        print(f"[pipeline] {name} completed in {elapsed}s")
        results[name] = result

    total_elapsed = round(time.time() - total_start, 2)
    print(f"\n{'='*50}")
    print(f" PIPELINE COMPLETE in {total_elapsed}s")
    print(f"{'='*50}")
    return results


if __name__ == "__main__":
    run_pipeline()
