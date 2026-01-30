from importlib import import_module

try:
    codegen_metrics = import_module(
        "lcb_runner.evaluation.compute_code_generation_metrics"
    ).codegen_metrics
except ModuleNotFoundError:
    # Some forks rename/omit this file; keep evaluation importable.
    codegen_metrics = None
from lcb_runner.evaluation.compute_code_execution_metrics import code_execution_metrics
from lcb_runner.evaluation.compute_test_output_prediction_metrics import (
    test_output_metrics,
)
from lcb_runner.evaluation.pass_k_utils import extract_instance_results
