"""Evaluation helpers."""

from brian_sphere_llm.eval.compute_report import estimate_gpu_hours, make_compute_report
from brian_sphere_llm.eval.cost_control_report import make_cost_control_report
from brian_sphere_llm.eval.difficulty import difficulty_step_correlation, summarize_difficulty_samples
from brian_sphere_llm.eval.reasoning import make_reasoning_report
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report

__all__ = [
    "difficulty_step_correlation",
    "estimate_gpu_hours",
    "make_compute_report",
    "make_cost_control_report",
    "make_reasoning_report",
    "make_stage_gate_report",
    "summarize_difficulty_samples",
]
