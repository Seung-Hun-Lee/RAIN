from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarize_results(all_results: list[dict]):
    grouped: dict[str, list[dict]] = {}
    for result in all_results:
        grouped.setdefault(result["task_id"], []).append(result)

    total_success = 0
    total_episodes = 0
    cat_rates: dict[str, list[float]] = {}
    task_summaries = []

    for task_id, task_results in grouped.items():
        successes = sum(1 for result in task_results if result["success"])
        num_episodes = len(task_results)
        category = task_results[0]["category"]
        description = task_results[0]["task_description"]
        avg_depth = np.mean([r["meta"]["subtasks_reached"] for r in task_results])
        num_subtasks = task_results[0]["meta"]["num_subtasks"]
        success_rate = successes / num_episodes * 100 if num_episodes else 0.0
        task_summaries.append(
            {
                "task_id": task_id,
                "task_description": description,
                "category": category,
                "successes": successes,
                "episodes": num_episodes,
                "success_rate": success_rate,
                "avg_subtasks_reached": float(avg_depth),
                "subtasks_total": num_subtasks,
            }
        )
        total_success += successes
        total_episodes += num_episodes
        cat_rates.setdefault(category, []).append(success_rate)

    overall = total_success / total_episodes * 100 if total_episodes else 0.0
    return grouped, task_summaries, total_success, total_episodes, cat_rates, overall


def print_console_report(
    grouped_results: dict[str, list[dict]],
    cat_rates: dict[str, list[float]],
    total_success: int,
    total_episodes: int,
    overall: float,
    elapsed: float,
) -> None:
    print("\n" + "=" * 80)
    print("LIBEROEX EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Task ID':<12} | {'Task Name':<55} | {'SR':>10} | {'Depth':>10}")
    print("-" * 100)
    for task_id, task_results in grouped_results.items():
        successes = sum(1 for result in task_results if result["success"])
        num_episodes = len(task_results)
        description = task_results[0]["task_description"]
        avg_depth = np.mean([r["meta"]["subtasks_reached"] for r in task_results])
        num_subtasks = task_results[0]["meta"]["num_subtasks"]
        print(
            f"{task_id:<12} | {description:<55} | "
            f"{successes:>4}/{num_episodes:<4} | {avg_depth:.1f}/{num_subtasks}"
        )
    print("-" * 100)
    for category in sorted(cat_rates):
        print(f"{category:<12}   {np.mean(cat_rates[category]):>6.1f}%  ({len(cat_rates[category])} tasks)")
    print(f"{'OVERALL':<12}   {overall:>6.1f}%  ({len(grouped_results)} tasks)")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 80)


def write_result_artifacts(save_dir: Path, output: dict) -> tuple[Path, Path, Path]:
    grouped_results = {
        task_id: list(task_results)
        for task_id, task_results in _group_results(output["episodes"]).items()
    }

    results_path = save_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    summary_path = save_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write("# LiberoEX Evaluation Results\n\n")
        f.write(f"- **Action checkpoint**: `{output['checkpoint']}`\n")
        f.write(f"- **Progress checkpoint**: `{output['progress_checkpoint']}`\n")
        f.write(f"- **Action plan source**: `{output['action_plan_source']}`\n")
        f.write(f"- **Mask JSON**: `{output['episodes_json']}`\n")
        f.write(
            f"- **Overall**: {output['success_rate']:.1f}% "
            f"({output['total_success']}/{output['total_episodes']})\n"
        )
        f.write(f"- **Time**: {output['elapsed_seconds']:.1f}s\n\n")
        f.write("| Task ID | Task Name | Category | SR | Depth |\n")
        f.write("|---------|-----------|----------|----|-------|\n")
        for task_id, task_results in grouped_results.items():
            successes = sum(1 for result in task_results if result["success"])
            num_episodes = len(task_results)
            avg_depth = np.mean([r["meta"]["subtasks_reached"] for r in task_results])
            num_subtasks = task_results[0]["meta"]["num_subtasks"]
            category = task_results[0]["category"]
            description = task_results[0]["task_description"]
            f.write(
                f"| {task_id} | {description} | {category} | "
                f"{successes}/{num_episodes} ({successes / num_episodes * 100:.0f}%) | "
                f"{avg_depth:.1f}/{num_subtasks} |\n"
            )

    tsv_path = save_dir / "results_for_excel.tsv"
    with tsv_path.open("w") as f:
        f.write("Task ID\tTask Name\tCategory\tSubtasks\tSuccess\tTotal\tSR(%)\n")
        for task_id, task_results in grouped_results.items():
            successes = sum(1 for result in task_results if result["success"])
            num_episodes = len(task_results)
            num_subtasks = task_results[0]["meta"]["num_subtasks"]
            category = task_results[0]["category"]
            description = task_results[0]["task_description"]
            success_rate = successes / num_episodes * 100 if num_episodes else 0.0
            f.write(
                f"{task_id}\t{description}\t{category}\t{num_subtasks}\t"
                f"{successes}\t{num_episodes}\t{success_rate:.1f}\n"
            )

    return results_path, summary_path, tsv_path


def _group_results(all_results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for result in all_results:
        grouped.setdefault(result["task_id"], []).append(result)
    return grouped
