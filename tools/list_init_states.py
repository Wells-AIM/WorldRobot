#!/usr/bin/env python3
"""How many fixed initial states LIBERO actually ships for each bundled task.

Stage 2 uses LIBERO's own fixed states rather than synthesising any, so this
records what is available and where it came from.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.config import load_task  # noqa: E402
from jev_libero.environment import load_libero  # noqa: E402

TASKS = ("top_drawer", "microwave", "alphabet_soup")


def main(out=Path("results/stage2/available_initial_states.json")):
    benchmark, _, _, _ = load_libero()
    records = {}
    for name in TASKS:
        cfg = load_task(name)
        binding = cfg["binding"]
        suite = benchmark.get_benchmark_dict()[binding["suite"]]()
        index = next(
            i for i in range(suite.n_tasks) if suite.get_task(i).name == binding["task"]
        )
        states = suite.get_task_init_states(index)
        records[name] = {
            "task": name,
            "libero_suite": binding["suite"],
            "libero_task": binding["task"],
            "available_initial_states": int(len(states)),
            "usable_ids": list(range(min(10, len(states)))),
            "source": "LIBERO get_task_init_states (fixed init files), not synthesised",
        }
        print(f"{name:16} {len(states):4} fixed initial states "
              f"({binding['suite']} / {binding['task']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
