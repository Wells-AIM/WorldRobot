"""CLI entry point; help and record inspection do not import the robot stack."""

import argparse
import json
from pathlib import Path

from . import __version__
from .config import TASKS, load_task
from .experiment import MODES
from .futures import CONTINUATIONS


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="jev-libero", description="Fine-grained Jev control with local physics previews."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="List bundled task configurations")
    validate = sub.add_parser(
        "validate-task", help="Validate a task JSON without simulation or API calls"
    )
    validate.add_argument("task", help="Bundled name or JSON path")
    show = sub.add_parser("inspect", help="Display a recorded outcome; no simulator/API needed")
    show.add_argument("folder", type=Path)
    live = sub.add_parser("run", help="Start a NEW, paid Jev-controlled episode")
    live.add_argument("--task", default="microwave", help="microwave, top_drawer, or a JSON path")
    live.add_argument(
        "--out", type=Path, required=True, help="New output directory (must not exist)"
    )
    live.add_argument("--seed", type=int, default=0)
    live.add_argument("--init-state", type=int, default=0)
    live.add_argument("--max-decisions", type=int, default=100)
    live.add_argument("--budget-usd", type=float, default=0.10)
    live.add_argument("--lookahead", type=int, choices=(1, 2), default=2)
    live.add_argument(
        "--mode",
        choices=MODES,
        default="original",
        help="original keeps the published three-layer pipeline; the others are controlled "
        "experiment arms that differ only in how much future consequence Jev sees",
    )
    live.add_argument(
        "--future-horizon",
        type=float,
        default=1.2,
        dest="future_horizon_s",
        help="Simulated future horizon in seconds for counterfactual_future/shuffle_future",
    )
    live.add_argument(
        "--candidate-count",
        type=int,
        default=27,
        help="Candidate chunks per decision, capped at the feasible pool. The default 27 "
        "offers every feasible input, which removes the sampler as a confound",
    )
    live.add_argument("--candidate-depth", type=int, default=3)
    live.add_argument("--shuffle-seed", type=int, default=0)
    live.add_argument(
        "--chunk-continuation",
        choices=CONTINUATIONS,
        default="repeat",
        help="How a candidate chunk continues past its first input. repeat saturates "
        "for large steps and hides their advantage; hold lets the action settle",
    )
    live.add_argument(
        "--no-render", action="store_true", help="Save controls/state, but no GIF or camera frames"
    )
    live.add_argument(
        "--provider",
        choices=("openrouter", "typesafe", "mock"),
        default="openrouter",
        help="mock is a deterministic offline stand-in: no network, no spend, and NOT Jev",
    )
    live.add_argument(
        "--api-key-file",
        type=Path,
        help="Private key file; otherwise use the selected provider's API_KEY[_FILE] variables",
    )
    replay = sub.add_parser("replay", help="Replay recorded controls; no API key or model calls")
    replay.add_argument("folder", type=Path)
    for command in (live, replay):
        command.add_argument(
            "--libero-root", type=Path, help="LIBERO checkout; defaults to LIBERO_ROOT"
        )
        command.add_argument(
            "--libero-config-dir", type=Path, help="Isolated config/cache directory"
        )
    args = parser.parse_args(argv)
    if args.command == "tasks":
        for path in sorted(TASKS.glob("*.json")):
            cfg = load_task(path)
            print(f"{path.stem:16} {cfg['policy']['task_text']}")
    elif args.command == "validate-task":
        print(json.dumps({"valid": True, "task": load_task(args.task)["name"]}, indent=2))
    elif args.command == "inspect":
        from .records import inspect

        print(json.dumps(inspect(args.folder), indent=2))
    elif args.command == "replay":
        from .records import replay

        print(json.dumps(replay(args.folder, args.libero_root, args.libero_config_dir), indent=2))
    elif args.command == "run":
        from .runner import run

        result = run(
            task=args.task,
            out=args.out,
            seed=args.seed,
            init_state=args.init_state,
            max_decisions=args.max_decisions,
            budget_usd=args.budget_usd,
            lookahead=args.lookahead,
            render=not args.no_render,
            libero_root=args.libero_root,
            config_dir=args.libero_config_dir,
            key_file=args.api_key_file,
            provider=args.provider,
            mode=args.mode,
            candidate_count=args.candidate_count,
            candidate_depth=args.candidate_depth,
            future_horizon_s=args.future_horizon_s,
            shuffle_seed=args.shuffle_seed,
            chunk_continuation=args.chunk_continuation,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
