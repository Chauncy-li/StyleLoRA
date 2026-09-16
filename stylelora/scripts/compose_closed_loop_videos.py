"""Select a small token set and compose nine closed-loop renders into 3x3 videos.

The NuPlan renderer writes one ``sim_<start_time_us>.avi`` file per scenario.
This utility uses the matching raw-step directory to recover the scenario token,
then aligns the same scenario across the nine StyleLoRA rho values.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # Token selection does not require video dependencies.
    cv2 = None
    np = None

from stylelora.closed_loop_constants import ROUTE_FAILURE_TOKENS


PANEL_LAYOUT = (
    ("rho=-1.00", Path("SHARD_NEGATIVE/rho_minus1.00")),
    ("rho=-0.75", Path("SHARD_NEGATIVE/rho_minus0.75")),
    ("rho=-0.50", Path("SHARD_NEGATIVE/rho_minus0.50")),
    ("rho=-0.25", Path("SHARD_CENTER/rho_minus0.25")),
    ("rho=+0.00", Path("SHARD_CENTER/rho_plus0.00")),
    ("rho=+0.25", Path("SHARD_CENTER/rho_plus0.25")),
    ("rho=+0.50", Path("SHARD_POSITIVE/rho_plus0.50")),
    ("rho=+0.75", Path("SHARD_POSITIVE/rho_plus0.75")),
    ("rho=+1.00", Path("SHARD_POSITIVE/rho_plus1.00")),
)
VIDEO_TIME_RE = re.compile(r"^sim_(\d+)$")
STEP_TIME_RE = re.compile(r"_(\d+)\.npz$")


def _load_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        partial = payload.get("partial_collection")
        if isinstance(partial, dict) and partial.get("common_successful_tokens_within_configuration"):
            payload = partial["common_successful_tokens_within_configuration"]
        elif isinstance(payload.get("settings"), dict):
            payload = payload["settings"].get("scenario_tokens")
        else:
            payload = payload.get("scenario_tokens", payload.get("tokens"))
    if not isinstance(payload, list) or not all(isinstance(item, str) and item for item in payload):
        raise ValueError(f"{path} must be a non-empty JSON token list")
    if len(payload) != len(set(payload)):
        raise ValueError(f"{path} contains duplicate tokens")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_tokens(args: argparse.Namespace) -> None:
    sources = [Path(item).expanduser().resolve() for item in args.source]
    output = Path(args.output).expanduser().resolve()
    token_lists = [_load_tokens(source) for source in sources]
    common = set.intersection(*(set(items) for items in token_lists))
    candidates = [
        token for token in token_lists[0] if token in common and token not in ROUTE_FAILURE_TOKENS
    ]
    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} usable tokens are available; requested {args.count}")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(candidates)), args.count))
    selected = [candidates[index] for index in selected_indices]
    _write_json(output, selected)
    _write_json(
        output.with_name(output.stem + "_report.json"),
        {
            "sources": [str(source) for source in sources],
            "source_token_counts": [len(items) for items in token_lists],
            "output": str(output),
            "seed": args.seed,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "scenario_tokens": selected,
        },
    )
    print(f"Selected {len(selected)} scenarios -> {output}")
    for index, token in enumerate(selected, start=1):
        print(f"  {index}: {token}")


def _video_index(video_dir: Path) -> dict[str, Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    result: dict[str, Path] = {}
    for path in sorted(video_dir.iterdir()):
        if path.suffix.lower() not in {".avi", ".mp4"}:
            continue
        match = VIDEO_TIME_RE.match(path.stem)
        if match:
            result[match.group(1)] = path
    if not result:
        raise FileNotFoundError(f"No sim_<timestamp>.avi/mp4 files found in: {video_dir}")
    return result


def _token_time_index(raw_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return token->start timestamp and timestamp->token indexes."""
    token_to_time: dict[str, str] = {}
    time_to_token: dict[str, str] = {}
    if not raw_dir.is_dir():
        return token_to_time, time_to_token
    for token_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        step_files = sorted(token_dir.glob("step_*.npz"))
        if not step_files:
            continue
        match = STEP_TIME_RE.search(step_files[0].name)
        if not match:
            continue
        timestamp = match.group(1)
        token_to_time[token_dir.name] = timestamp
        time_to_token[timestamp] = token_dir.name
    return token_to_time, time_to_token


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _label_panel(frame: np.ndarray, label: str, ended: bool) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.62, output, 0.38, 0, output)
    text = label + ("  [ended]" if ended else "")
    cv2.putText(output, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return output


def _compose_one(
    inputs: list[tuple[str, Path]],
    output: Path,
    panel_size: int,
    requested_fps: float | None,
) -> dict[str, object]:
    captures = [cv2.VideoCapture(str(path)) for _, path in inputs]
    try:
        unopened = [str(path) for (_, path), capture in zip(inputs, captures) if not capture.isOpened()]
        if unopened:
            raise RuntimeError("Could not open input videos:\n  - " + "\n  - ".join(unopened))
        source_fps = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
        fps = requested_fps or next((value for value in source_fps if value > 0), 10.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (panel_size * 3, panel_size * 3)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output}")

        last_frames: list[np.ndarray | None] = [None] * len(captures)
        ended = [False] * len(captures)
        frame_count = 0
        try:
            while True:
                received_new_frame = False
                panels: list[np.ndarray] = []
                for index, ((label, _), capture) in enumerate(zip(inputs, captures)):
                    if not ended[index]:
                        ok, frame = capture.read()
                        if ok:
                            last_frames[index] = frame
                            received_new_frame = True
                        else:
                            ended[index] = True
                    if last_frames[index] is None:
                        panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                    else:
                        panel = _letterbox(last_frames[index], panel_size)
                    panels.append(_label_panel(panel, label, ended[index]))
                if not received_new_frame:
                    break
                rows = [cv2.hconcat(panels[index : index + 3]) for index in range(0, 9, 3)]
                writer.write(cv2.vconcat(rows))
                frame_count += 1
        finally:
            writer.release()
        return {
            "output": str(output),
            "fps": float(fps),
            "frame_count": frame_count,
            "duration_seconds": frame_count / float(fps),
            "inputs": {label: str(path) for label, path in inputs},
        }
    finally:
        for capture in captures:
            capture.release()


def _ordered_scenarios(
    common_times: set[str],
    token_to_time: dict[str, str],
    time_to_token: dict[str, str],
    tokens_file: str | None,
    expected_scenes: int,
) -> list[tuple[str, str]]:
    if tokens_file:
        tokens = _load_tokens(Path(tokens_file).expanduser().resolve())
        if len(tokens) != expected_scenes:
            raise ValueError(
                f"Token file contains {len(tokens)} scenarios, expected exactly {expected_scenes}: {tokens_file}"
            )
        missing_raw = [token for token in tokens if token not in token_to_time]
        if missing_raw:
            raise RuntimeError("No raw-step timestamp found for tokens:\n  - " + "\n  - ".join(missing_raw))
        result = [(token, token_to_time[token]) for token in tokens]
        missing_video = [(token, timestamp) for token, timestamp in result if timestamp not in common_times]
        if missing_video:
            details = "\n  - ".join(f"{token} (sim_{timestamp})" for token, timestamp in missing_video)
            raise RuntimeError("The following scenarios do not have all nine videos:\n  - " + details)
        return result

    ordered_times = sorted(common_times, key=int)
    if len(ordered_times) != expected_scenes:
        raise RuntimeError(
            f"Found {len(ordered_times)} scenarios shared by all nine rho values; expected {expected_scenes}. "
            "Use a fresh video output root or pass --tokens-file."
        )
    return [(time_to_token.get(timestamp, f"timestamp_{timestamp}"), timestamp) for timestamp in ordered_times]


def _compose(args: argparse.Namespace) -> None:
    if cv2 is None or np is None:
        raise RuntimeError("The compose command requires OpenCV (cv2) and NumPy")
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    panel_indexes: list[tuple[str, Path, dict[str, Path]]] = []
    for label, relative_root in PANEL_LAYOUT:
        rho_root = input_root / relative_root
        panel_indexes.append((label, rho_root, _video_index(rho_root / "simulation_video")))

    common_times = set.intersection(*(set(index) for _, _, index in panel_indexes))
    reference_root = panel_indexes[0][1]
    token_to_time, time_to_token = _token_time_index(reference_root / "raw_step_data")
    scenarios = _ordered_scenarios(
        common_times,
        token_to_time,
        time_to_token,
        args.tokens_file,
        args.expected_scenes,
    )

    records = []
    for scene_index, (token, timestamp) in enumerate(scenarios, start=1):
        inputs = [(label, index[timestamp]) for label, _, index in panel_indexes]
        output = output_dir / f"scene_{scene_index:02d}_{token}.mp4"
        print(f"[{scene_index}/{len(scenarios)}] {token} -> {output}", flush=True)
        record = _compose_one(inputs, output, args.panel_size, args.fps)
        record.update({"scene_index": scene_index, "scenario_token": token, "start_time_us": timestamp})
        records.append(record)

    report_path = output_dir / "video_grid_report.json"
    _write_json(
        report_path,
        {
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "layout": [label for label, _ in PANEL_LAYOUT],
            "grid_size": [3, 3],
            "panel_size": args.panel_size,
            "video_count": len(records),
            "videos": records,
        },
    )
    print(f"Finished: {len(records)} grid videos -> {output_dir}")
    print(f"Report: {report_path}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Deterministically select five tokens")
    select_parser.add_argument(
        "--source",
        required=True,
        action="append",
        help="Token JSON or closed-loop report; repeat to select from their successful-token intersection",
    )
    select_parser.add_argument("--output", required=True, help="Selected token JSON output")
    select_parser.add_argument("--count", type=_positive_int, default=5)
    select_parser.add_argument("--seed", type=int, default=17)
    select_parser.set_defaults(func=_select_tokens)

    compose_parser = subparsers.add_parser("compose", help="Compose nine rho videos into 3x3 MP4 files")
    compose_parser.add_argument(
        "--input-root",
        required=True,
        help="Root containing SHARD_NEGATIVE, SHARD_CENTER and SHARD_POSITIVE",
    )
    compose_parser.add_argument("--output-dir", required=True)
    compose_parser.add_argument("--tokens-file", default=None, help="Selected token JSON used for rendering")
    compose_parser.add_argument("--expected-scenes", type=_positive_int, default=5)
    compose_parser.add_argument("--panel-size", type=_positive_int, default=360)
    compose_parser.add_argument("--fps", type=float, default=None, help="Default: use source FPS")
    compose_parser.set_defaults(func=_compose)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "fps", None) is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
