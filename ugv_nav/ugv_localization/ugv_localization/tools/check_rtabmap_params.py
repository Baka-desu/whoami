"""Verify every RTAB-Map parameter we set actually exists in the installed RTAB-Map.

RTAB-Map silently ignores unknown parameter names, and names drift between versions
(e.g. Grid/FromDepth → Grid/Sensor). A typo in a config would quietly run defaults.

Library params (Group/Name) are checked against `rtabmap --params`. Node params (no "/") are
checked only when a live node's `ros2 param list` dump is given with --node-check CONFIG=DUMP.

Usage (on a machine with ROS + rtabmap_ros):
    ros2 run rtabmap_slam rtabmap --params > /tmp/rtabmap_params.txt
    ros2 run ugv_localization check_rtabmap_params --dump /tmp/rtabmap_params.txt
    # optional, with the stack running:
    ros2 param list /rtabmap/rtabmap > /tmp/slam_node.txt
    ros2 run ugv_localization check_rtabmap_params --dump /tmp/rtabmap_params.txt
        --node-check config/rtabmap_rgbd.yaml=/tmp/slam_node.txt   (same line)
Exit 0 = all known; 1 = unknown names printed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from ugv_localization.modes import Mode

_PARAM_LINE = re.compile(r"^\s*(?:Param:\s*)?([A-Za-z0-9_]+/[A-Za-z0-9_]+)\s*=")
_DEFAULT_CONFIGS = ("rtabmap_rgbd.yaml", "rgbd_odometry.yaml")


def parse_params_dump(text: str) -> set[str]:
    names = {m.group(1) for line in text.splitlines() if (m := _PARAM_LINE.match(line))}
    if not names:
        raise ValueError("no parameters recognized in dump — is this `rtabmap --params` output?")
    return names


def _config_keys(config_path: str | Path) -> set[str]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node_block in data.values():
        params = node_block.get("ros__parameters", {}) if isinstance(node_block, dict) else {}
        keys |= set(params)
    return keys


def config_library_keys(config_path: str | Path) -> set[str]:
    return {k for k in _config_keys(config_path) if "/" in k}


def config_node_keys(config_path: str | Path) -> set[str]:
    return {k for k in _config_keys(config_path) if "/" not in k}


def parse_node_param_list(text: str) -> set[str]:
    """`ros2 param list /node` output: indented names; a trailing-colon line is a node header."""
    names = {line.strip() for line in text.splitlines() if line.strip() and not line.strip().endswith(":")}
    names = {n for n in names if "/" not in n}  # node params only
    if not names:
        raise ValueError("no node parameters recognized — is this `ros2 param list /node` output?")
    return names


def _mode_keys() -> set[str]:
    # Local import keeps the module importable without touching the filesystem.
    from ugv_localization.modes.plan import _MODE_PARAMS

    return {k for mode in Mode for k in _MODE_PARAMS[mode]}


def unknown_keys(ours: set[str], known: set[str]) -> list[str]:
    return sorted(ours - known)


def _default_configs() -> list[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory

        cfg = Path(get_package_share_directory("ugv_localization")) / "config"
    except Exception:  # noqa: BLE001 — no ROS: fall back to the source tree
        cfg = Path(__file__).resolve().parents[2] / "config"
    return [cfg / name for name in _DEFAULT_CONFIGS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="file with `rtabmap --params` output, or - for stdin")
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help=f"params YAML, repeatable (default: installed {', '.join(_DEFAULT_CONFIGS)})",
    )
    ap.add_argument(
        "--node-check",
        action="append",
        default=[],
        metavar="CONFIG=DUMP",
        help="also check CONFIG's node params against a `ros2 param list /node` dump",
    )
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.dump == "-" else Path(args.dump).read_text(encoding="utf-8")
    known = parse_params_dump(text)
    cfgs = [Path(c) for c in args.config] if args.config else _default_configs()
    ours = set().union(*(config_library_keys(c) for c in cfgs)) | _mode_keys()
    failed = False
    bad = unknown_keys(ours, known)
    if bad:
        failed = True
        print(f"UNKNOWN RTAB-Map library parameters ({len(bad)}) in {', '.join(map(str, cfgs))} / mode params:")
        for name in bad:
            print(f"  {name}")
    checked_nodes = 0
    for pair in args.node_check:
        cfg_text, sep, dump_text = pair.partition("=")
        if not sep:
            raise SystemExit(f"--node-check wants CONFIG=DUMP, got {pair!r}")
        node_known = parse_node_param_list(Path(dump_text).read_text(encoding="utf-8"))
        node_ours = config_node_keys(cfg_text)
        checked_nodes += len(node_ours)
        node_bad = unknown_keys(node_ours, node_known)
        if node_bad:
            failed = True
            print(f"UNKNOWN node parameters ({len(node_bad)}) in {cfg_text}:")
            for name in node_bad:
                print(f"  {name}")
    if failed:
        return 1
    extra = f", {checked_nodes} node parameters" if args.node_check else ""
    print(f"OK: {len(ours)} library parameters{extra} all known to this RTAB-Map ({len(known)} available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
