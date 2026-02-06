import argparse
import os
import shutil
import subprocess
import sys


DEFAULT_IMAGE = os.environ.get("CODERUNNER_IMAGE", "coderunner-claude")
DEFAULT_CONTAINER = os.environ.get("CODERUNNER_CLAUDE_CONTAINER", "coderunner-claude")


def _container_available() -> bool:
    return shutil.which("container") is not None


def _run_claude(args: argparse.Namespace, passthrough: list[str]) -> int:
    if not _container_available():
        print("Error: Apple 'container' CLI not found. Run ./install.sh to install it.", file=sys.stderr)
        return 1

    workdir = os.path.abspath(args.workdir)
    cmd = [
        "container",
        "run",
        "--rm",
        "-i",
        "-t",
        "--entrypoint",
        "claude",
        "--name",
        args.container,
        "-e",
        "ANTHROPIC_API_KEY",
    ]

    if not args.no_mount:
        cmd += ["--volume", f"{workdir}:/workspace", "--workdir", "/workspace"]

    cmd.append(args.image)
    cmd.extend(passthrough)
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(prog="coderunner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    claude_parser = subparsers.add_parser(
        "claude",
        help="Run Claude Code inside an Apple container",
    )
    claude_parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Container image (default: {DEFAULT_IMAGE})",
    )
    claude_parser.add_argument(
        "--container",
        default=DEFAULT_CONTAINER,
        help=f"Container name (default: {DEFAULT_CONTAINER})",
    )
    claude_parser.add_argument(
        "--workdir",
        default=os.getcwd(),
        help="Host directory to mount into /workspace",
    )
    claude_parser.add_argument(
        "--no-mount",
        action="store_true",
        help="Do not mount the host working directory",
    )

    args, passthrough = parser.parse_known_args()

    if args.command == "claude":
        raise SystemExit(_run_claude(args, passthrough))


if __name__ == "__main__":
    main()
