from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from pyboy import PyBoy


DEFAULT_SAVE_PATH = Path("pyboy_states/manual.state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Pokemon Red ROM manually in PyBoy and save emulator states."
    )
    parser.add_argument(
        "rom_path",
        type=Path,
        help="Path to the Pokemon Red ROM, e.g. train_skill_red.gb.",
    )
    parser.add_argument(
        "--save-state",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help=f"Path to write the saved PyBoy state. Default: {DEFAULT_SAVE_PATH}",
    )
    parser.add_argument(
        "--load-state",
        type=Path,
        default=None,
        help="Optional state file to load before manual control starts.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Emulation speed multiplier. Use 1.0 for normal speed.",
    )
    return parser.parse_args()


def start_command_reader() -> queue.SimpleQueue[str]:
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()

    def _reader():
        while True:
            try:
                line = input().strip().lower()
            except EOFError:
                break
            if line:
                commands.put(line)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return commands


def normalize_save_path(user_input: str, default_save_dir: Path) -> Path:
    candidate = Path(user_input).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".state")
    if candidate.parent == Path("."):
        candidate = default_save_dir / candidate.name
    return candidate


def save_state(pyboy: PyBoy, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as state_file:
        pyboy.save_state(state_file)
    print(f"saved state: {save_path}")


def load_state(pyboy: PyBoy, load_path: Path) -> None:
    with load_path.open("rb") as state_file:
        pyboy.load_state(state_file)
    print(f"loaded state: {load_path}")


def main() -> None:
    args = parse_args()
    if not args.rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {args.rom_path}")
    if args.load_state is not None and not args.load_state.exists():
        raise FileNotFoundError(f"State not found: {args.load_state}")

    pyboy = PyBoy(str(args.rom_path), window="SDL2")
    pyboy.set_emulation_speed(args.speed)

    if args.load_state is not None:
        load_state(pyboy, args.load_state)

    print("Manual PyBoy control started.")
    print("Controls:")
    print("- gameplay: use the PyBoy window")
    print("- save current default file: save")
    print("- save to custom name: save <filename>  (e.g. save cut_ready)")
    print("- quit: quit")
    print(f"Default save target: {args.save_state}")
    command_queue = start_command_reader()
    default_save_dir = args.save_state.parent if args.save_state.parent != Path("") else Path(".")

    try:
        while pyboy.tick():
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = ""

            if command:
                parts = command.split(maxsplit=1)
                action = parts[0]
                arg = parts[1] if len(parts) > 1 else ""

                if action in {"q", "quit", "exit"}:
                    break

                if action in {"save", "s"}:
                    if arg:
                        save_path = normalize_save_path(arg, default_save_dir)
                    else:
                        save_path = args.save_state
                    save_state(pyboy, save_path)
                else:
                    print("unknown command. use: save [filename] | quit")

            time.sleep(0.001)
    finally:
        pyboy.stop()


if __name__ == "__main__":
    main()
