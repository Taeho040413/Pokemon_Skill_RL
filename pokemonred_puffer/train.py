import functools
import importlib
import os
import re
import sqlite3
import sys
import warnings
from tempfile import NamedTemporaryFile
import time
from contextlib import contextmanager, nullcontext
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Callable

import gymnasium
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import typer
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch import nn

# torch.compile / Inductor: harmless upstream noise during graph compile.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\._dynamo\.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.fx\.interpreter",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Online softmax is disabled on the fly.*",
    category=UserWarning,
    module=r"torch\._inductor\.lowering",
)

import wandb
from pokemonred_puffer import cleanrl_puffer
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.wrappers.async_io import AsyncWrapper
from pokemonred_puffer.wrappers.sqlite import SqliteStateResetWrapper

app = typer.Typer(pretty_exceptions_enable=False)

DEFAULT_CONFIG = "config.yaml"
DEFAULT_POLICY = "multi_convolutional.MultiConvolutionalPolicy"
DEFAULT_REWARD = "baseline.ObjectRewardRequiredEventsMapIdsFieldMoves"
DEFAULT_WRAPPER = "baseline"
DEFAULT_ROM = "train_teach_skill.gb"
# Default run folder: runs/<DEFAULT_EXP_ID>/ (model_*.pt, trainer_state.pt). Override with --exp-name / -e.
DEFAULT_EXP_ID = "pokeSkill001"


class Vectorization(Enum):
    multiprocessing = "multiprocessing"
    serial = "serial"
    ray = "ray"


def make_policy(env: RedGymEnv, policy_name: str, config: DictConfig) -> nn.Module:
    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    policy = policy_class(env, **config.policies[policy_name].policy)
    if config.train.use_rnn:
        rnn_config = config.policies[policy_name].rnn
        policy_class = getattr(policy_module, rnn_config.name)
        policy = policy_class(env, policy, **rnn_config.args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(config.train.device)


def load_from_config(config: DictConfig, debug: bool) -> DictConfig:
    default_keys = ["env", "train", "policies", "rewards", "wrappers", "wandb"]
    defaults = OmegaConf.create({key: config.get(key, {}) for key in default_keys})

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", OmegaConf.create({})) if debug else OmegaConf.create({})

    defaults.merge_with(debug_config)
    return defaults


# Applied to train.* then removed so CleanPuffeRL never sees them.
_TRAIN_DEVICE_PROFILE_KEYS = ("auto_device", "cuda", "cpu")
_CUDA_BACKEND_KEYS = (
    "cudnn_benchmark",
    "cudnn_deterministic",
    "matmul_allow_tf32",
    "cudnn_allow_tf32",
)


def _strip_train_profile_blocks(train: DictConfig) -> None:
    """Remove cuda/cpu profile sections so trainers never see nested train.cuda / train.cpu."""
    with open_dict(train):
        for key in _TRAIN_DEVICE_PROFILE_KEYS:
            if key in train:
                del train[key]


def _log_device_selection(
    train: DictConfig,
    *,
    mode: str,
    profile_key: str | None = None,
    cuda_available: bool | None = None,
    gpu_name: str | None = None,
) -> None:
    """Single-line stdout log for how train.device and vec knobs were chosen."""
    dev = train.get("device", "?")
    parts = [f"[train] device: {mode}", f"train.device={dev}"]
    if cuda_available is not None:
        parts.append(f"torch.cuda.is_available()={cuda_available}")
    if profile_key is not None:
        parts.append(f"profile=train.{profile_key}")
    if gpu_name:
        parts.append(f"gpu={gpu_name!r}")
    for key in ("num_envs", "num_workers", "env_batch_size"):
        if key in train and train.get(key) is not None:
            parts.append(f"{key}={train[key]}")
    print(" | ".join(parts))


def apply_train_device_profile(config: DictConfig) -> None:
    """If train.auto_device is true, merge train.cuda or train.cpu and set torch backends."""
    train = config.train
    cuda_available = torch.cuda.is_available()

    if not train.get("auto_device", True):
        _strip_train_profile_blocks(train)
        _log_device_selection(
            train,
            mode="manual (auto_device=false)",
            cuda_available=cuda_available,
        )
        return

    use_cuda = cuda_available
    profile_key = "cuda" if use_cuda else "cpu"
    profile = train.get(profile_key)
    profile_dict: dict = {}
    if profile is not None:
        profile_dict = OmegaConf.to_container(profile, resolve=True) or {}
        if not isinstance(profile_dict, dict):
            profile_dict = {}

    with open_dict(train):
        for key, value in profile_dict.items():
            if value is not None:
                train[key] = value

        default_dev = "cuda" if use_cuda else "cpu"
        dev = profile_dict.get("device")
        train.device = default_dev if dev is None else str(dev)
        if use_cuda and train.device == "cpu":
            train.device = "cuda"

        _strip_train_profile_blocks(train)

    if use_cuda:
        if train.get("cudnn_benchmark") is not None:
            torch.backends.cudnn.benchmark = bool(train.cudnn_benchmark)
        if train.get("cudnn_deterministic") is not None:
            torch.backends.cudnn.deterministic = bool(train.cudnn_deterministic)
        if train.get("matmul_allow_tf32") is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(train.matmul_allow_tf32)
        if train.get("cudnn_allow_tf32") is not None:
            torch.backends.cudnn.allow_tf32 = bool(train.cudnn_allow_tf32)

    with open_dict(train):
        for key in _CUDA_BACKEND_KEYS:
            if key in train:
                del train[key]

    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    _log_device_selection(
        train,
        mode="auto",
        profile_key=profile_key,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def make_env_creator(
    wrapper_classes: list[tuple[str, ModuleType]],
    reward_class: RedGymEnv,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]:
    def env_creator(
        env_config: DictConfig,
        wrappers_config: list[dict[str, Any]],
        reward_config: DictConfig,
        async_config: dict[str, Queue] | None = None,
        sqlite_config: dict[str, str] | None = None,
    ) -> pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env:
        env = reward_class(env_config, reward_config)
        for cfg, (_, wrapper_class) in zip(wrappers_config, wrapper_classes):
            env = wrapper_class(env, OmegaConf.create([x for x in cfg.values()][0]))
        if async_wrapper and async_config:
            env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])
        if sqlite_wrapper and sqlite_config:
            env = SqliteStateResetWrapper(env, sqlite_config["database"])
        if puffer_wrapper:
            env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
        return env

    return env_creator


def setup_agent(
    wrappers: list[str],
    reward_name: str,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes = [
        (
            k,
            getattr(
                importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
                k.split(".")[1],
            ),
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    ]
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(f"pokemonred_puffer.rewards.{reward_module}"), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(
        wrapper_classes, reward_class, async_wrapper, sqlite_wrapper, puffer_wrapper
    )

    return env_creator


@contextmanager
def init_wandb(
    config: DictConfig,
    exp_name: str | None,
    reward_name: str,
    policy_name: str,
    wrappers_name: str,
    policy_checkpoint: dict[str, Any] | None = None,
):
    if not config.track:
        yield None
    else:
        assert config.wandb.project is not None, "Please set the wandb project in config.yaml"
        assert config.wandb.entity is not None, "Please set the wandb entity in config.yaml"
        run_label = exp_name if exp_name else config.train.exp_id
        # W&B run id: new random id each launch so every run appears separately (not exp_id / pokeSkill001).
        wandb_run_id = wandb.util.generate_id()
        run_config: dict[str, Any] = {
            "cleanrl": config.train,
            "env": config.env,
            "reward_module": reward_name,
            "policy_module": policy_name,
            "reward": config.rewards[reward_name],
            "policy": config.policies[policy_name],
            "wrappers": config.wrappers[wrappers_name],
            "rnn": "rnn" in config.policies[policy_name],
            "checkpoint_storage_exp_id": config.train.exp_id,
            "wandb_run_id": wandb_run_id,
        }
        if policy_checkpoint is not None:
            run_config["policy_checkpoint"] = policy_checkpoint
        wandb_kwargs = {
            "id": wandb_run_id,
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "group": config.wandb.group,
            "config": run_config,
            "name": f"{run_label}-{wandb_run_id}",
            "monitor_gym": True,
            "save_code": True,
            "resume": False,
        }
        base_url = config.wandb.get("base_url")
        if base_url:
            wandb_kwargs["settings"] = wandb.Settings(base_url=str(base_url))
        client = wandb.init(**wandb_kwargs)
        try:
            client.summary["wandb_run_id"] = wandb_run_id
            client.summary["checkpoint_storage_exp_id"] = config.train.exp_id
            if policy_checkpoint is not None:
                client.summary["checkpoint_status"] = policy_checkpoint["status"]
                client.summary["policy_checkpoint"] = policy_checkpoint
        except Exception as exc:
            print(
                f"[checkpoint] could not write wandb summary: {exc}",
                file=sys.stderr,
                flush=True,
            )
        yield client
        client.finish()


def setup(
    config: DictConfig,
    debug: bool,
    wrappers_name: str,
    reward_name: str,
    rom_path: Path,
    track: bool,
    puffer_wrapper: bool = True,
    run_id: str | None = None,
) -> tuple[DictConfig, Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]]:
    # debug merge often pins CPU/small envs; skip auto CUDA so debug.train stays authoritative.
    if not debug:
        apply_train_device_profile(config)
    else:
        _strip_train_profile_blocks(config.train)
        _log_device_selection(
            config.train,
            mode="debug (profiles not merged)",
            cuda_available=torch.cuda.is_available(),
        )
    rid = (run_id or "").strip()
    config.train.exp_id = rid if rid else DEFAULT_EXP_ID
    config.env.gb_path = rom_path
    config.track = track
    if debug:
        config.vectorization = Vectorization.serial

    async_wrapper = config.train.get("async_wrapper", False)
    sqlite_wrapper = config.train.get("sqlite_wrapper", False)
    env_creator = setup_agent(
        config.wrappers[wrappers_name], reward_name, async_wrapper, sqlite_wrapper, puffer_wrapper
    )
    # So torch.compile / Inductor see TF32 matmul policy (matches config.yaml train.float32_matmul_precision).
    matmul_prec = config.train.get("float32_matmul_precision")
    if matmul_prec:
        torch.set_float32_matmul_precision(str(matmul_prec))
    return config, env_creator


def find_latest_model_path(data_dir: str | Path, exp_id: str) -> Path | None:
    """Latest `model_*.pt` under ``data_dir / exp_id`` (checkpoints stay per run)."""
    run_dir = Path(data_dir) / exp_id
    if not run_dir.is_dir():
        return None

    model_paths = [path for path in run_dir.glob("model_*.pt") if path.is_file()]
    if not model_paths:
        return None

    return max(model_paths, key=lambda path: path.stat().st_mtime)


def _torch_load_module_checkpoint(path: str | Path, map_location: str | torch.device):
    """Saved files are full ``nn.Module`` pickles, not ``state_dict``. PyTorch 2.6+ defaults ``weights_only=True``."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _report_policy_checkpoint(run_dir: Path, detail: str) -> None:
    """W&B / Rich often hide stdout; stderr + a small file in the run dir stay inspectable."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    block = f"[checkpoint] {stamp}\n{detail}\n"
    banner = "\n" + "=" * 72 + "\n" + block + "=" * 72 + "\n"
    print(banner, file=sys.stderr, flush=True)
    print(banner, flush=True)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoint_load.log").write_text(block, encoding="utf-8")
    except OSError as exc:
        print(
            f"[checkpoint] could not write {run_dir / 'checkpoint_load.log'}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def find_newest_saved_model(data_dir: str | Path) -> Path | None:
    """Newest `model_*.pt` anywhere under ``data_dir`` (e.g. all runs under `runs/`)."""
    root = Path(data_dir)
    if not root.is_dir():
        return None

    candidates = [path for path in root.rglob("model_*.pt") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_explicit_checkpoint_path(checkpoint_path: str | Path) -> Path | None:
    """Return a resolved path if the checkpoint exists.

    On Linux/WSL, a Windows absolute path like ``C:\\Users\\...`` is not understood by
    :class:`pathlib.Path` the same way as on Windows; it is treated as a relative path and
    gets joined with the current working directory. Map ``X:/...`` to ``/mnt/x/...`` when
    needed so ``--checkpoint`` works from WSL shells.
    """
    raw = os.fspath(checkpoint_path).strip().strip('"').strip("'")
    normalized = raw.replace("\\", "/")
    try_paths: list[Path] = [Path(raw)]
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):(?:/(.*))?$", normalized)
        if match:
            drive = match.group(1).lower()
            rest = (match.group(2) or "").strip("/")
            if rest:
                try_paths.append(Path(f"/mnt/{drive}/{rest}"))
    for candidate in try_paths:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_checkpoint_for_rollout(
    *,
    data_dir: str | Path,
    checkpoint_path: Path | None,
    exp_name: str | None,
) -> Path:
    """Pick a checkpoint: explicit path, latest under one run, or newest under data_dir."""
    if checkpoint_path is not None:
        path = _resolve_explicit_checkpoint_path(checkpoint_path)
        if path is None:
            raw = os.fspath(checkpoint_path).strip().strip('"').strip("'")
            tried = [str(Path(raw).resolve())]
            if os.name != "nt":
                norm = raw.replace("\\", "/")
                m = re.match(r"^([A-Za-z]):(?:/(.*))?$", norm)
                if m:
                    drive, rest = m.group(1).lower(), (m.group(2) or "").strip("/")
                    if rest:
                        tried.append(str((Path(f"/mnt/{drive}") / rest).resolve()))
            raise typer.BadParameter(
                "Checkpoint not found. Tried:\n  " + "\n  ".join(tried)
            )
        return path

    exp = (exp_name or "").strip()
    if exp:
        found = find_latest_model_path(data_dir, exp)
        if found is None:
            raise typer.BadParameter(
                f"No model_*.pt under {(Path(data_dir) / exp).resolve()} (exp_name={exp!r})"
            )
        print(f"[evaluate] Using latest checkpoint for run {exp!r}: {found.resolve()}")
        return found

    found = find_newest_saved_model(data_dir)
    if found is None:
        raise typer.BadParameter(
            f"No model_*.pt under {Path(data_dir).resolve()}. Train with save_checkpoint or pass "
            f"--checkpoint / --exp-name."
        )
    print(f"[evaluate] Using newest checkpoint (all runs): {found.resolve()}")
    return found


def _apply_rollout_display_config(config: DictConfig, headless: bool) -> None:
    """Interactive rollout: PyBoy SDL2 window + OpenCV preview need env.headless False."""
    with open_dict(config.env):
        config.env.headless = headless


def load_latest_policy_if_available(
    policy: nn.Module, config: DictConfig
) -> tuple[nn.Module, dict[str, Any]]:
    run_dir = Path(config.train.data_dir) / config.train.exp_id
    latest_model_path = find_latest_model_path(config.train.data_dir, config.train.exp_id)
    exp_id = config.train.exp_id
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    base_info: dict[str, Any] = {
        "logged_at": stamp,
        "exp_id": exp_id,
        "run_dir": str(run_dir.resolve()),
        "model_path": None,
        "model_file": None,
        "error": None,
    }
    if latest_model_path is None:
        base_info["status"] = "none"
        _report_policy_checkpoint(
            run_dir,
            "\n".join(
                [
                    f"No model_*.pt under {run_dir.resolve()}",
                    f"exp_id={exp_id}",
                    "Action: using newly initialized policy.",
                ]
            ),
        )
        return policy, base_info

    base_info["model_path"] = str(latest_model_path.resolve())
    base_info["model_file"] = latest_model_path.name
    try:
        loaded_policy = _torch_load_module_checkpoint(
            latest_model_path, config.train.device
        )
        loaded_policy = loaded_policy.to(config.train.device)
        base_info["status"] = "loaded"
        _report_policy_checkpoint(
            run_dir,
            "\n".join(
                [
                    f"Loaded OK: {latest_model_path.resolve()}",
                    f"exp_id={exp_id}",
                    "Continuing training with these weights.",
                ]
            ),
        )
        return loaded_policy, base_info
    except Exception as exc:
        base_info["status"] = "failed"
        base_info["error"] = str(exc)[:4000]
        _report_policy_checkpoint(
            run_dir,
            "\n".join(
                [
                    f"Load failed: {latest_model_path.resolve()}",
                    f"exp_id={exp_id}",
                    f"Error: {exc!r}",
                    "Action: using newly initialized policy.",
                ]
            ),
        )
        return policy, base_info


@app.command()
def evaluate(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    checkpoint_path: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Policy weights (.pt). If omitted, uses newest under train.data_dir "
            "(or under one run when --exp-name is set).",
        ),
    ] = None,
    exp_name: Annotated[
        str | None,
        typer.Option(
            "--exp-name",
            "-e",
            help="Run folder under train.data_dir (e.g. same as train --exp-name). "
            "Latest model_*.pt in that folder is used unless --checkpoint is set.",
        ),
    ] = None,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    rom_path: Annotated[
        Path,
        typer.Argument(help="Game Boy ROM path (.gb)."),
    ] = Path(DEFAULT_ROM),
    headless: Annotated[
        bool,
        typer.Option(
            "--headless",
            help="No PyBoy window (null display). Default opens SDL2 game window for play.",
        ),
    ] = False,
):
    config = load_from_config(config, False)
    _apply_rollout_display_config(config, headless=headless)
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    model_path = resolve_checkpoint_for_rollout(
        data_dir=config.train.data_dir,
        checkpoint_path=checkpoint_path,
        exp_name=exp_name,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    try:
        cleanrl_puffer.rollout(
            env_creator,
            env_kwargs,
            model_path=model_path,
            device=config.train.device,
        )
    except KeyboardInterrupt:
        os._exit(0)


@app.command("play")
def play(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    checkpoint_path: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Policy weights (.pt). If omitted, uses newest under train.data_dir.",
        ),
    ] = None,
    exp_name: Annotated[
        str | None,
        typer.Option(
            "--exp-name",
            "-e",
            help="Run folder under train.data_dir; latest checkpoint in that folder.",
        ),
    ] = None,
    policy_name: Annotated[
        str,
        typer.Option("--policy-name", "-p", help="Policy module to use in policies."),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option("--reward-name", "-r", help="Reward module to use in rewards"),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option("--wrappers-name", "-w", help="Wrappers stack name from config."),
    ] = DEFAULT_WRAPPER,
    rom_path: Annotated[
        Path,
        typer.Argument(help="Game Boy ROM path (.gb)."),
    ] = Path(DEFAULT_ROM),
    headless: Annotated[
        bool,
        typer.Option(
            "--headless",
            help="No PyBoy window. Default: SDL2 + OpenCV windows.",
        ),
    ] = False,
):
    """Load the latest (or chosen) checkpoint and run the policy with on-screen render (no training)."""
    evaluate(
        config=config,
        checkpoint_path=checkpoint_path,
        exp_name=exp_name,
        policy_name=policy_name,
        reward_name=reward_name,
        wrappers_name=wrappers_name,
        rom_path=rom_path,
        headless=headless,
    )


@app.command()
def autotune(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = "empty",
    rom_path: Annotated[
        Path,
        typer.Argument(help="Game Boy ROM path (.gb)."),
    ] = Path(DEFAULT_ROM),
):
    config = load_from_config(config, False)
    config.vectorization = "multiprocessing"
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    pufferlib.vector.autotune(
        functools.partial(env_creator, **env_kwargs), batch_size=config.train.env_batch_size
    )


@app.command()
def debug(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = "empty",
    rom_path: Annotated[
        Path,
        typer.Argument(help="Game Boy ROM path (.gb)."),
    ] = Path(DEFAULT_ROM),
):
    config = load_from_config(config, True)
    config.env.gb_path = rom_path
    config, env_creator = setup(
        config=config,
        debug=True,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
        puffer_wrapper=False,
    )
    env = env_creator(
        config.env, config.wrappers[wrappers_name], config.rewards[reward_name]["reward"]
    )
    env.reset()
    while True:
        env.step(5)
        time.sleep(0.2)
    env.close()


@app.command()
def train(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    exp_name: Annotated[
        str | None,
        typer.Option(
            "--exp-name",
            "-e",
            help="Run id and folder name under train.data_dir (e.g. runs/<exp-name>/). "
            f"If omitted, uses {DEFAULT_EXP_ID!r}. Pass another name for a separate checkpoint directory.",
        ),
    ] = None,
    rom_path: Annotated[
        Path,
        typer.Argument(help="Game Boy ROM path (.gb)."),
    ] = Path(DEFAULT_ROM),
    track: Annotated[bool, typer.Option(help="Track on wandb.")] = False,
    debug: Annotated[bool, typer.Option(help="debug")] = False,
    vectorization: Annotated[
        Vectorization, typer.Option(help="Vectorization method")
    ] = "multiprocessing",
):
    config = load_from_config(config, debug)
    config.vectorization = vectorization
    config, env_creator = setup(
        config=config,
        debug=debug,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=track,
        run_id=exp_name,
    )
    run_root = Path(config.train.data_dir) / config.train.exp_id
    print(f"[train] exp_id={config.train.exp_id}\n[train] run directory: {run_root.resolve()}")

    vec = config.vectorization
    if vec == Vectorization.serial:
        vec = pufferlib.vector.Serial
    elif vec == Vectorization.multiprocessing:
        vec = pufferlib.vector.Multiprocessing
    elif vec == Vectorization.ray:
        vec = pufferlib.vector.Ray
    else:
        vec = pufferlib.vector.Multiprocessing

    # TODO: Remove the +1 once the driver env doesn't permanently increase the env id
    env_send_queues = []
    env_recv_queues = []
    if config.train.get("async_wrapper", False):
        env_send_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]
        env_recv_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]

    sqlite_context = nullcontext
    if config.train.get("sqlite_wrapper", False):
        sqlite_context = functools.partial(NamedTemporaryFile, suffix="sqlite")

    # Vecenv + checkpoint load BEFORE wandb.init so [checkpoint] lines are not swallowed.
    with sqlite_context() as sqlite_db:
        db_filename = None
        if config.train.get("sqlite_wrapper", False):
            db_filename = sqlite_db.name
            with sqlite3.connect(db_filename) as conn:
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE states(env_id INT PRIMARY KEY, pyboy_state BLOB, reset BOOLEAN, required_rate REAL, pid INT);"
                )

        vecenv = pufferlib.vector.make(
            env_creator,
            env_kwargs={
                "env_config": config.env,
                "wrappers_config": config.wrappers[wrappers_name],
                "reward_config": config.rewards[reward_name]["reward"],
                "async_config": {
                    "send_queues": env_send_queues,
                    "recv_queues": env_recv_queues,
                },
                "sqlite_config": {"database": db_filename},
            },
            num_envs=config.train.num_envs,
            num_workers=config.train.num_workers,
            batch_size=config.train.env_batch_size,
            zero_copy=config.train.zero_copy,
            backend=vec,
        )
        policy = make_policy(vecenv.driver_env, policy_name, config)
        policy, policy_ckpt_info = load_latest_policy_if_available(policy, config)

        with init_wandb(
            config=config,
            exp_name=exp_name,
            reward_name=reward_name,
            policy_name=policy_name,
            wrappers_name=wrappers_name,
            policy_checkpoint=policy_ckpt_info,
        ) as wandb_client:
            config.train.env = "Pokemon Red"
            with CleanPuffeRL(
                exp_name=exp_name or config.train.exp_id,
                config=config.train,
                vecenv=vecenv,
                policy=policy,
                env_recv_queues=env_recv_queues,
                env_send_queues=env_send_queues,
                sqlite_db=db_filename,
                wandb_client=wandb_client,
            ) as trainer:
                try:
                    while not trainer.done_training():
                        trainer.evaluate()
                        trainer.train()
                except KeyboardInterrupt:
                    print("KeyboardInterrupt received. Saving checkpoint and stopping training...")

        print("Done training")


if __name__ == "__main__":
    app()
