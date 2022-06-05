"""Darknet config and training."""
import hashlib
import os
import random
import subprocess
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import List, Optional, Union

import coolname
from omegaconf import DictConfig

from src import utils

log = utils.get_logger(__name__)


LOCAL_USER = [
    "-e",
    f"LOCAL_USER_ID={os.getuid()}",
    "-e",
    f"LOCAL_GROUP_ID={os.getgid()}",
]


def get_coolname(text: str) -> str:
    """Get repeatable coolname for given text."""
    coolname.replace_random(random.Random(hashlib.md5(text.encode()).digest()))
    return coolname.generate_slug(2)


def prepare_config(
    data_dir: str,
    template: str,
    batch: int,
    subdivisions: int,
    max_batches: int,
    size: int,
    n_classes: int,
    results_dir: str = "yolo/results",
) -> Path:
    """Prepare yolo config given training parameters."""
    template_path = Path(template)
    model_version = template_path.stem.split(".")[0]
    with template_path.open("r") as fp:
        config = fp.read().format(
            batch=batch,
            n_classes=n_classes,
            yolo_filters=(n_classes + 5) * 3,
            subdivisions=subdivisions,
            width=size,
            height=size,
            max_batches=max_batches,
            steps=f"{int(0.8*max_batches)},{int(0.9*max_batches)}",
        )

    name = get_coolname(config)
    data_path = Path(data_dir)
    config_filename = f"{data_path.stem}-{model_version}-{name}.cfg"
    config_path = data_path.parent.joinpath(results_dir).joinpath(config_filename)

    with config_path.open("w") as fp:
        fp.writelines(config)
    return config_path


def run_command(cmd: Union[str, List[str]], **kwargs):
    """Run command in shell."""
    log.info("Running command %s", cmd)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError("Command %s failed", cmd)


@contextmanager
def tmpfs_volume():
    """Create tmpfs volume and take care of removing it."""
    name = str(uuid.uuid4())[:8]
    opts = ["--opt", "type=tmpfs", "--opt", "device=tmpfs", "--opt", "o=,nodev,nosuid"]
    opts.extend(["--name", name])
    try:
        run_command(["docker", "volume", "create", *opts])
        yield name
    finally:
        run_command(["docker", "volume", "rm", name])


@contextmanager
def docker_container(
    image: str,
    run_args: Optional[List[str]] = None,
    gpu: int = 0,
    name: str = "dev",
):
    """Run docker container and take care of closing and removing it."""
    with ExitStack() as stack:
        try:
            name = f"{name}-gpu-{gpu}-{str(uuid.uuid4())[:8]}"
            run_args = run_args or []
            run_args.extend(["--gpus", f'"device={gpu}"', "--ipc", "host"])
            run_args.extend(["-d", "--rm", "--name", name, *LOCAL_USER])
            run_command(["docker", "run", *run_args, image, "sleep", "infinity"])
            yield name
        finally:
            run_command(["docker", "kill", name])


def yolo(config: DictConfig) -> str:
    """Train YOLOv4 model in Docker container."""
    prepared = prepare_config(
        data_dir=config.darknet.training.data_dir,
        template=config.darknet.training.cfg,
        batch=config.darknet.training.batch,
        subdivisions=config.darknet.training.subdivisions,
        max_batches=config.darknet.training.max_batches,
        size=config.darknet.training.size,
        n_classes=config.darknet.training.n_classes,
    )

    with tmpfs_volume() as volume:
        with docker_container(
            image=config.darknet.docker.image,
            run_args=["-v", f"{volume}:/home/user/darknet/data", "-P"],
            gpu=config.darknet.docker.gpu_id,
        ) as container:
            cmd = [
                "bash",
                "-c",
                "cd /home/user/darknet && "
                "./darknet detector train "
                "data/obj.data data/model.cfg data/yolov4.pretrained "
                "-dont_show -mjpeg_port 8090",
            ]

            run_command(
                f"tar -ch -C {config.darknet.training.data_dir} . | pv |"
                f" docker exec -i {container} tar -xf - -C /home/user/darknet/data/",
                shell=True,
            )
            run_command(
                [
                    "docker",
                    "cp",
                    "-L",
                    f"{prepared}",
                    f"{container}:/home/user/darknet/data/model.cfg",
                ]
            )
            run_command(
                [
                    "docker",
                    "cp",
                    "-L",
                    f"{config.darknet.training.pretrained}",
                    f"{container}:/home/user/darknet/data/yolov4.pretrained",
                ]
            )

            run_command(["docker", "exec", "-it", container, *cmd])

            trained = prepared.with_suffix(".weights")
            run_command(
                [
                    "docker",
                    "cp",
                    f"{container}:/home/user/darknet/data/model_final.weights",
                    trained,
                ]
            )
    return str(trained)
