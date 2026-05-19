import json
import os
import random
from pprint import pprint
from typing import cast

import hydra
import lm_eval
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable
from lm_eval.evaluator import simple_evaluate
from omegaconf import DictConfig, OmegaConf

from src.lm import MELM
from src.ray_engine import RayGenerationEngine

load_dotenv()


def _resolve_models(model_entries, config_dir):
    models = []
    for entry in model_entries:
        if isinstance(entry, str):
            model_cfg = OmegaConf.load(os.path.join(config_dir, f"{entry}.yaml"))
            models.append(OmegaConf.to_container(model_cfg, resolve=True))
        else:
            models.append(OmegaConf.to_container(entry, resolve=True))
    return models


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    seed = cfg.generation.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    project_root = get_original_cwd()
    config_root = os.path.join(project_root, "configs")
    cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg.eval.tasks, str):
        tasks = [item.strip() for item in cfg.eval.tasks.split(",") if item.strip()]
    else:
        tasks = list(cfg.eval.tasks)
    cfg_dict["models"] = _resolve_models(cfg.models, os.path.join(config_root, "model"))
    include_path = os.path.join(project_root, "tasks")

    engine = RayGenerationEngine(cfg_dict)
    lm = MELM(engine)
    task_manager = TaskManager(include_path=include_path)

    results = cast(
        dict,
        simple_evaluate(
            model=lm,
            tasks=tasks,  # type: ignore
            num_fewshot=cfg.eval.num_fewshot,
            limit=cfg.eval.limit,
            task_manager=task_manager,
            log_samples=cfg.eval.log_samples,
            gen_kwargs=cfg_dict["generation"],
        ),
    )
    results["throughput"] = engine.timer.to_dict()
    pprint(results.get("results", results))

    output_path = os.path.join(HydraConfig.get().runtime.output_dir, "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            default=handle_non_serializable,
        )

    engine.shutdown()


if __name__ == "__main__":
    main()
