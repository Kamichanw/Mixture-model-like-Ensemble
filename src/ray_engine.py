import random
from typing import Any, Optional

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from .token_mapping import TokenMapper
from .utils import TokenTimer


def _quantization_config(name):
    if name == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if name == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    return None


def _normalize_weights(models):
    weights = torch.tensor([float(model.get("weight", 1.0)) for model in models])
    return (weights / weights.sum()).tolist()


def _sample_from_probs(probs, gen_cfg):
    probs = probs.float()
    if not gen_cfg.get("do_sample", False) or gen_cfg.get("temperature", 1.0) <= 0:
        return int(torch.argmax(probs).item())

    temperature = float(gen_cfg.get("temperature", 1.0))
    if temperature != 1.0:
        probs = torch.pow(probs.clamp_min(1e-12), 1.0 / temperature)
        probs = probs / probs.sum()

    top_k = int(gen_cfg.get("top_k", 0) or 0)
    if top_k > 0 and top_k < probs.numel():
        values, indices = torch.topk(probs, top_k)
        filtered = torch.zeros_like(probs)
        filtered[indices] = values
        probs = filtered / filtered.sum()

    top_p = float(gen_cfg.get("top_p", 1.0) or 1.0)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=0) - sorted_probs <= top_p
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


def _weighted_original_vocab_probs(indexed_probs, weights, top_k):
    merged = torch.zeros_like(indexed_probs[0][1])
    for model_idx, probs in indexed_probs:
        merged += weights[model_idx] * probs

    if top_k > 0 and len(indexed_probs) > 1:
        candidate_ids = set()
        for _, probs in indexed_probs:
            candidate_ids.update(
                torch.topk(probs, min(top_k, probs.numel())).indices.tolist()
            )
        filtered = torch.zeros_like(merged)
        candidate_ids = list(candidate_ids)
        filtered[candidate_ids] = merged[candidate_ids]
        merged = filtered / filtered.sum()

    return merged


class ModelRunner:
    def __init__(self, model_cfg):
        self.name = model_cfg["name"]
        self.path = model_cfg["path"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        quantization = _quantization_config(model_cfg.get("quantization", "none"))
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            self.path, padding_side="left", trust_remote_code=True
        )
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "trust_remote_code": True,
            "quantization_config": quantization,
        }
        if quantization is not None:
            model_kwargs["device_map"] = "auto"
        self.model: Any = AutoModelForCausalLM.from_pretrained(self.path, **model_kwargs)
        if quantization is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_ids: Any = None
        self.pending_input_ids: Any = None
        self.past_key_values: Any = None
        self.prompt_len = 0

    def reset(self, prompt):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        self.input_ids = encoded.input_ids.to(self.device)
        self.pending_input_ids = self.input_ids
        self.past_key_values = None
        self.prompt_len = self.input_ids.shape[1]

    def next_probs(self):
        with torch.no_grad():
            outputs = self.model(
                self.pending_input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        self.past_key_values = outputs.past_key_values
        self.pending_input_ids = None
        return torch.softmax(outputs.logits[:, -1, :], dim=-1)[0].detach().cpu()

    def append(self, token_ids):
        new_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        self.input_ids = torch.cat([self.input_ids, new_ids], dim=1)
        if self.pending_input_ids is None:
            self.pending_input_ids = new_ids
        else:
            # ME only forwards the sampled model; others must keep skipped tokens pending.
            self.pending_input_ids = torch.cat([self.pending_input_ids, new_ids], dim=1)

    def generated_text(self):
        generated = self.input_ids[0, self.prompt_len :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def is_eos(self, token_ids):
        eos = self.tokenizer.eos_token_id
        return eos is not None and eos in token_ids

    def tokenizer_info(self):
        return {
            "name": self.name,
            "path": self.path,
            "vocab": self.tokenizer.get_vocab(),
            "vocab_size": self.model.config.vocab_size,
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
        }


class ModelWorker:
    def __init__(self, model_cfg):
        self.runner = ModelRunner(model_cfg)

    def reset(self, prompt):
        self.runner.reset(prompt)

    def next_probs(self):
        return self.runner.next_probs()

    def append(self, token_ids):
        self.runner.append(token_ids)

    def generated_text(self):
        return self.runner.generated_text()

    def is_eos(self, token_ids):
        return self.runner.is_eos(token_ids)

    def tokenizer_info(self):
        return self.runner.tokenizer_info()


class ModelGroupWorker:
    def __init__(self, model_cfgs):
        self.runners = [ModelRunner(cfg) for cfg in model_cfgs]

    def reset(self, prompt):
        for runner in self.runners:
            runner.reset(prompt)

    def next_probs(self, index: Optional[int] = None):
        if index is not None:
            return [self.runners[index].next_probs()]
        return [runner.next_probs() for runner in self.runners]

    def append_all(self, token_ids_by_model):
        for runner, token_ids in zip(self.runners, token_ids_by_model):
            runner.append(token_ids)

    def generated_text(self):
        return self.runners[0].generated_text()

    def any_eos(self, token_ids_by_model):
        return any(
            runner.is_eos(token_ids)
            for runner, token_ids in zip(self.runners, token_ids_by_model)
        )

    def tokenizer_info(self):
        return [runner.tokenizer_info() for runner in self.runners]


class RayGenerationEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.method = cfg["method"]
        self.models = cfg["models"]
        self.active_models = self.models[:1] if self.method == "single" else self.models
        self.weights = _normalize_weights(self.active_models)
        self.generation = cfg["generation"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._rng = random.Random(self.generation.get("seed", 42))
        self.timer = TokenTimer()
        self.group: Any = None
        self.workers: Any = None

        if not ray.is_initialized():
            ray.init(local_mode=cfg["ray"].get("local_mode", False), ignore_reinit_error=True)

        self._build_workers()
        self.tokenizer_infos = self._tokenizer_infos()
        self.tokenizer_name = self.tokenizer_infos[0]["name"]
        self.mapper = TokenMapper(self.tokenizer_infos)

    def _build_workers(self):
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        num_gpus = self.cfg["ray"].get("num_gpus_per_model", 1) if visible_gpus else 0
        use_group = self.method in {"single", "ce_single"} or visible_gpus < len(self.active_models)

        if use_group:
            worker_cls = ray.remote(num_gpus=num_gpus)(ModelGroupWorker)
            self.group = worker_cls.remote(self.active_models)
            self.workers = None
            return

        worker_cls = ray.remote(num_gpus=num_gpus)(ModelWorker)
        self.workers = [worker_cls.remote(model) for model in self.active_models]
        self.group = None

    def _tokenizer_infos(self):
        if self.group is not None:
            return ray.get(self.group.tokenizer_info.remote())
        return ray.get([worker.tokenizer_info.remote() for worker in self.workers])

    def _reset(self, prompt):
        if self.group is not None:
            ray.get(self.group.reset.remote(prompt))
        else:
            ray.get([worker.reset.remote(prompt) for worker in self.workers])

    def _next_probs(self, selected_index=None):
        if self.group is not None:
            probs = ray.get(self.group.next_probs.remote(selected_index))
            if selected_index is not None:
                return [(selected_index, probs[0])]
            return list(enumerate(probs))
        if selected_index is not None:
            return [(selected_index, ray.get(self.workers[selected_index].next_probs.remote()))]
        probs = ray.get([worker.next_probs.remote() for worker in self.workers])
        return list(enumerate(probs))

    def _append_all(self, token_ids_by_model):
        if self.group is not None:
            ray.get(self.group.append_all.remote(token_ids_by_model))
        else:
            ray.get(
                [
                    worker.append.remote(token_ids)
                    for worker, token_ids in zip(self.workers, token_ids_by_model)
                ]
            )

    def _generated_text(self):
        if self.group is not None:
            return ray.get(self.group.generated_text.remote())
        return ray.get(self.workers[0].generated_text.remote())

    def _any_eos(self, token_ids_by_model):
        if self.group is not None:
            return ray.get(self.group.any_eos.remote(token_ids_by_model))
        return any(
            ray.get(
                [
                    worker.is_eos.remote(token_ids)
                    for worker, token_ids in zip(self.workers, token_ids_by_model)
                ]
            )
        )

    def _select_token(self, indexed_probs, gen_cfg):
        if self.method == "me":
            model_idx, probs = indexed_probs[0]
            token_id = _sample_from_probs(probs, gen_cfg)
            return self.mapper.convert_model_token(model_idx, token_id)

        top_k = int(gen_cfg.get("top_k", 0) or 0)
        if self.mapper.same_tokenizer:
            merged = _weighted_original_vocab_probs(indexed_probs, self.weights, top_k)
            token_id = _sample_from_probs(merged, gen_cfg)
            return [[token_id] for _ in self.active_models]

        if top_k > 0 and len(indexed_probs) > 1:
            # UniTE-style ensembling aligns only the union of each model's top-k tokens.
            union_probs = self.mapper.merge_topk_probs(indexed_probs, self.weights, top_k)
            union_id = _sample_from_probs(union_probs, gen_cfg)
            return self.mapper.convert_union_token(union_id)

        union_probs = self.mapper.merge_probs(indexed_probs, self.weights)
        union_id = _sample_from_probs(union_probs, gen_cfg)
        return self.mapper.convert_union_token(union_id)

    def generate(self, prompt, gen_kwargs):
        gen_cfg = {**self.generation, **gen_kwargs}
        max_new_tokens = int(
            gen_cfg.get("max_gen_toks")
            or gen_cfg.get("max_new_tokens")
            or self.generation.get("max_gen_toks", 128)
        )
        until = gen_cfg.get("until") or []
        if isinstance(until, str):
            until = [until]

        self._reset(prompt)
        with self.timer:
            for _ in range(max_new_tokens):
                selected = None
                if self.method == "me":
                    selected = self._rng.choices(
                        range(len(self.active_models)), self.weights, k=1
                    )[0]
                indexed_probs = self._next_probs(selected)
                token_ids_by_model = self._select_token(indexed_probs, gen_cfg)
                self._append_all(token_ids_by_model)
                # Throughput follows the same model whose text is returned.
                self.timer.add_tokens(len(token_ids_by_model[0]))

                text = self._generated_text()
                if self._any_eos(token_ids_by_model) or any(stop in text for stop in until):
                    break

        text = self._generated_text()
        for stop in until:
            if stop in text:
                text = text.split(stop, 1)[0]
        return text

    def shutdown(self):
        if self.group is not None:
            ray.kill(self.group)
        if self.workers is not None:
            for worker in self.workers:
                ray.kill(worker)
