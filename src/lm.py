from lm_eval.api.model import LM
from tqdm.auto import tqdm


class MELM(LM):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._device = engine.device

    @property
    def tokenizer_name(self):
        return self.engine.tokenizer_name

    def loglikelihood(self, requests):
        raise NotImplementedError("This release only supports generate_until tasks.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This release only supports generate_until tasks.")

    def generate_until(self, requests, disable_tqdm=False):
        outputs = []
        progress = tqdm(
            requests,
            desc="Generating",
            disable=disable_tqdm or self.rank != 0,
            dynamic_ncols=True,
        )
        for request in progress:
            context, gen_kwargs = request.args
            outputs.append(self.engine.generate(context, gen_kwargs))
        return outputs
