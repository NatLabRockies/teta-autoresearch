import optuna


def build(seed: int = 42) -> optuna.samplers.BaseSampler:
    return optuna.samplers.RandomSampler(seed=seed)
