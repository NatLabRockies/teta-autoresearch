"""Search space definition.

Every suggest_* call is namespaced by model family (e.g. ``rf_max_depth``)
so Optuna's TPE surrogate never conflates parameters across families.

``sample_config(trial)`` is the single entry-point called by the objective.
It returns a plain dict that ``models.run_trial()`` can execute.
"""

import optuna

from feature_pipeline import (
    ALL_FEATURES,
    REQUIRED_FEATURES,
    SEQ_FEATURE_POOL,
    STATIC_FEATURE_POOL,
)

TABULAR_FAMILIES = ["rf", "extra_trees", "hgbr", "xgb", "lgbm", "mlp"]
SEQUENTIAL_FAMILIES = ["cnn", "gru"]
ALL_FAMILIES = TABULAR_FAMILIES + SEQUENTIAL_FAMILIES


# ---------------------------------------------------------------------------
# Per-family hyperparameter samplers
# ---------------------------------------------------------------------------


def _sample_rf(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_categorical("rf_max_depth", [None, 10, 20, 30, 40]),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "max_features": trial.suggest_float("rf_max_features", 0.3, 1.0),
        "max_samples": trial.suggest_float("rf_max_samples", 0.3, 0.8),
        "random_state": 42,
        "n_jobs": -1,
    }


def _sample_extra_trees(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("et_n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_categorical("et_max_depth", [None, 10, 20, 30, 40]),
        "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 20),
        "max_features": trial.suggest_float("et_max_features", 0.3, 1.0),
        "max_samples": trial.suggest_float("et_max_samples", 0.3, 0.8),
        "random_state": 42,
        "n_jobs": -1,
    }


def _sample_hgbr(trial: optuna.Trial) -> dict:
    return {
        "max_iter": trial.suggest_int("hgbr_max_iter", 100, 1000, step=50),
        "max_depth": trial.suggest_int("hgbr_max_depth", 3, 15),
        "learning_rate": trial.suggest_float("hgbr_lr", 0.01, 0.3, log=True),
        "l2_regularization": trial.suggest_float("hgbr_l2", 1e-4, 1.0, log=True),
        "max_leaf_nodes": trial.suggest_int("hgbr_max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("hgbr_min_samples_leaf", 5, 50),
        "random_state": 42,
    }


def _sample_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }


def _sample_lgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 2000, step=100),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 12),
        "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _sample_mlp(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int("mlp_n_layers", 1, 4)
    layer_size = trial.suggest_int("mlp_layer_size", 64, 512, step=64)
    return {
        "hidden_layer_sizes": tuple([layer_size] * n_layers),
        "activation": trial.suggest_categorical("mlp_activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("mlp_alpha", 1e-5, 0.1, log=True),
        "learning_rate_init": trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical(
            "mlp_batch_size", [256, 512, 1024, 2048]
        ),
        "max_iter": 1000,
        "early_stopping": True,
        "random_state": 42,
    }


def _sample_cnn(trial: optuna.Trial) -> dict:
    return {
        "seq_len": trial.suggest_int("cnn_seq_len", 3, 7),
        "n_layers": trial.suggest_int("cnn_n_layers", 2, 4),
        "channels": trial.suggest_categorical("cnn_channels", [64, 128, 256]),
        "kernel_size": trial.suggest_categorical("cnn_kernel_size", [3, 5]),
        "dropout": trial.suggest_float("cnn_dropout", 0.0, 0.3),
        "lr": trial.suggest_float("cnn_lr", 5e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical(
            "cnn_batch_size", [1024, 2048, 4096]
        ),
        "weight_decay": trial.suggest_float("cnn_weight_decay", 1e-5, 1e-2, log=True),
        "grad_clip": trial.suggest_float("cnn_grad_clip", 0.5, 2.0),
    }


def _sample_gru(trial: optuna.Trial) -> dict:
    return {
        "seq_len": trial.suggest_int("gru_seq_len", 3, 7),
        "hidden_size": trial.suggest_categorical("gru_hidden_size", [64, 128, 256]),
        "n_layers": trial.suggest_int("gru_n_layers", 1, 3),
        "dropout": trial.suggest_float("gru_dropout", 0.0, 0.3),
        "lr": trial.suggest_float("gru_lr", 5e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical(
            "gru_batch_size", [1024, 2048, 4096]
        ),
        "weight_decay": trial.suggest_float("gru_weight_decay", 1e-5, 1e-2, log=True),
        "grad_clip": trial.suggest_float("gru_grad_clip", 0.5, 2.0),
    }


# ---------------------------------------------------------------------------
# Feature subset samplers
# ---------------------------------------------------------------------------


def _sample_tabular_features(trial: optuna.Trial) -> list[str]:
    """Sample a subset; required features are always included."""
    optional = [f for f in ALL_FEATURES if f not in REQUIRED_FEATURES]
    selected = set(REQUIRED_FEATURES)
    for f in optional:
        if trial.suggest_categorical(f"feat_{f}", [True, False]):
            selected.add(f)
    return [f for f in ALL_FEATURES if f in selected]


def _sample_seq_features(trial: optuna.Trial) -> tuple[list[str], list[str]]:
    """Sample per-timestep and static feature subsets for sequential models."""
    seq_feats = [
        f
        for f in SEQ_FEATURE_POOL
        if trial.suggest_categorical(f"seq_feat_{f}", [True, False])
    ]
    # Guarantee at least speed + grade so the model has a useful signal
    if not seq_feats:
        seq_feats = ["speed_mph", "grade_percent"]
    static_feats = [
        f
        for f in STATIC_FEATURE_POOL
        if trial.suggest_categorical(f"static_feat_{f}", [True, False])
    ]
    return seq_feats, static_feats


# ---------------------------------------------------------------------------
# Main entry-points: joint, phase 1, and phase 2
# ---------------------------------------------------------------------------


def sample_config(trial: optuna.Trial, families: list[str] | None = None) -> dict:
    """Sample a complete experiment config for one Optuna trial (joint search)."""
    family_list = families if families is not None else ALL_FAMILIES
    family = trial.suggest_categorical("family", family_list)
    config: dict = {"family": family}

    if family in SEQUENTIAL_FAMILIES:
        seq_feats, static_feats = _sample_seq_features(trial)
        config["seq_features"] = seq_feats
        config["static_features"] = static_feats
        config["model_params"] = (
            _sample_cnn(trial) if family == "cnn" else _sample_gru(trial)
        )
    else:
        config["features"] = _sample_tabular_features(trial)
        samplers = {
            "rf": _sample_rf,
            "extra_trees": _sample_extra_trees,
            "hgbr": _sample_hgbr,
            "xgb": _sample_xgb,
            "lgbm": _sample_lgbm,
            "mlp": _sample_mlp,
        }
        config["model_params"] = samplers[family](trial)

    return config


def sample_config_phase1(
    trial: optuna.Trial, families: list[str] | None = None
) -> dict:
    """Phase 1: Sample family + HPs only; fix features to all available.
    
    This allows the TPE surrogate to converge on good HP ranges without
    confusion from varying feature sets.
    """
    family_list = families if families is not None else ALL_FAMILIES
    family = trial.suggest_categorical("family", family_list)
    config: dict = {"family": family}

    if family in SEQUENTIAL_FAMILIES:
        # Use all available sequential features (no sampling)
        config["seq_features"] = SEQ_FEATURE_POOL
        config["static_features"] = STATIC_FEATURE_POOL
        config["model_params"] = (
            _sample_cnn(trial) if family == "cnn" else _sample_gru(trial)
        )
    else:
        # Use all tabular features (no sampling)
        config["features"] = ALL_FEATURES
        samplers = {
            "rf": _sample_rf,
            "extra_trees": _sample_extra_trees,
            "hgbr": _sample_hgbr,
            "xgb": _sample_xgb,
            "lgbm": _sample_lgbm,
            "mlp": _sample_mlp,
        }
        config["model_params"] = samplers[family](trial)

    return config


def sample_config_phase2(
    trial: optuna.Trial,
    fixed_family: str,
    fixed_params: dict,
) -> dict:
    """Phase 2: Fix family + HPs from Phase 1; sample feature subsets only.
    
    This enables clean feature ablation: each trial is a direct A/B test
    of feature inclusion/exclusion.
    """
    config: dict = {"family": fixed_family}

    if fixed_family in SEQUENTIAL_FAMILIES:
        # Sample feature subsets for sequential models
        seq_feats, static_feats = _sample_seq_features(trial)
        config["seq_features"] = seq_feats
        config["static_features"] = static_feats
    else:
        # Sample feature subsets for tabular models
        config["features"] = _sample_tabular_features(trial)

    config["model_params"] = fixed_params

    return config


# ---------------------------------------------------------------------------
# Warm-start seeds (enqueued before the first random trial)
# Best known configs from learnings.md, translated to Optuna param dicts.
# ---------------------------------------------------------------------------

_optional_features = [f for f in ALL_FEATURES if f not in REQUIRED_FEATURES]

WARM_START_CONFIGS: list[dict] = [
    # Best known CNN — apr14/exp22, RMSE 0.006126
    {
        "family": "cnn",
        **{f"seq_feat_{f}": True for f in SEQ_FEATURE_POOL},
        **{f"static_feat_{f}": True for f in STATIC_FEATURE_POOL},
        "cnn_seq_len": 5,
        "cnn_n_layers": 3,
        "cnn_channels": 128,
        "cnn_kernel_size": 3,
        "cnn_dropout": 0.1,
        "cnn_lr": 3e-3,
        "cnn_batch_size": 2048,
        "cnn_weight_decay": 1e-4,
        "cnn_grad_clip": 1.0,
    },
    # Best known RF — apr13b/exp20, RMSE 0.006400
    {
        "family": "rf",
        **{f"feat_{f}": True for f in _optional_features},
        "rf_n_estimators": 2000,
        "rf_max_depth": None,
        "rf_min_samples_split": 10,
        "rf_max_features": 0.7,
        "rf_max_samples": 0.5,
    },
]
