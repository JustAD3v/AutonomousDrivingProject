"""Which settings each algorithm has, their defaults, and how to save them.

No Qt code here, so it can be tested on its own. The defaults mirror
src/Trainers/TrainConfig.py plus the overrides in src/main.py, so update them
here when those files change.
"""
import copy


FIELDS = {
    "env_name": ("env", "highway-fast-v0"),
    "seed": ("seed", None),
    "iterations": ("int", 40),
    "total_timesteps": ("int", 200000),
    "learning_rate": ("float", 5e-4),
    "gamma": ("float", 0.92),
    "batch_size": ("int", 64),
    "buffer_size": ("int", 2048),
    "max_grad_norm": ("float", 0.5),
    "policy_kwargs.net_arch": ("int_list", [256, 256]),
    "log_freq": ("int", 20),

    "env_config.lanes_count": ("int", 4),
    "env_config.vehicles_count": ("int", 40),
    "env_config.duration": ("int", 30),
    "env_config.ego_spacing": ("float", 2),
    "env_config.vehicles_density": ("float", 1.1),
    "env_config.collision_reward": ("float", -5),
    "env_config.right_lane_reward": ("float", 0.0),
    "env_config.high_speed_reward": ("float", 1.0),
    "env_config.lane_change_reward": ("float", 0.02),
    "env_config.reward_speed_range": ("float_list", [25, 40]),
    "env_config.normalize_reward": ("bool", True),
    "env_config.offroad_terminal": ("bool", False),
    "env_config.simulation_frequency": ("int", 5),
    "env_config.policy_frequency": ("int", 1),

    "epsilon_start": ("float", 1.0),
    "epsilon": ("float", 1.0),
    "epsilon_min": ("float", 0.01),
    "epsilon_decay": ("float", 0.994),
    "exploration_frac": ("float", 0.1),
    "warmup_size": ("int", 800),
    "copy_freq": ("int", 800),
    "train_freq": ("int", 1),

    "ppo_lm": ("float", 0.8),
    "ppo_clip_epsilon": ("float", 0.2),
    "ent_coef": ("float", 0.02),
    "ppo_update_epochs": ("int", 10),
    "ppo_minibatch_size": ("opt_int", 64),
    "ppo_actor_update_steps": ("int", 10),
    "ppo_critic_update_steps": ("int", 10),
    "parallel": ("bool", False),
    "ppo_nworkers": ("int", 8),
    "ppo_continuous": ("bool", False),
    "log_std_min": ("float", -5.0),

    "n_envs": ("int", 8),
    "n_steps": ("int", 2048),
    "ppo_batch_size": ("int", 64),
    "n_epochs": ("int", 10),
    "ppo_gamma": ("float", 0.9),
    "gae_lambda": ("float", 0.8),
    "clip_range": ("float", 0.2),
    "vf_coef": ("float", 0.5),
    "ppo_net_arch": ("int_list", [256, 256]),
    "tensorboard_log": ("path", None),

    "eval_freq": ("int", 5000),
    "best_model_save_path": ("str", "./logs/"),
    "log_path": ("str", "./logs/"),

    "pretrained_net": ("path", None),
    "pretrained_critic_net": ("path", None),
    "k_wst": ("float", 0.5),
    "k_wst_start": ("float", 0.0),
    "k_wst_frac": ("float", 0.8),
    "k_reg": ("float", 0.2),
    "attack_budget": ("float", 0.05),
    "attack_budget_start": ("float", 0.0),
    "attack_budget_frac": ("float", 0.8),
    "wocar_grad_step_size": ("float", 5e-4),
    "wocar_grad_steps": ("int", 100),
    "wocar_hinge_c": ("float", 1.0),
    "woca_update_epochs": ("int", 4),
    "woca_baseline_samples": ("int", 8),
}

ENV_SECTION = "Environment settings"
ENVIRONMENT = [key for key in FIELDS if key.startswith("env_config.")]

DQN = ["iterations", "learning_rate", "gamma", "batch_size", "buffer_size",
       "max_grad_norm", "policy_kwargs.net_arch", "seed", "log_freq"]
EXPLORATION = ["epsilon_start", "epsilon", "epsilon_min", "epsilon_decay",
               "exploration_frac", "warmup_size", "copy_freq", "train_freq"]
PPO_GENERAL = ["iterations", "learning_rate", "gamma", "buffer_size",
               "max_grad_norm", "policy_kwargs.net_arch", "seed", "log_freq"]
PPO = ["ppo_lm", "ppo_clip_epsilon", "ent_coef", "ppo_update_epochs",
       "ppo_minibatch_size", "ppo_actor_update_steps",
       "ppo_critic_update_steps", "parallel", "ppo_nworkers", "ppo_continuous",
       "log_std_min"]
WOCAR = ["k_wst", "k_wst_start", "k_wst_frac", "k_reg", "attack_budget",
         "attack_budget_start", "attack_budget_frac", "wocar_grad_step_size",
         "wocar_grad_steps"]
CALLBACK = ["eval_freq", "best_model_save_path", "log_path"]

ALGORITHMS = {
    "dqn": {"General": DQN, "Exploration": EXPLORATION},
    "sb3_dqn": {
        "General": ["iterations", "learning_rate", "gamma", "batch_size",
                    "buffer_size", "policy_kwargs.net_arch", "seed"],
        "Exploration": ["epsilon_min", "warmup_size", "copy_freq", "train_freq"],
        "Evaluation during training": CALLBACK,
    },
    "wocar_dqn": {
        "General": DQN, "Exploration": EXPLORATION,
        "WocaR": ["pretrained_net"] + WOCAR + ["wocar_hinge_c"],
    },
    "ppo": {"General": PPO_GENERAL, "PPO": PPO},
    "wocar_ppo": {
        "General": PPO_GENERAL, "PPO": PPO,
        "WocaR": ["pretrained_net", "pretrained_critic_net"] + WOCAR
                 + ["woca_update_epochs", "woca_baseline_samples"],
    },
    "sb3-ppo": {
        "General": ["total_timesteps", "learning_rate", "max_grad_norm", "seed"],
        "SB3 PPO": ["n_envs", "n_steps", "ppo_batch_size", "n_epochs",
                    "ppo_gamma", "gae_lambda", "clip_range", "ent_coef",
                    "vf_coef", "ppo_net_arch", "tensorboard_log"],
        "Evaluation during training": CALLBACK,
    },
}

ALGORITHM_DEFAULTS = {"wocar_dqn": {"k_wst_start": 1.0}}

def sections(algorithm):
    own = dict(ALGORITHMS[algorithm])
    general = ["env_name"] + own.pop("General")
    return {"General": general, ENV_SECTION: ENVIRONMENT, **own}


def get_value(config, key, algorithm):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return ALGORITHM_DEFAULTS.get(algorithm, {}).get(key, FIELDS[key][1])
        node = node[part]
    return node


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _to_int(text):
    try:
        return int(text)
    except ValueError:
        number = float(text)  # allows 5e3
        if not number.is_integer():
            raise ValueError("not a whole number")
        return int(number)


def parse(text, kind):
    text = text.strip()
    if not text:
        if kind in ("path", "opt_int", "seed"):
            return None
        raise ValueError("the field is empty")
    if kind in ("int", "opt_int", "seed"):
        return _to_int(text)
    if kind == "float":
        return float(text)
    if kind in ("int_list", "float_list"):
        convert = _to_int if kind == "int_list" else float
        return [convert(part.strip()) for part in text.split(",")]
    return text


def build_config(base, texts, use_env_config):
   
    config = copy.deepcopy(base)
    if not use_env_config:
        if config.get("env_config") is not None:
            del config["env_config"]
    elif config.get("env_config") is None:
        config["env_config"] = {}

    for key, raw in texts.items():
        if key.startswith("env_config.") and not use_env_config:
            continue
        kind = FIELDS[key][0]
        try:
            value = raw if kind == "bool" else parse(raw, kind)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{key}: invalid value '{raw}' ({error})")
        if kind == "seed" and value is None:
            config.pop(key, None)
            continue
        *parents, last = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return config
