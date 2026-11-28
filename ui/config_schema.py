"""Which config values the GUI can edit, and how to convert them.

No Qt code here, so it can be tested on its own. To support a new
hyperparameter, add one line to SECTIONS.
"""
import copy

# section title -> list of (key, type). A dot means a nested key.
SECTIONS = {
    "General": [
        ("env_name", "env"), ("seed", "int"), ("iterations", "int"),
        ("total_timesteps", "int"), ("learning_rate", "float"), ("gamma", "float"),
        ("batch_size", "int"), ("buffer_size", "int"), ("max_grad_norm", "float"),
        ("policy_kwargs.net_arch", "int_list"), ("parallel", "bool"),
        ("log_freq", "int"), ("pretrained_net", "path"),
        ("pretrained_critic_net", "path"),
    ],
    "Environment settings": [
        ("env_config.lanes_count", "int"), ("env_config.vehicles_count", "int"),
        ("env_config.duration", "int"), ("env_config.ego_spacing", "float"),
        ("env_config.vehicles_density", "float"),
        ("env_config.collision_reward", "float"),
        ("env_config.right_lane_reward", "float"),
        ("env_config.high_speed_reward", "float"),
        ("env_config.lane_change_reward", "float"),
        ("env_config.reward_speed_range", "float_list"),
        ("env_config.normalize_reward", "bool"),
        ("env_config.simulation_frequency", "int"),
        ("env_config.policy_frequency", "int"),
    ],
    "DQN / exploration": [
        ("epsilon_start", "float"), ("epsilon", "float"), ("epsilon_min", "float"),
        ("epsilon_decay", "float"), ("exploration_frac", "float"),
        ("warmup_size", "int"), ("copy_freq", "int"), ("train_freq", "int"),
    ],
    "PPO": [
        ("ent_coef", "float"), ("ppo_actor_update_steps", "int"),
        ("ppo_critic_update_steps", "int"), ("ppo_update_epochs", "int"),
        ("ppo_clip_epsilon", "float"), ("ppo_lm", "float"), ("ppo_nworkers", "int"),
        ("ppo_minibatch_size", "int"), ("ppo_continuous", "bool"),
        ("log_std_min", "float"),
    ],
    "SB3 PPO": [
        ("n_steps", "int"), ("ppo_batch_size", "int"), ("n_epochs", "int"),
        ("ppo_gamma", "float"), ("gae_lambda", "float"), ("clip_range", "float"),
        ("vf_coef", "float"),
    ],
    "WocaR": [
        ("k_wst", "float"), ("k_wst_start", "float"), ("k_wst_frac", "float"),
        ("k_reg", "float"), ("attack_budget", "float"),
        ("attack_budget_start", "float"), ("attack_budget_frac", "float"),
        ("wocar_grad_step_size", "float"), ("wocar_grad_steps", "int"),
        ("wocar_hinge_c", "float"), ("woca_update_epochs", "int"),
        ("woca_baseline_samples", "int"),
    ],
}


def has_key(config, key):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def get_value(config, key):
    for part in key.split("."):
        config = config[part]
    return config


def set_value(config, key, value):
    *parents, last = key.split(".")
    for part in parents:
        config = config.setdefault(part, {})
    config[last] = value


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
    """Turn the text of an input field into a JSON value. Raises ValueError."""
    text = text.strip()
    if kind == "path":
        return text or None
    if not text:
        raise ValueError("the field is empty")
    if kind == "int":
        return _to_int(text)
    if kind == "float":
        return float(text)
    if kind in ("int_list", "float_list"):
        convert = _to_int if kind == "int_list" else float
        return [convert(part.strip()) for part in text.split(",")]
    return text


def build_config(base, texts, use_env_config):
    """Copy base and overwrite the edited values. Unknown keys are kept.

    texts: {key: field text, or bool for checkboxes}
    """
    config = copy.deepcopy(base)
    if not use_env_config:
        if "env_config" in config:
            config["env_config"] = None
    elif config.get("env_config") is None:
        config["env_config"] = {}

    kinds = {key: kind for fields in SECTIONS.values() for key, kind in fields}
    for key, raw in texts.items():
        if key.startswith("env_config.") and not use_env_config:
            continue
        kind = kinds[key]
        try:
            value = raw if kind == "bool" else parse(raw, kind)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{key}: invalid value '{raw}' ({error})")
        set_value(config, key, value)
    return config
