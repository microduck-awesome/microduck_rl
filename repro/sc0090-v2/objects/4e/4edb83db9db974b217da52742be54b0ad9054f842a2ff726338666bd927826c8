"""Proven SC0090 recovery curriculum port, adapted to measured V2 poses.

Source: backup/local-work-20261015, mdp.py. Deterministic evaluation, not
stochastic training reward, promotes roll frontiers. Fresh V2 policies retain
gradual head/CoM curricula instead of inheriting a mature old-model actor.
"""
import math
import torch
from mjlab.managers import SceneEntityCfg
from mjlab_microduck.tasks.mdp import set_random_ground_state
from .training import STAND_HEIGHT, SIT_HEIGHT
_DEFAULT_ASSET_CFG = SceneEntityCfg('robot')
SC0090_RECOVERY_BUCKETS = ('standing','sitting','prone','frontier','left_side','right_side','supine')
SC0090_ROLL_FRONTIERS = (150., 120., 90., 60., 30., 0.)

class SC0090RecoveryState:
    """Episode buffers plus checkpointable success-driven curriculum statistics."""

    def __init__(self, env):
        n, device = env.num_envs, env.device
        self.bucket = torch.full((n,), -1, dtype=torch.long, device=device)
        self.spawn_stage = torch.zeros(n, dtype=torch.long, device=device)
        self.hold = torch.zeros(n, dtype=torch.long, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.stage = 0
        self.last_assessment_step = 0
        self.polish = False
        self.trials = [0] * 7
        self.wins = [0] * 7
        self.rates = [0.0] * 7
        self.total_trials = [0] * 7

    def state_dict(self):
        return {k: getattr(self, k) for k in (
            "stage", "last_assessment_step", "polish", "trials", "wins",
            "rates", "total_trials")}

    def load_state_dict(self, state):
        for key in self.state_dict():
            setattr(self, key, state[key])
        # Physics/episodes are not serialized by mjlab. Never count an episode
        # from before restart toward the restored frontier's success rate.
        self.valid.zero_()
        self.hold.zero_()
        self.success.zero_()


def _sc0090_recovery_state(env):
    if not hasattr(env, "_sc0090_recovery"):
        env._sc0090_recovery = SC0090RecoveryState(env)
    return env._sc0090_recovery


def sc0090_reset_recovery(env, env_ids, asset_cfg=_DEFAULT_ASSET_CFG,
                          prone_z_min=.05, prone_z_max=.09,
                          sitting_z_min=.05, sitting_z_max=.09,
                          standing_z_min=.11, standing_z_max=.12,
                          sitting_joint_overrides=None, sitting_joint_noise_std=.12,
                          sitting_tilt_max=math.radians(10), evaluation_buckets=False):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = _sc0090_recovery_state(env)
    # Rehearse known skills (25%); explore a reachable roll frontier (40%);
    # always train actual left/right/full-back failures (35%).
    if evaluation_buckets:
        if env.num_envs % len(SC0090_RECOVERY_BUCKETS):
            raise ValueError("Recovery evaluation needs equal-sized pose groups")
        bucket = env_ids // (env.num_envs // len(SC0090_RECOVERY_BUCKETS))
    else:
        bucket = torch.multinomial(torch.tensor((.05, .10, .10, .40, .10, .10, .15),
                                                device=env.device), len(env_ids), True)
    state.bucket[env_ids] = bucket
    state.spawn_stage[env_ids] = state.stage
    state.hold[env_ids] = 0
    state.success[env_ids] = False
    state.valid[env_ids] = True
    for group in range(7):
        ids = env_ids[bucket == group]
        if not len(ids):
            continue
        set_random_ground_state(
            env, ids, asset_cfg=asset_cfg, standing_prob=float(group == 0),
            sitting_prob=float(group == 1), face_down_prob=float(group == 2),
            face_up_prob=float(group >= 3), face_up_roll_max=0.0,
            prone_z_min=prone_z_min, prone_z_max=prone_z_max,
            sitting_z_min=sitting_z_min, sitting_z_max=sitting_z_max,
            standing_z_min=standing_z_min, standing_z_max=standing_z_max,
            sitting_joint_overrides=sitting_joint_overrides,
            sitting_joint_noise_std=sitting_joint_noise_std,
            sitting_tilt_max=sitting_tilt_max,
        )
        if group in (3, 4, 5):
            if group == 3:
                low = SC0090_ROLL_FRONTIERS[state.stage]
                angle = torch.deg2rad(low + 30 * torch.rand(len(ids), device=env.device))
                angle *= torch.where(torch.rand_like(angle) < .5, -1.0, 1.0)
            else:
                # +90 about body z from supine puts body +y (left) down.
                angle = torch.full((len(ids),), math.pi / 2 * (1 if group == 4 else -1),
                                   device=env.device)
            ct, st = torch.cos(angle / 2), torch.sin(angle / 2)
            w, x, y, z = env.sim.data.qpos[ids, 3:7].unbind(dim=1)
            env.sim.data.qpos[ids, 3:7] = torch.stack((w*ct-z*st, x*ct+y*st,
                                                     y*ct-x*st, w*st+z*ct), dim=1)


def sc0090_recovery_potential(quat, height):
    # gravity in body x: -1 supine, 0 side, +1 prone. Upright remains the
    # best potential, so completing prone->stand does not give progress back.
    gravity_x = 2 * (quat[:, 0] * quat[:, 2] - quat[:, 1] * quat[:, 3])
    belly = .5 * (1 + gravity_x.clamp(-1, 1))
    cos_tilt = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    standing = 2 * ((height - SIT_HEIGHT) / (STAND_HEIGHT - SIT_HEIGHT)).clamp(0, 1) * cos_tilt.clamp(0, 1)
    return torch.maximum(belly, standing)


def sc0090_recovery_progress(env):
    """Signed potential change per second: stationary/closed-loop poses cannot farm it."""
    robot = env.scene["robot"].data
    height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    potential = torch.nan_to_num(sc0090_recovery_potential(robot.root_link_quat_w, height))
    previous = getattr(env, "_sc0090_potential_prev", potential)
    delta = potential - previous
    delta = torch.where(env.episode_length_buf <= 1, 0.0, delta)
    env._sc0090_potential_prev = potential.clone()
    return delta / env.step_dt


def sc0090_stable_standing(env):
    """Continuous 0.5 s stability, not a one-frame height crossing or a jackpot."""
    state = _sc0090_recovery_state(env)
    robot = env.scene["robot"].data
    q = robot.root_link_quat_w
    cosine = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    feet = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs, 2, -1)
    stable = ((height >= STAND_HEIGHT - .010) & (cosine >= math.cos(math.radians(20)))
              & (robot.root_link_lin_vel_b.norm(dim=1) < .15)
              & (robot.root_link_ang_vel_b.norm(dim=1) < 1.5)
              & feet.bool().any(dim=2).all(dim=1))
    state.hold = torch.where(stable, state.hold + 1, 0)
    completed = state.hold >= math.ceil(.5 / env.step_dt)
    state.success |= completed
    return completed.float()


def sc0090_apply_recovery_evaluation(env, report, request, min_stage_steps=2400,
                                    success_threshold=.7):
    """Validate an isolated assessment before changing any curriculum state.

    Each seed must retain the three easy skills AND pass the current frontier.
    Late/duplicate results, mismatched checkpoints and incomplete episodes
    cannot promote. Training's stochastic success remains a separate metric.
    """
    state = _sc0090_recovery_state(env)
    history = getattr(env, "_sc0090_eval_history", {})
    for key in ("schema", "checkpoint_sha256", "config_sha256", "iteration", "stage",
                "seeds", "samples_per_group", "duration_s", "deterministic"):
        if report.get(key) != request[key]:
            raise ValueError(f"Recovery assessment mismatch: {key}")
    if (report["schema"] != 1 or report["deterministic"] is not True
            or report["stage"] != state.stage or report["duration_s"] != 8.0
            or len(set(report["seeds"])) != len(report["seeds"])
            or len(report["seeds"]) < 2):
        raise ValueError("Invalid recovery assessment protocol/stage")
    if report["iteration"] <= history.get("last_iteration", -1):
        raise ValueError("Recovery assessment was already consumed")
    cases = report.get("cases", [])
    if len(cases) != len(report["seeds"]):
        raise ValueError("Missing recovery evaluation seeds")
    totals = {name: [0, 0] for name in SC0090_RECOVERY_BUCKETS}
    per_seed = []
    for seed, case in zip(report["seeds"], cases):
        if case.get("seed") != seed or case.get("finite") is not True:
            raise ValueError("Invalid/nonfinite recovery evaluation")
        groups = case.get("groups", {})
        if set(groups) != set(totals):
            raise ValueError("Missing recovery evaluation poses")
        rates = {}
        for name, counts in groups.items():
            trials, wins = counts.get("trials"), counts.get("wins")
            if (type(trials) is not int or type(wins) is not int
                    or trials != report["samples_per_group"] or trials < 1
                    or not 0 <= wins <= trials):
                raise ValueError("Invalid recovery trial/win counts")
            totals[name][0] += trials
            totals[name][1] += wins
            rates[name] = wins / trials
        per_seed.append(rates)
    eligible = (totals["frontier"][0] >= 256
                and env.common_step_counter - history.get("last_promotion_step", 0) >= min_stage_steps)
    easy_and_frontier = ("standing", "sitting", "prone", "frontier")
    mastered = eligible and all(r[name] >= success_threshold
                               for r in per_seed for name in easy_and_frontier)
    promoted = mastered and state.stage < len(SC0090_ROLL_FRONTIERS) - 1
    polished = (mastered and not promoted
                and all(r[name] >= success_threshold for r in per_seed
                        for name in ("left_side", "right_side", "supine")))
    if promoted:
        state.stage += 1
        state.trials = [0] * len(SC0090_RECOVERY_BUCKETS)
        state.wins = [0] * len(SC0090_RECOVERY_BUCKETS)
        state.last_assessment_step = env.common_step_counter
    if polished:
        state.polish = True
    env._sc0090_eval_history = {
        **history, "last_iteration": report["iteration"],
        "last_promotion_step": (env.common_step_counter if promoted
                                else history.get("last_promotion_step", 0)),
        "evaluated_stage": report["stage"], "stage_after": state.stage,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "counts": totals, "rates": {k: w / n for k, (n, w) in totals.items()},
        "promoted": bool(promoted), "polish": bool(state.polish),
    }
    return env._sc0090_eval_history
