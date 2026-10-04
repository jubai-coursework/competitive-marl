import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3
from algo import PPOAgent, compute_gae

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--max_cycles", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from previously saved models if they exist.")
    return parser.parse_args()

def check_capture(env):
    world = env.unwrapped.world
    good_agents = [agent for agent in world.agents if not agent.adversary]
    adversaries = [agent for agent in world.agents if agent.adversary]

    for good_agent in good_agents:
        for adv in adversaries:
            delta_pos = good_agent.state.p_pos - adv.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = good_agent.size + adv.size
            if dist < dist_min:
                return True
    return False

def min_distance_to_prey(env):
    """返回所有adversary与good agent之间的最小距离"""
    world = env.unwrapped.world
    good_agents = [agent for agent in world.agents if not agent.adversary]
    adversaries = [agent for agent in world.agents if agent.adversary]
    if not good_agents or not adversaries:
        return 0.0
    min_dist = float('inf')
    for good_agent in good_agents:
        for adv in adversaries:
            delta_pos = good_agent.state.p_pos - adv.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            min_dist = min(min_dist, dist)
    return min_dist

def make_env(max_cycles):
    return simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=0,
        max_cycles=max_cycles, continuous_actions=True
    )

def train():
    args = parse_args()
    print(f"Starting {args.algo} Training...")

    env = make_env(args.max_cycles)
    env.reset()

    adversaries = [agent for agent in env.agents if 'adversary' in agent]
    good_agents = [agent for agent in env.agents if agent.startswith('agent')]

    def get_obs_dim(agent_id):
        return env.observation_space(agent_id).shape[0]

    def get_act_dim(agent_id):
        return env.action_space(agent_id).shape[0]

    adv_obs_dim = get_obs_dim(adversaries[0])
    adv_act_dim = get_act_dim(adversaries[0])
    good_obs_dim = get_obs_dim(good_agents[0])
    good_act_dim = get_act_dim(good_agents[0])

    global_state_dim = sum([get_obs_dim(a) for a in env.agents])

    adv_state_dim = global_state_dim if args.algo == "MAPPO" else adv_obs_dim
    good_state_dim = global_state_dim if args.algo == "MAPPO" else good_obs_dim

    policy_adv = PPOAgent(adv_obs_dim, adv_state_dim, adv_act_dim, entropy_coef=0.01)
    policy_good = PPOAgent(good_obs_dim, good_state_dim, good_act_dim, entropy_coef=0.01)

    if args.resume:
        try:
            policy_adv.actor.load_state_dict(torch.load(f"results/{args.algo}_adv_actor.pth"))
            policy_good.actor.load_state_dict(torch.load(f"results/{args.algo}_good_actor.pth"))
            print(f"Resumed {args.algo} from the existing weights successfully.")
        except Exception as e:
            print(f"Failed to resume {args.algo} from weights: {e}. Starting from scratch.")

    records = {
        "adv_rewards": [],
        "good_rewards": [],
        "capture_rates": [],
        "survive_steps": []
    }

    if args.resume and os.path.exists(f"results/{args.algo}_records.npy"):
        try:
            old_records = np.load(f"results/{args.algo}_records.npy", allow_pickle=True).item()
            for k in records.keys():
                records[k] = old_records[k]
            print(f"Resumed {args.algo} historical records successfully.")
        except Exception as e:
            print(f"Failed to load records: {e}")

    def get_global_state(obs_dict):
        state = []
        for a in env.agents:
            state.append(obs_dict[a])
        return np.concatenate(state)

    best_cap_rate = 0.0

    for episode in range(1, args.episodes + 1):
        obs_dict, _ = env.reset()

        rollouts_adv = {'obs': [], 'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}
        rollouts_good = {'obs': [], 'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

        ep_adv_reward = 0
        ep_good_reward = 0
        steps = 0

        while env.agents:
            steps += 1
            actions_dict = {}
            values_dict = {}
            log_probs_dict = {}

            global_state = get_global_state(obs_dict) if args.algo == "MAPPO" else None

            for agent in env.agents:
                obs = obs_dict[agent]
                if 'adversary' in agent:
                    state = global_state if args.algo == "MAPPO" else obs
                    act, log_prob = policy_adv.get_action(obs)
                    val = policy_adv.get_value(state)

                    rollouts_adv['obs'].append(obs)
                    rollouts_adv['states'].append(state)
                    rollouts_adv['actions'].append(act)
                    rollouts_adv['log_probs'].append(log_prob)
                    rollouts_adv['values'].append(val)
                else:
                    state = global_state if args.algo == "MAPPO" else obs
                    act, log_prob = policy_good.get_action(obs)
                    val = policy_good.get_value(state)

                    rollouts_good['obs'].append(obs)
                    rollouts_good['states'].append(state)
                    rollouts_good['actions'].append(act)
                    rollouts_good['log_probs'].append(log_prob)
                    rollouts_good['values'].append(val)

                actions_dict[agent] = act

            next_obs_dict, rewards_dict, terminations, truncations, _ = env.step(actions_dict)

            # Reward shaping: 距离惩罚，鼓励adversary靠近good agent
            min_dist = min_distance_to_prey(env)
            distance_penalty = -0.1 * min_dist  # 距离越远惩罚越大

            adv_r = sum([rewards_dict.get(a, 0) for a in adversaries]) + distance_penalty * len(adversaries)
            good_r = sum([rewards_dict.get(a, 0) for a in good_agents]) - distance_penalty

            ep_adv_reward += adv_r
            ep_good_reward += good_r

            done = any(terminations.values()) or any(truncations.values())

            for _ in range(len(adversaries)):
                if len(rollouts_adv['rewards']) < len(rollouts_adv['obs']):
                    rollouts_adv['rewards'].append(adv_r / len(adversaries))
                    rollouts_adv['dones'].append(1 if done else 0)

            for _ in range(len(good_agents)):
                if len(rollouts_good['rewards']) < len(rollouts_good['obs']):
                    rollouts_good['rewards'].append(good_r / len(good_agents))
                    rollouts_good['dones'].append(1 if done else 0)

            obs_dict = next_obs_dict
            if not env.agents:
                break

        # Final value
        global_state = get_global_state(obs_dict) if args.algo == "MAPPO" and env.agents else None

        next_v_adv = 0
        if len(env.agents) > 0 and adversaries[0] in obs_dict:
            state = global_state if args.algo == "MAPPO" else obs_dict[adversaries[0]]
            next_v_adv = policy_adv.get_value(state)

        next_v_good = 0
        if len(env.agents) > 0 and good_agents[0] in obs_dict:
            state = global_state if args.algo == "MAPPO" else obs_dict[good_agents[0]]
            next_v_good = policy_good.get_value(state)

        # Pass dones to compute_gae
        returns_adv, advs_adv = compute_gae(
            rollouts_adv['rewards'], rollouts_adv['values'], next_v_adv, rollouts_adv['dones']
        )
        rollouts_adv['returns'] = returns_adv
        rollouts_adv['advantages'] = advs_adv

        returns_good, advs_good = compute_gae(
            rollouts_good['rewards'], rollouts_good['values'], next_v_good, rollouts_good['dones']
        )
        rollouts_good['returns'] = returns_good
        rollouts_good['advantages'] = advs_good

        policy_adv.update(rollouts_adv)
        policy_good.update(rollouts_good)

        records["adv_rewards"].append(ep_adv_reward)
        records["good_rewards"].append(ep_good_reward)

        if episode % args.eval_interval == 0:
            eval_adv_r, eval_good_r, cap_rate, surv_steps = evaluate(args, policy_adv, policy_good, adversaries, good_agents)
            records["capture_rates"].append(cap_rate)
            records["survive_steps"].append(surv_steps)
            print(f"Ep: {episode}, Adv R: {eval_adv_r:.2f}, Good R: {eval_good_r:.2f}, Cap Rate: {cap_rate:.2f}, Surv Steps: {surv_steps:.2f}")

            if cap_rate > best_cap_rate:
                best_cap_rate = cap_rate
                os.makedirs("results", exist_ok=True)
                torch.save(policy_adv.actor.state_dict(), f"results/{args.algo}_adv_actor_best.pth")
                torch.save(policy_good.actor.state_dict(), f"results/{args.algo}_good_actor_best.pth")

    os.makedirs("results", exist_ok=True)
    torch.save(policy_adv.actor.state_dict(), f"results/{args.algo}_adv_actor.pth")
    torch.save(policy_good.actor.state_dict(), f"results/{args.algo}_good_actor.pth")

    np.save(f"results/{args.algo}_records.npy", records)
    plot_results(records, args.algo)
    print(f"Training complete. Best capture rate: {best_cap_rate:.3f}")

def evaluate(args, policy_adv, policy_good, adversaries, good_agents):
    """使用独立的env进行评估"""
    eval_env = make_env(args.max_cycles)
    adv_rewards = []
    good_rewards = []
    capture_counts = 0
    first_capture_steps = []

    def get_global_state(obs_dict):
        state = []
        for a in eval_env.agents:
            state.append(obs_dict[a])
        return np.concatenate(state)

    for _ in range(args.eval_episodes):
        obs_dict, _ = eval_env.reset()
        ep_adv_r = 0
        ep_good_r = 0
        captured = False
        steps = 0

        while eval_env.agents:
            steps += 1
            actions_dict = {}
            for agent in eval_env.agents:
                obs = obs_dict[agent]
                if 'adversary' in agent:
                    act, _ = policy_adv.get_action(obs, deterministic=True)
                else:
                    act, _ = policy_good.get_action(obs, deterministic=True)
                actions_dict[agent] = act

            obs_dict, rewards_dict, term, trunc, _ = eval_env.step(actions_dict)

            ep_adv_r += sum([rewards_dict.get(a, 0) for a in adversaries])
            ep_good_r += sum([rewards_dict.get(a, 0) for a in good_agents])

            if not captured and check_capture(eval_env):
                captured = True
                first_capture_steps.append(steps)

            if not eval_env.agents:
                break

        adv_rewards.append(ep_adv_r)
        good_rewards.append(ep_good_r)
        if captured:
            capture_counts += 1
        if not captured:
            first_capture_steps.append(args.max_cycles)

    mean_survive = np.mean(first_capture_steps) if first_capture_steps else args.max_cycles
    eval_env.close()
    return np.mean(adv_rewards), np.mean(good_rewards), capture_counts / args.eval_episodes, mean_survive

def plot_results(records, algo):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(records["adv_rewards"], alpha=0.3, color='red', label='raw')
    window = max(1, len(records["adv_rewards"]) // 100)
    smoothed = np.convolve(records["adv_rewards"], np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(records["adv_rewards"])), smoothed, color='red', label=f'smoothed ({window})')
    plt.title(f"{algo} Adversary Rewards")
    plt.xlabel("Episode")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(records["good_rewards"], alpha=0.3, color='green', label='raw')
    smoothed = np.convolve(records["good_rewards"], np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(records["good_rewards"])), smoothed, color='green', label=f'smoothed ({window})')
    plt.title(f"{algo} Good Rewards")
    plt.xlabel("Episode")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(records["capture_rates"])
    plt.title(f"{algo} Capture Rate")
    plt.xlabel("Evaluation (x Interval)")
    plt.ylim(0, 1)

    plt.subplot(2, 2, 4)
    plt.plot(records["survive_steps"])
    plt.title(f"{algo} Steps to First Capture")
    plt.xlabel("Evaluation (x Interval)")

    plt.tight_layout()
    plt.savefig(f"results/{algo}_curves.png")
    plt.close()

if __name__ == "__main__":
    train()
