import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_tag_v3
from algo import PPOAgent

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

def plot_trajectory(algo, max_cycles=100):
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=max_cycles, continuous_actions=True)
    env.reset()

    adversaries = [a for a in env.agents if 'adversary' in a]
    good_agents = [a for a in env.agents if a.startswith('agent')]

    def get_obs_dim(agent_id):
        return env.observation_space(agent_id).shape[0]
    def get_act_dim(agent_id):
        return env.action_space(agent_id).shape[0]

    adv_obs_dim = get_obs_dim(adversaries[0])
    adv_act_dim = get_act_dim(adversaries[0])
    good_obs_dim = get_obs_dim(good_agents[0])
    good_act_dim = get_act_dim(good_agents[0])
    global_state_dim = sum([get_obs_dim(a) for a in env.agents])

    adv_state_dim = global_state_dim if algo == "MAPPO" else adv_obs_dim
    good_state_dim = global_state_dim if algo == "MAPPO" else good_obs_dim

    policy_adv = PPOAgent(adv_obs_dim, adv_state_dim, adv_act_dim)
    policy_good = PPOAgent(good_obs_dim, good_state_dim, good_act_dim)

    # Try loading best model first, then regular
    for suffix in ["_best", ""]:
        try:
            policy_adv.actor.load_state_dict(torch.load(f"results/{algo}_adv_actor{suffix}.pth"))
            policy_good.actor.load_state_dict(torch.load(f"results/{algo}_good_actor{suffix}.pth"))
            print(f"Loaded models for {algo}{suffix}.")
            break
        except Exception:
            continue
    else:
        print(f"Model for {algo} not found, skipping plot.")
        env.close()
        return

    obs_dict, _ = env.reset()

    positions = {agent: [] for agent in env.agents}
    capture_points = []

    steps = 0
    was_captured = False  # prevent duplicate capture points for the same event
    while env.agents:
        world = env.unwrapped.world
        for agent in world.agents:
            positions[agent.name].append(agent.state.p_pos.copy())

        actions_dict = {}
        for agent in env.agents:
            obs = obs_dict[agent]
            if 'adversary' in agent:
                act, _ = policy_adv.get_action(obs, deterministic=True)
            else:
                act, _ = policy_good.get_action(obs, deterministic=True)
            actions_dict[agent] = act

        obs_dict, _, _, _, _ = env.step(actions_dict)

        if check_capture(env) and not was_captured:
            for agent in world.agents:
                if not agent.adversary:
                    capture_points.append(agent.state.p_pos.copy())
            was_captured = True

        steps += 1
        if steps >= max_cycles:
            break

    env.close()

    # Plotting
    plt.figure(figsize=(8, 8))

    colors = {'adversary': 'red', 'good': 'green'}

    for agent_name, pos_list in positions.items():
        pos_list = np.array(pos_list)
        if len(pos_list) == 0:
            continue
        c = colors['adversary'] if 'adversary' in agent_name else colors['good']
        plt.plot(pos_list[:, 0], pos_list[:, 1], color=c, alpha=0.6, label=f"{agent_name} Track")
        plt.scatter(pos_list[0, 0], pos_list[0, 1], marker='o', color=c, s=100, label=f"{agent_name} Start")
        plt.scatter(pos_list[-1, 0], pos_list[-1, 1], marker='x', color=c, s=100, label=f"{agent_name} End")

    for cp in capture_points:
        plt.scatter(cp[0], cp[1], marker='*', color='black', s=200, label='Capture Point')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.title(f"{algo} Trajectory")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.savefig(f"results/{algo}_trajectory.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
    parser.add_argument("--max_cycles", type=int, default=100)
    args = parser.parse_args()
    plot_trajectory(args.algo, max_cycles=args.max_cycles)
