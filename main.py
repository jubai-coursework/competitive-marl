import subprocess
import os

def main():
    print("--- 竞争型多智能体强化学习实验 ---")
    print("环境要求：Gymnasium, PettingZoo[mpe], PyTorch, Matplotlib")

    os.makedirs("results", exist_ok=True)

    # 1. IPPO
    print("\n>>> 开始训练 IPPO (50000 episodes) <<<")
    subprocess.run(["python", "train.py", "--algo", "IPPO", "--episodes", "50000", "--max_cycles", "100", "--resume"])

    # 2. MAPPO
    print("\n>>> 开始训练 MAPPO (50000 episodes) <<<")
    subprocess.run(["python", "train.py", "--algo", "MAPPO", "--episodes", "50000", "--max_cycles", "100", "--resume"])

    # 3. 绘制轨迹
    print("\n>>> 绘制 IPPO 轨迹图 <<<")
    subprocess.run(["python", "eval_plot.py", "--algo", "IPPO", "--max_cycles", "100"])

    print("\n>>> 绘制 MAPPO 轨迹图 <<<")
    subprocess.run(["python", "eval_plot.py", "--algo", "MAPPO", "--max_cycles", "100"])

    print("\n实验完成，所有结果保存在 results 目录下！")

if __name__ == "__main__":
    main()
