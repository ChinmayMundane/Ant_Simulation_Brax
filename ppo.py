import functools
import matplotlib.pyplot as plt
from brax import envs
from brax.training.agents.ppo import train as ppo

# Load Environment
env = envs.get_environment(env_name="ant", backend="positional")  # can also be "generalized" or "spring"

# Training Hyperparameters
train_fn = functools.partial(
    ppo.train,
    num_timesteps=75_000_000,
    num_evals=10,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=2048,
    seed=1,
)

# Track Training Progress
xdata, ydata = [], []
def progress(num_steps, metrics):
    xdata.append(num_steps)
    ydata.append(metrics["eval/episode_reward"])

# Train the PPO Agent
inference_fn, params, state = train_fn(environment=env, progress_fn=progress)

# Save Training Plot
plt.plot(xdata, ydata)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.grid()
plt.savefig("training_progress.png")
print("Saved plot: training_progress.png")
