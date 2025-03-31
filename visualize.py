import functools
import jax
import mujoco
import mujoco.viewer
import numpy as np
import time
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from IPython.display import HTML
from brax.io import html

# Initialize Brax environment
env = envs.create(env_name="ant", backend="positional")

# Define PPO training function
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
    seed=1
)

make_inference_fn, params, _ = train_fn(environment=env)
model.save_params('ant_ppo_policy', params)
params = model.load_params("ant_ppo_policy")

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params))

# Load MuJoCo model
xml_path = "ant.xml"
with open(xml_path, 'r') as f:
    xml_string = f.read()
model_mj = mujoco.MjModel.from_xml_string(xml_string)
data_mj = mujoco.MjData(model_mj)
viewer = mujoco.viewer.launch_passive(model_mj, data_mj)

# Configure Camera
torso_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, 'torso')
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
viewer.cam.trackbodyid = torso_id
viewer.cam.distance = 7.0
viewer.cam.azimuth = 0.0
viewer.cam.elevation = -20.0

# Run Policy in MuJoCo and Visualize Rollout
rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
    data_mj.ctrl[:] = np.array(act)
    mujoco.mj_step(model_mj, data_mj)
    viewer.sync()
    time.sleep(0.01)

HTML(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
print("Simulation ended.")
