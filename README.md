# Ant Robot Simulation and Control with MuJoCo and Brax

This repo demonstrates the use of MuJoCo and Brax for simulating and controlling an Ant robot. It includes creating a basic simulation with random controls, replicating multiple Ant robots in a single environment, training a PPO control policy using Brax, and applying the trained policy in a MuJoCo simulation.


## Task Overview
- **Task 1(Basic Simulation)**: Simulate a single Ant robot in MuJoCo with random control inputs using the `mujoco.viewer`.
- **Task 2(Environment Replication)**: Create a function to replicate multiple Ant robots in a single MuJoCo model and simulate them with random controls.
- **Task 3(PPO Training)**: Train a control policy for the Ant using Brax's PPO implementation.
- **Task 4(Policy Visualization)**: Load the trained PPO policy and test it in a MuJoCo simulation.

## Prerequisites 
- Virtual environment
```bash
python3 -m venv mjenv
source mjenv/bin/activate

```

After you enter virtual env, install these libraries
- Jax (follow this [reference](https://docs.jax.dev/en/latest/quickstart.html) or if you have cuda 12 then just do the following)
```bash
pip install -U "jax[cuda12]"
```

- Other dependencies
```bash
pip install mujoco brax numpy ipython matplotlib
```
Now just clone the repo and cd under it

```bash
git clone https://github.com/ChinmayMundane/Ant_Simulation_Brax.git
cd Ant_Simulation_Brax
```

## File Structure
```
Ant_Simulation_Brax/
├── ant_control.py                     # task1
│  
├── multiple_ant_control.py            # task2         
│      
├── ppo.py                            # task3
│      
├── visualize.py                     # task4
│   
├── ant.xml
├── humanoid.xml
├── half_cheetah.xml
│   
└── Readme.md
```

## Usage

## Task 1: Single Ant Simulation

Run a simulation of a single Ant robot with random controls:
```bash
python3 ant_control.py
```
- Loads ant.xml and applies random control inputs for 10 seconds.
- Displays the simulation using mujoco.viewer.

[1.webm](https://github.com/user-attachments/assets/d71dcc1a-b974-4e28-aa7a-0dbe4d0d38ae)

## Task 2: Multiple Ant Environments


Replicate and simulate multiple Ant robots:

```bash
python3 multiple_ant_control.py

```
- Uses ant.xml by default (update to half_cheetah.xml or humanoid.xml if needed).
- Replicates 6 models in a 2x3 grid with a separation of 3 units.
- Applies random controls to all robots.
- Note: You can only simulate models which doesn't have global co-ordinates like ant, humanoid and half_cheetah.

</br > 

### Ant

[2.1.webm](https://github.com/user-attachments/assets/ddb43ca4-4623-4bcc-9fe1-016e73228c88)

</br > 


### Half_Cheetah
[2.2.webm](https://github.com/user-attachments/assets/33648050-bde9-4030-ab63-7d1a8f31326e)

</br > 


#### Humanoid
[2.3.webm](https://github.com/user-attachments/assets/e2f68fb1-b7d4-4801-a986-308c45f23ad1)

</br > 

## Task 3: PPO Training


Train a PPO policy for the Ant:
```bash
python3 ppo.py

```
- Trains the policy for 75M timesteps using Brax's Ant environment.
- Saves a plot of the training progress as training_progress.png to show the rewards.
</br > 

###  2.5×10^7 steps
![training_progress](https://github.com/user-attachments/assets/3cc53acc-5e6f-4bf0-a637-df843ae362fd)

- trained on RTX 3050 GPU and took 4mins 18 sec to complete.

###  5×10^7 steps (given by brax implementation)
![training_progress1](https://github.com/user-attachments/assets/ecf32a71-9789-4187-9a16-24d061d3ccc7)

- trained on RTX 3050 GPU and took 7mins to complete.


###  7.5×10^7 steps
![training_progress2](https://github.com/user-attachments/assets/f3a5f270-bc57-4f53-a484-d5bda4dea243)

- trained on RTX 3050 GPU and took 10mins 30 sec to complete.


Based on these, 75M timesteps appears to be the best for the task of training a control policy using Brax' PPO implementation because:

- It achieves the highest maximum reward (approximately 5900)

- & It shows faster initial learning (steeper slope in the early phase)

Even after some performance decline, it maintains a higher final reward than either of the other runs.

## Task 4: Visualize Trained Policy


Test the trained PPO policy in MuJoCo:
```bash
python3 visualize.py
```


- Saves and loads the trained policy from ant_ppo_policy.
- Applies it to the Ant in ant.xml and visualizes the rollout.


[4.webm](https://github.com/user-attachments/assets/8d062ba3-bbab-430f-b6e0-2064ec864f8f)


(Note that third script only saves the reward of the training while fourth script trains the model, saves and load it and then use that for evaluation.)


Youtube demonstration is available here.
- Task1 : https://www.youtube.com/watch?v=Y36Vi_03POo&authuser=0
- Task2 : https://www.youtube.com/watch?v=XTZE3wCcno8&authuser=0
- Task3 : https://www.youtube.com/watch?v=XNvRIW9-fDo&authuser=0
