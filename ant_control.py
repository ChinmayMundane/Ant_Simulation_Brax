import mujoco
import mujoco.viewer
import numpy as np
import time

def load_model(xml_path: str): # load mujoco model from xml file
    with open(xml_path, 'r') as f:
        xml_string = f.read()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data

def configure_camera(viewer, model): # configure camera to track robot
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = torso_id
    viewer.cam.distance = 5.0
    viewer.cam.azimuth = 0.0
    viewer.cam.elevation = -10.0

def run_simulation(xml_path: str, duration: float = 10.0, dt: float = 0.01): # run simulation with random control
    model, data = load_model(xml_path)
    viewer = mujoco.viewer.launch_passive(model, data)
    configure_camera(viewer, model)
    
    time.sleep(1)
    start_time = time.time()

    while viewer.is_running() and (time.time() - start_time) < duration:
        data.ctrl[:] = np.random.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

    print("Simulation ended")

if __name__ == "__main__":
    run_simulation("ant.xml")
