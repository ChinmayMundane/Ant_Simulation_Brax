import mujoco
import mujoco.viewer
import numpy as np
import time

def replicate_env(num_envs, env_separation, envs_per_row, xml_string): # replicate multiple env
    spec = mujoco.MjSpec.from_string(xml_string)
    spec.copy_during_attach = True 

    new_spec = mujoco.MjSpec()
    new_spec.copy_during_attach = True
    chequered = new_spec.add_texture(
        name="chequered", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5]
    )
    grid = new_spec.add_material(
        name='grid', texrepeat=[5, 5], reflectance=.2
    ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'chequered'
    
    new_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .1], material='grid'
    )

    # Add ambient lighting
    for x in [-20, 20]:
        new_spec.worldbody.add_light(pos=[x, -1, 3], dir=[-x, 1, -2])

    # Cache the robot spec for efficient reuse
    robot_spec = mujoco.MjSpec.from_string(xml_string)
    robot_spec.copy_during_attach = True 

    # Replicate environments
    for i in range(num_envs):
        row, col = divmod(i, envs_per_row)
        x_pos, y_pos = col * env_separation, row * env_separation
        frame = new_spec.worldbody.add_frame(pos=[x_pos, y_pos, 0])  
        frame.attach_body(robot_spec.body('torso'), str(i), '')  

    # Compile model and create data
    model = new_spec.compile()
    data = mujoco.MjData(model)

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Dynamic camera adjustments
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [
            (envs_per_row - 1) * env_separation / 2,
            (num_envs // envs_per_row) * env_separation / 2,
            0.5
        ]
        viewer.cam.distance = max(num_envs * env_separation * 0.75, 10)
        viewer.cam.azimuth, viewer.cam.elevation = 0, 0

        # Simulation loop
        while viewer.is_running():
            data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    xml_path = "ant.xml" # can try half_cheetah or humanoid
    with open(xml_path, 'r') as f:
        xml_string = f.read()
    replicate_env(num_envs=6, env_separation=3, envs_per_row=2, xml_string=xml_string)
