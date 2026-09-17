import mujoco
import mujoco.viewer
import numpy as np


# Helper movement functions
def body_position(body_id,data):
    """
    Get the world-space XYZ position of a body.
    """
    return data.xpos[body_id].copy()

def site_position(site_id,data):
    """
    Get the world-space XYZ position of a site.
    """
    return data.site_xpos[site_id].copy()

def get_joint_positions(joint_ids,data,model):
    """
    Get current positions of the robot's arm joints.
    """

    q = np.zeros(len(joint_ids))

    for i, joint_id in enumerate(joint_ids):

        qpos_address = model.jnt_qposadr[joint_id]

        q[i] = data.qpos[qpos_address]

    return q

def set_joint_targets(q, actuator_ids,model,data):
    """
    Send desired joint positions to the MuJoCo
    position actuators.
    """

    for i, actuator_id in enumerate(actuator_ids):

        # Respect actuator limits
        low, high = model.actuator_ctrlrange[actuator_id]

        target = np.clip(
            q[i],
            low,
            high
        )

        data.ctrl[actuator_id] = target

def set_gripper(gripper_actuator,data,open_gripper):
    """
    Control the gripper.

    Based on your XML:
        range="-1.60 0.032"

    0.0  = close
    -1.0 = open
    """

    if open_gripper:
        data.ctrl[gripper_actuator] = -0.7
    else:
        data.ctrl[gripper_actuator] = 0.032

def get_position_jacobian(ee_id,joint_ids,model,data):
    """
    Calculate the translational Jacobian of the
    gripper.

    J maps joint velocity to end-effector velocity.
    """

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        ee_id
    )

    J = np.zeros((3, len(joint_ids)))

    for i, joint_id in enumerate(joint_ids):

        dof_address = model.jnt_dofadr[joint_id]

        J[:, i] = jacobian_position[:, dof_address]

    return J



