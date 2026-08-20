
import mujoco
import mujoco.viewer
import numpy as np
import time

# ============================================================
# Configuration
# ============================================================

XML_FILE = "low_cost_robot_arm/scene.xml"

CUBE_NAME = "cube"
TARGET_NAME = "target"

# The body at the end of the robot arm.
END_EFFECTOR_NAME = "gripper_static_finger"

# Robot joints that we use for positioning the gripper.
JOINT_NAMES = [
    "base_rotation",
    "pitch",
    "elbow",
    "wrist_pitch",
    "wrist_roll",
]

# How far above the cube we approach from.
APPROACH_HEIGHT = 0.10

# How close the gripper needs to be.
POSITION_TOLERANCE = 0.01

# IK tuning
IK_GAIN = 2.0
DAMPING = 0.05

# Simulation timestep
DT = 0.01


# ============================================================
# Load MuJoCo
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_FILE)
data = mujoco.MjData(model)


# ============================================================
# Find bodies
# ============================================================

cube_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    CUBE_NAME
)

target_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    TARGET_NAME
)

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    END_EFFECTOR_NAME
)

if cube_id == -1:
    raise ValueError(f"Could not find cube '{CUBE_NAME}'")

if target_id == -1:
    raise ValueError(f"Could not find target '{TARGET_NAME}'")

if ee_id == -1:
    raise ValueError(
        f"Could not find end effector '{END_EFFECTOR_NAME}'"
    )


# ============================================================
# Find joints and actuators
# ============================================================

joint_ids = []
actuator_ids = []

for joint_name in JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(
            f"Could not find joint '{joint_name}'"
        )

    if actuator_id == -1:
        raise ValueError(
            f"Could not find actuator '{joint_name}'"
        )

    joint_ids.append(joint_id)
    actuator_ids.append(actuator_id)


gripper_actuator = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper"
)


# ============================================================
# Helper functions
# ============================================================

def body_position(body_id):
    """
    Get the world-space XYZ position of a body.
    """
    return data.xpos[body_id].copy()


def get_joint_positions():
    """
    Get current positions of the robot's arm joints.
    """

    q = np.zeros(len(joint_ids))

    for i, joint_id in enumerate(joint_ids):

        qpos_address = model.jnt_qposadr[joint_id]

        q[i] = data.qpos[qpos_address]

    return q


def set_joint_targets(q):
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


def set_gripper(open_gripper):
    """
    Control the gripper.

    Based on your XML:
        range="-1.60 0.032"

    0.0  = open
    -1.0 = closed
    """

    if open_gripper:
        data.ctrl[gripper_actuator] = 0.0
    else:
        data.ctrl[gripper_actuator] = -1.0


def get_position_jacobian():
    """
    Calculate the translational Jacobian of the
    gripper.

    J maps joint velocity to end-effector velocity.
    """

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
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


# ============================================================
# Inverse Kinematics
# ============================================================

def move_gripper_to(target_position, viewer):
    """
    Move the robot gripper toward target_position
    using damped least-squares inverse kinematics.
    """

    print(f"Moving gripper to {target_position}")

    for step in range(2000):

        # Current gripper position
        current_position = body_position(ee_id)

        # XYZ error
        error = target_position - current_position

        distance = np.linalg.norm(error)

        # We reached the target
        if distance < POSITION_TOLERANCE:

            print(
                f"Reached position. Error = {distance:.4f}"
            )

            return True

        # ----------------------------------------------------
        # Calculate Jacobian
        # ----------------------------------------------------

        J = get_position_jacobian()

        # ----------------------------------------------------
        # Damped least-squares IK
        # ----------------------------------------------------

        JJt = J @ J.T

        damping = (
                DAMPING ** 2
                * np.eye(3)
        )

        joint_velocity = (
                J.T
                @ np.linalg.solve(
            JJt + damping,
            error
        )
        )

        joint_velocity *= IK_GAIN

        # Prevent excessively large movements
        joint_velocity = np.clip(
            joint_velocity,
            -1.0,
            1.0
        )

        # Current joint positions
        q = get_joint_positions()

        # New desired joint positions
        q_target = q + joint_velocity * DT

        # ----------------------------------------------------
        # Respect joint limits
        # ----------------------------------------------------

        for i, joint_id in enumerate(joint_ids):

            if model.jnt_limited[joint_id]:

                minimum = model.jnt_range[
                    joint_id,
                    0
                ]

                maximum = model.jnt_range[
                    joint_id,
                    1
                ]

                q_target[i] = np.clip(
                    q_target[i],
                    minimum,
                    maximum
                )

        # ----------------------------------------------------
        # Send commands to robot
        # ----------------------------------------------------

        set_joint_targets(q_target)

        # Keep gripper open while moving
        set_gripper(open_gripper=True)

        # Advance simulation
        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(DT)

    print("WARNING: Could not reach target.")

    return False


# ============================================================
# Pick and Place
# ============================================================

with mujoco.viewer.launch_passive(
        model,
        data
) as viewer:

    # --------------------------------------------------------
    # Initialize physics
    # --------------------------------------------------------

    mujoco.mj_forward(model, data)

    cube_position = body_position(cube_id)
    target_position = body_position(target_id)

    print()
    print("================================")
    print("MuJoCo Pick and Place")
    print("================================")

    print("Cube:")
    print(cube_position)

    print("Target:")
    print(target_position)

    # --------------------------------------------------------
    # Open gripper
    # --------------------------------------------------------

    set_gripper(open_gripper=True)

    # --------------------------------------------------------
    # Let simulation settle
    # --------------------------------------------------------

    for _ in range(100):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(DT)


    # ========================================================
    # STEP 1
    # Move above cube
    # ========================================================

    cube_position = body_position(cube_id)

    above_cube = cube_position.copy()

    above_cube[2] += APPROACH_HEIGHT

    print("\n1. Moving above cube...")

    move_gripper_to(
        above_cube,
        viewer
    )


    # ========================================================
    # STEP 2
    # Lower toward cube
    # ========================================================

    cube_position = body_position(cube_id)

    grasp_position = cube_position.copy()

    grasp_position[2] += 0.035

    print("\n2. Moving toward cube...")

    move_gripper_to(
        grasp_position,
        viewer
    )


    # ========================================================
    # STEP 3
    # Close gripper
    # ========================================================

    print("\n3. Closing gripper...")

    set_gripper(open_gripper=False)

    for _ in range(200):

        # Hold arm position
        q = get_joint_positions()

        set_joint_targets(q)

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(DT)


    # ========================================================
    # STEP 4
    # Lift cube
    # ========================================================

    cube_position = body_position(cube_id)

    lift_position = cube_position.copy()

    lift_position[2] += APPROACH_HEIGHT

    print("\n4. Lifting cube...")

    move_gripper_to(
        lift_position,
        viewer
    )


    # ========================================================
    # STEP 5
    # Move above target
    # ========================================================

    target_position = body_position(target_id)

    above_target = target_position.copy()

    above_target[2] += APPROACH_HEIGHT

    print("\n5. Moving to target...")

    move_gripper_to(
        above_target,
        viewer
    )


    # ========================================================
    # STEP 6
    # Lower cube onto target
    # ========================================================

    target_position = body_position(target_id)

    place_position = target_position.copy()

    place_position[2] += 0.035

    print("\n6. Lowering cube...")

    move_gripper_to(
        place_position,
        viewer
    )


    # ========================================================
    # STEP 7
    # Release cube
    # ========================================================

    print("\n7. Opening gripper...")

    set_gripper(open_gripper=True)

    for _ in range(200):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(DT)


    print()
    print("================================")
    print("Pick and place complete!")
    print("================================")


    # Keep viewer open
    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(DT)
