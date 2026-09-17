import mujoco
import numpy as np
import motor_pid
import time

# ============================================================
# Configuration
# ============================================================
XML_FILE = "low_cost_robot_arm/scene.xml"
CUBE_NAME = "cube"
TARGET_NAME = "target"
END_EFFECTOR_NAME = "gripper_tip" # Site between the finger pads — the actual grasp point.
# Robot joints that we use for positioning the gripper.
JOINT_NAMES = [
    "base_rotation",
    "pitch",
    "elbow",
    "wrist_pitch",
    "wrist_roll",
]
APPROACH_HEIGHT = 0.05 # How far above the cube we approach from.
POSITION_TOLERANCE = 0.001 # How close the gripper needs to be.

# ============================================================
# Load MuJoCo
# ============================================================
model = mujoco.MjModel.from_xml_path(XML_FILE)
data = mujoco.MjData(model)

# Use the model's actual physics timestep for real-time pacing —
# this is NOT the same thing as PLANNING_HORIZON above.
DT = model.opt.timestep
DAMPING = 0.03
# IK tuning
IK_GAIN = 1.0

# How far ahead (in seconds) the position command "leads" the actuator.
# This is intentionally larger than DT: a position actuator only produces
# torque proportional to (target - current), so if the per-step target is
# only DT worth of motion away, the resulting torque is tiny (kp * a few
# thousandths of a radian) and gets lost to gravity/friction. Leading the
# target further ahead gives the PD controller a real error to act on.
# Tune this: too small -> sluggish/stalls, too large -> overshoot/oscillation.
PLANNING_HORIZON = 0.05

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
    mujoco.mjtObj.mjOBJ_SITE,
    END_EFFECTOR_NAME
)


if cube_id == -1:
    raise ValueError(f"Could not find cube '{CUBE_NAME}'")

if target_id == -1:
    raise ValueError(f"Could not find target '{TARGET_NAME}'")

if ee_id == -1:
    raise ValueError(f"Could not find end effector '{END_EFFECTOR_NAME}'")

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

# Inverse Kinematics
def move_gripper_to(target_position, viewer, max_seconds=20.0):
    """
    Move the robot gripper toward target_position
    using damped least-squares inverse kinematics.
    """

    print(f"Moving gripper to {target_position}")

    max_steps = int(max_seconds / DT)

    for step in range(max_steps):
        #tip_offset = np.array([0.0, 0.0, -0.05])

        # Current gripper position (tracked at the fingertip site, not the wrist body)
        current_position = motor_pid.site_position(ee_id,data)

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

        J = motor_pid.get_position_jacobian(ee_id,joint_ids,model,data)

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
        q = motor_pid.get_joint_positions(joint_ids,data,model)

        # New desired joint positions — lead the target ahead by
        # PLANNING_HORIZON so the PD actuator sees a real error to act on
        # (see comment at the top of the file).
        q_target = q + joint_velocity * PLANNING_HORIZON

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

        motor_pid.set_joint_targets(q_target,actuator_ids,model,data)

        # Keep gripper open while moving
        #set_gripper(open_gripper=True)

        # Advance simulation
        mujoco.mj_step(model, data)

        if step % 200 == 0:
            forces = [data.actuator_force[a] for a in actuator_ids]
            ranges = [model.actuator_forcerange[a] for a in actuator_ids]
            saturated = [
                JOINT_NAMES[i] for i, f in enumerate(forces)
                if abs(f) >= 0.95 * ranges[i][1]
            ]
            #print(f"  step {step}: dist={distance:.4f} saturated={saturated}")

        viewer.sync()

        time.sleep(DT)

    print("WARNING: Could not reach target.")

    return False



def pick_and_drop():
    # Initialize physics
    mujoco.mj_forward(model, data)
    cube_position = motor_pid.body_position(cube_id,data)
    target_position = motor_pid.body_position(target_id,data)

    print()
    print("================================")
    print("MuJoCo Pick and Place")
    print("================================")

    print("Cube:")
    print(cube_position)

    print("Target:")
    print(target_position)

    # Open gripper
    motor_pid.set_gripper(gripper_actuator,data,open_gripper=True)
    # Let simulation settle
    #for _ in range(100):
    #    mujoco.mj_step(model, data)
    #    viewer.sync()
    #    time.sleep(DT)

    # STEP 1 : Move above cube
    print("\n1. Moving above cube...")
    cube_position = motor_pid.body_position(cube_id,data)
    above_cube = cube_position.copy()
    above_cube[1] += 0.02
    above_cube[2] += APPROACH_HEIGHT

    move_gripper_to(above_cube,viewer)

    # STEP 2 : Lower to just above the cube (pre-grasp)
    print("\n2. Moving above cube (pre-grasp)...")
    pre_grasp_position = above_cube.copy()
    pre_grasp_position[2] = cube_position[2] - 0.01
    move_gripper_to(pre_grasp_position,viewer)

    # STEP 3 : Close gripper

    print("\n3. Closing gripper...")
    motor_pid.set_gripper(gripper_actuator,data,open_gripper=False)
    for _ in range(200):
        # Hold arm position
        q = motor_pid.get_joint_positions(joint_ids,data,model)
        motor_pid.set_joint_targets(q,actuator_ids,model,data)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)

    # STEP 4 : Lift cube
    print("\n4. Lifting cube...")
    lift_position = above_cube.copy()
    lift_position[2] += APPROACH_HEIGHT
    move_gripper_to(lift_position,viewer)

    # STEP 5 : Move above target
    print("\n5. Moving to target...")
    target_position = motor_pid.body_position(target_id,data)
    above_target = target_position.copy()
    above_target[2] += APPROACH_HEIGHT
    move_gripper_to(above_target,viewer)

    # STEP 6 : Lower cube onto target
    print("\n6. Lowering cube...")
    target_position = motor_pid.body_position(target_id,data)
    place_position = target_position.copy()
    move_gripper_to(place_position,viewer)

    # STEP 7 : Release cube
    print("\n7. Opening gripper...")
    start_position = motor_pid.body_position(gripper_actuator,data)
    motor_pid.set_gripper(gripper_actuator,data,open_gripper=True)
    #for _ in range(200):
    #    mujoco.mj_step(model, data)
    #    viewer.sync()
    #    time.sleep(DT)

    # reset arm
    move_gripper_to(start_position,viewer)
    print()
    print("================================")
    print("Pick and place complete!")
    print("================================")

    # Keep viewer open
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)

if __name__ == "__main__":
    with mujoco.viewer.launch_passive(
            model,
            data
    ) as viewer:

        pick_and_drop()