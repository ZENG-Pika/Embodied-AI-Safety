import numpy as np
from omni.isaac.core.utils.transformations import pose_from_tf_matrix
from scipy.spatial.transform import Rotation as R


def get_orientation(euler, quaternion, degrees: bool = True):
    """Return orientation as quaternion [w, x, y, z] from euler or quaternion."""
    if euler:
        if degrees:
            euler = np.radians(euler)
        r = R.from_euler("xyz", euler)
        orientation = r.as_quat(scalar_first=True)  # wxyz
    elif quaternion:
        orientation = quaternion
    else:
        orientation = [1.0, 0.0, 0.0, 0.0]
    return orientation


def perturb_position(translation, max_noise_m=0.05, num_samples=1):
    """Perturb translation with uniform random noise in [-max_noise_m, max_noise_m]."""
    if num_samples == 1:
        noise = np.random.uniform(low=-max_noise_m, high=max_noise_m, size=3)
        return translation + noise
    noise = np.random.uniform(low=-max_noise_m, high=max_noise_m, size=(num_samples, 3))
    return translation + noise  # broadcasting: (3,) + (num_samples, 3) -> (num_samples, 3)


def perturb_orientation(orientation, max_noise_deg=5, num_samples=1):
    """Perturb orientation quaternion with random rotation noise (in degrees)."""

    def quaternion_multiply(q1, q2):
        # q1: (4,) or (N, 4), q2: (4,) or (N, 4)
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.stack([w, x, y, z], axis=-1)

    max_noise_rad = np.deg2rad(max_noise_deg)

    if num_samples == 1:
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis)
        random_angle = np.random.uniform(low=-max_noise_rad, high=max_noise_rad)
        noise_quaternion = np.array([np.cos(random_angle / 2), *(np.sin(random_angle / 2) * random_axis)])
        return quaternion_multiply(orientation, noise_quaternion)

    random_axes = np.random.randn(num_samples, 3)
    random_axes /= np.linalg.norm(random_axes, axis=1, keepdims=True)
    random_angles = np.random.uniform(low=-max_noise_rad, high=max_noise_rad, size=num_samples)

    half_angles = random_angles / 2
    cos_half = np.cos(half_angles)
    sin_half_axis = np.sin(half_angles)[:, np.newaxis] * random_axes

    noise_quaternions = np.concatenate([cos_half[:, np.newaxis], sin_half_axis], axis=1)

    orientation_expanded = np.broadcast_to(orientation, (num_samples, 4))
    return quaternion_multiply(orientation_expanded, noise_quaternions)


def poses_from_tf_matrices(tf_matrices):
    # tf_matrices.shape [N, 4, 4]
    # return ee_trans_batch_np, ee_ori_batch_np
    ee_trans_list, ee_ori_list = [], []
    for idx in range(tf_matrices.shape[0]):
        ee_trans, ee_ori = pose_from_tf_matrix(tf_matrices[idx])
        ee_trans_list.append(ee_trans)
        ee_ori_list.append(ee_ori)

    ee_trans_batch_np = np.stack(ee_trans_list)
    ee_ori_batch_np = np.stack(ee_ori_list)

    return (ee_trans_batch_np, ee_ori_batch_np)


def create_pose_matrices(pos_batch, rot_batch):
    mats = np.tile(np.eye(4), (len(pos_batch), 1, 1))
    mats[:, :3, :3] = rot_batch
    mats[:, :3, 3] = pos_batch
    return mats


def pose_to_6d(pose, degrees: bool = False):
    """Convert 4x4 pose matrix to 6D representation [x, y, z, roll, pitch, yaw]."""
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def _6d_to_pose(pose6d, degrees: bool = False):
    """Convert 6D representation [x, y, z, roll, pitch, yaw] back to 4x4 pose matrix."""
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose


def get_tf_mat(i, dh):
    """Compute single Denavit–Hartenberg transform matrix for index i."""
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta
    return np.array(
        [
            [np.cos(q), -np.sin(q), 0, a],
            [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
            [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
            [0, 0, 0, 1],
        ]
    )


def get_fk_solution(joint_angles, target_index: int = 8):
    """Simple Franka forward-kinematics using hard-coded DH parameters."""
    dh_params = [
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        [0, 0.107, 0, 0],
        [0, 0, 0, -np.pi / 4],
        [0.0, 0.1034, 0, 0],
    ]

    T = np.eye(4)
    for idx in range(target_index):
        T = T @ get_tf_mat(idx, dh_params)
    return T
