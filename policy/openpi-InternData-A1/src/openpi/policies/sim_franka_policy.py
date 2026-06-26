import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from pdb import set_trace


@dataclasses.dataclass(frozen=True)
class SimFrankaInputs(transforms.DataTransformFn):
    """Inputs for the Franka policy.
    """

    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_franka(data, adapt_to_pi=self.adapt_to_pi)
        if "images" in data:
            in_images = data["images"]
            if set(in_images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

            # Assume that base image always exists.
            base_image = in_images["cam_high"]

            images = {
                "base_0_rgb": base_image,
            }
            image_masks = {
                "base_0_rgb": np.True_,
            }

            # Add the extra images.
            extra_image_names = {
                "left_wrist_0_rgb": "cam_left_wrist",
                "right_wrist_0_rgb": "cam_right_wrist",
            }
            for dest, source in extra_image_names.items():
                if source in in_images:
                    images[dest] = in_images[source]
                    image_masks[dest] = np.True_
                else:
                    images[dest] = np.zeros_like(base_image)
                    image_masks[dest] = np.False_

            inputs = {
                "image": images,
                "image_mask": image_masks,
                "state": data["state"],
                "pose": data["pose"],
            }
        else:
            inputs = {
                "state": data["state"],
                "pose": data["pose"],
            }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SimFrankaOutputs(transforms.DataTransformFn):
    """Outputs for the Lift2 policy."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_franka(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state_dict = data["state_dict"]
    data["state"], data["pose"] = _decode_state(state_dict, adapt_to_pi=adapt_to_pi)
    del data["state_dict"]
    action_dict = data["action_dict"]
    data["actions"] = _decode_action(action_dict, adapt_to_pi=adapt_to_pi)
    del data["action_dict"]

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")
    if "images" in data:
        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}

        data["images"] = images_dict
    return data


def _decode_state(state, *, adapt_to_pi: bool = False) -> np.ndarray:
    gripper_position = state["gripper_position"][None]
    gripper_pose = state["gripper_pose"]
    joint_position = state["joint_position"]
    state = np.concatenate([joint_position, gripper_position], axis=0)
    pose = np.concatenate([gripper_pose, gripper_position], axis=0)
    return state, pose

def _decode_action(action, *, adapt_to_pi: bool = False) -> np.ndarray:
    gripper_pose = action["gripper_pose"]
    gripper_openness = action["gripper_openness"][..., None]
    action = np.concatenate([gripper_pose, gripper_openness], axis=1)
    return action


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    return actions
