"""
Language instruction utilities for simbox workflows.

This module provides a small helper to post-process natural language
instructions based on the current randomized objects in a task config.

- update_language:
  - Reads `language_instruction` and `detailed_language_instruction` templates
    from `cfg["data"]` (semicolon-separated list).
  - For each object with `apply_randomization=True`, replaces occurrences of
    the object `name` in the templates with its (possibly normalized) `category`
    field, so that instructions stay consistent with the randomized assets.
  - Returns two lists: updated high-level and detailed language instructions.
"""


def update_language(cfg):
    language_instructions = cfg["data"].get("language_instruction", "Pick up the ${objects.0.name} with left arm")
    detailed_language_instructions = cfg["data"].get(
        "detailed_language_instruction", "Grasp and lift the ${objects.0.name} with the left robotic arm"
    )

    language_instructions = language_instructions.split(";")
    detailed_language_instructions = detailed_language_instructions.split(";")

    result_language_instructions = []
    result_detailed_language_instructions = []

    for language_instruction, detailed_language_instruction in zip(
        language_instructions, detailed_language_instructions
    ):
        for obj_cfg in cfg["objects"]:
            apply_randomization = obj_cfg.get("apply_randomization", False)
            if apply_randomization:
                category = obj_cfg["category"]
                name = obj_cfg["name"]

                language_instruction = language_instruction.replace(name, category)
                detailed_language_instruction = detailed_language_instruction.replace(name, category)

        result_language_instructions.append(language_instruction)
        result_detailed_language_instructions.append(detailed_language_instruction)

    return result_language_instructions, result_detailed_language_instructions
