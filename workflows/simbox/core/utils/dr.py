"""
Domain randomization utilities for simbox workflows.

This module centralizes logic for:

- Updating table/scene pairs from precomputed JSON files (`update_scene_pair`).
- Randomizing kitchen scenes from `HEARTH_KITCHENS` (`update_hearths`/`update_scenes`).
- Randomizing articulated objects and pick objects from asset libraries
  (`update_articulated_objs`, `update_rigid_objs`, `update_conveyor_objects`),
  including:
  - Selecting random USD assets within a category or scope.
  - Reading per-object info (e.g. scale, gap) from YAML/JSON.
  - Setting orientation based on:
    - `orientation_mode == "suggested"` → uses `CATEGORIES` via `get_category_euler`.
    - `orientation_mode == "random"` → uniform random Euler in [-180, 180]^3.
    - `orientation_mode == "keep"`  → keep the existing `euler` in config.
  - Setting scale based on:
    - `scale_mode == "keep"` → keep config scale.
    - `scale_mode == "suggested"` → use `CATEGORIES_SCALE_SUGGESTED` and
      override with `OBJECT_SCALE_SUGGESTED` when available.
- Mapping raw category names (e.g. "omniobject3d-banana") to human-readable
  strings for language instructions.
- Looking up preferred category rotations with `get_category_euler`, which falls
  back to [0, 0, 0] and prints a warning if a category is unknown.
"""

import glob
import json
import os
import random
import time

import numpy as np
from core.utils.constants import (
    CATEGORIES,
    CATEGORIES_SCALE_SUGGESTED,
    HEARTH_KITCHENS,
    OBJECT_SCALE_SUGGESTED,
)
from omegaconf import OmegaConf


def update_scene_pair(cfg):
    if (
        len(cfg["arena"]["fixtures"]) == 2
        and cfg["arena"]["fixtures"][0].get("apply_randomization", False)
        and cfg["arena"]["fixtures"][1].get("apply_randomization", False)
    ):
        involved_scenes = (cfg["arena"]["involved_scenes"]).split(",")
        table_scene_pairs = []
        for involved_scene in involved_scenes:
            with open(f"{involved_scene}.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                table_scene_pairs += data

        random.seed(os.getpid() + int(time.time() * 1000))
        table_scene_dict = random.choice(table_scene_pairs)

        table_scene_info = list(table_scene_dict.values())[0]

        # Update table
        assert "table" in cfg["arena"]["fixtures"][0]["name"]
        cfg["arena"]["fixtures"][0]["path"] = table_scene_info["table"]["path"]
        cfg["arena"]["fixtures"][0]["target_class"] = table_scene_info["table"]["target_class"]
        cfg["arena"]["fixtures"][0]["scale"] = table_scene_info["table"].get("scale", [1.0, 1.0, 1.0])
        cfg["arena"]["fixtures"][0]["translation"] = table_scene_info["table"].get("translation", [0.0, 0.0, 0.0])
        cfg["arena"]["fixtures"][0]["euler"] = table_scene_info["table"].get("euler", [0.0, 0.0, 0.0])

        # Update scene
        assert "scene" in cfg["arena"]["fixtures"][1]["name"]
        cfg["arena"]["fixtures"][1]["path"] = table_scene_info["scene"]["path"]
        cfg["arena"]["fixtures"][1]["target_class"] = table_scene_info["scene"]["target_class"]
        cfg["arena"]["fixtures"][1]["scale"] = table_scene_info["scene"].get("scale", [1.0, 1.0, 1.0])
        cfg["arena"]["fixtures"][1]["translation"] = table_scene_info["scene"].get("translation", [0.0, 0.0, 0.0])
        cfg["arena"]["fixtures"][1]["euler"] = table_scene_info["scene"].get("euler", [0.0, 0.0, 0.0])
    return cfg


def update_hearths(cfg):
    for fix_cfg in cfg["arena"]["fixtures"]:
        apply_randomization = fix_cfg.get("apply_randomization", False)
        name = fix_cfg.get("name", None)
        if name == "scene" and apply_randomization:
            fix_dict = random.choice(list(HEARTH_KITCHENS.values()))

            # Substitute the scene cfg
            fix_cfg["path"] = fix_dict["path"]
            fix_cfg["target_class"] = fix_dict["target_class"]
            fix_cfg["scale"] = fix_dict["scale"]
            fix_cfg["translation"] = fix_dict["translation"]
            fix_cfg["euler"] = fix_dict["euler"]
    return cfg


def update_scenes(cfg):
    flag = False
    for obj_cfg in cfg["objects"]:
        name = obj_cfg["name"]
        if "hearth" in name:
            flag = True
    if flag:
        return update_hearths(cfg)
    else:
        return update_scene_pair(cfg)


def update_articulated_objs(cfg):
    for obj_cfg in cfg["objects"]:
        apply_randomization = obj_cfg.get("apply_randomization", False)
        if apply_randomization and obj_cfg["target_class"] == "ArticulatedObject":
            dirs = os.path.join(cfg["asset_root"], os.path.dirname(os.path.dirname(obj_cfg["path"])))
            paths = glob.glob(os.path.join(dirs, "*"))
            paths.sort()
            path = random.choice(paths)

            info_name = obj_cfg["info_name"]
            info_path = f"{path}/Kps/{info_name}/info.json"
            with open(info_path, "r", encoding="utf-8") as file:
                info = json.load(file)
            scale = info["object_scale"][:3]
            asset_root = cfg["asset_root"]

            obj_cfg["path"] = path.replace(f"{asset_root}/", "", 1) + "/instance.usd"
            obj_cfg["category"] = path.split("/")[-2]
            obj_cfg["obj_info_path"] = info_path.replace(f"{asset_root}/", "", 1)
            obj_cfg["scale"] = scale

            name = obj_cfg["path"].split("/")[-1]
            if name in [
                "microwave0",
                "microwave1",
                "microwave2",
                "microwave4",
                "microwave6",
                "microwave7",
                "microwave9",
                "microwave36754355",
                "microwave52640732",
                "microwave72789794",
                "microwave93878040",
                "microwave122930336",
                "microwave160239099",
                "microwave184070552",
                "microwave192951465",
                "microwave198542292",
                "microwave202437483",
                "microwave208204033",
                "microwave231691637",
                "microwave279963897",
                "microwave305778636",
                "microwave353130638",
                "microwave461303737",
                "microwave482895779",
            ]:
                obj_cfg["euler"] = [0.0, 0.0, 270.0]
            elif name in [
                "microwave_0001",
                "microwave_0002",
                "microwave_0003",
                "microwave_0013",
                "microwave_0044",
                "microwave_0045",
            ]:
                obj_cfg["euler"] = [0.0, 0.0, 0.0]
            elif name in [
                "microwave7119",
                "microwave7128",
                "microwave7167",
                "microwave7236",
                "microwave7263",
                "microwave7265",
                "microwave7296",
                "microwave7304",
                "microwave7310",
                "microwave7320",
            ]:
                obj_cfg["euler"] = [0.0, 0.0, 90.0]

            if "nightstand" in name:
                obj_cfg["euler"] = [0.0, 0.0, 0.0]
            elif "StorageFurniture" in name or "laptop" in name:
                obj_cfg["euler"] = [0.0, 0.0, 90.0]

    return cfg


def update_rigid_objs(cfg):
    for obj_cfg in cfg["objects"]:
        apply_randomization = obj_cfg.get("apply_randomization", False)
        if apply_randomization and obj_cfg["target_class"] == "RigidObject":
            scope = obj_cfg.get("randomization_scope", "category")
            if isinstance(scope, str):
                if scope == "category":
                    # Randomize within the same category as the current object
                    dirs = os.path.join(cfg["asset_root"], os.path.dirname(os.path.dirname(obj_cfg["path"])))
                    usds = glob.glob(os.path.join(dirs, "*", "*.usd"))
                elif scope == "full":
                    # Randomize across all categories
                    dirs = os.path.join(
                        cfg["asset_root"], os.path.dirname(os.path.dirname(os.path.dirname(obj_cfg["path"])))
                    )
                    usds = glob.glob(os.path.join(dirs, "*", "*", "*.usd"))
            elif isinstance(scope, list):
                # Randomize only from the specified list of categories
                usds = []
                for category in scope:
                    category_dir = os.path.join(
                        cfg["asset_root"], os.path.dirname(os.path.dirname(os.path.dirname(obj_cfg["path"]))), category
                    )
                    category_usds = glob.glob(os.path.join(category_dir, "*", "*.usd"))
                    usds.extend(category_usds)

            if len(usds) > 0:
                this_usd_path = random.choice(usds)
                asset_root = cfg["asset_root"]
                this_usd_path = this_usd_path.replace(f"{asset_root}/", "", 1)
                obj_cfg["path"] = this_usd_path
                tmp_category = this_usd_path.split("/")[-3]
                object_name = this_usd_path.split("/")[-2]

                gap_yaml_path = cfg["asset_root"] + "/" + os.path.join(os.path.dirname(this_usd_path), "gap.yaml")
                if os.path.exists(gap_yaml_path):
                    with open(gap_yaml_path, "r", encoding="utf-8") as file:
                        gap_data = OmegaConf.load(file)
                        gap = gap_data.get("gap", None)
                        obj_cfg["gap"] = gap

                # Update orientation
                orientation_mode = obj_cfg.get("orientation_mode", "keep")
                if orientation_mode == "suggested":
                    obj_cfg["euler"] = get_category_euler(tmp_category)
                elif orientation_mode == "random":
                    obj_cfg["euler"] = (np.random.uniform(-180, 180, size=3)).tolist()
                elif orientation_mode == "keep":
                    assert "euler" in obj_cfg, "euler not found in obj_cfg for keep mode"
                else:
                    raise NotImplementedError

                # Update scale
                scale_mode = obj_cfg.get("scale_mode", "keep")
                if scale_mode == "keep":
                    assert "scale" in obj_cfg, f"scale not found in obj_cfg for keep mode, category: {tmp_category}"
                elif scale_mode == "suggested":
                    if tmp_category in CATEGORIES_SCALE_SUGGESTED:
                        scale = CATEGORIES_SCALE_SUGGESTED[tmp_category]
                        if object_name in OBJECT_SCALE_SUGGESTED:
                            scale = OBJECT_SCALE_SUGGESTED[object_name]
                        obj_cfg["scale"] = scale

                # Update category for languages
                replace_texts = ["google_scan-", "omniobject3d-", "phocal-", "real-"]
                for replace_text in replace_texts:
                    if replace_text in tmp_category:
                        tmp_category = tmp_category.replace(replace_text, "")
                tmp_category = tmp_category.replace("_", " ")
                obj_cfg["category"] = tmp_category

    return cfg


def update_conveyor_objects(cfg):
    asset_root = cfg["asset_root"]
    for obj_cfg in cfg["objects"]:
        apply_randomization = obj_cfg.get("apply_randomization", False)
        if apply_randomization:
            dirs = os.path.join(asset_root, os.path.dirname(os.path.dirname(obj_cfg["path"])))
            usds = glob.glob(os.path.join(dirs, "*", "*.usd"))
            if len(usds) > 0:
                this_usd_path = random.choice(usds)
                this_usd_path = this_usd_path.replace(f"{asset_root}/", "", 1)
                gap_yaml_path = asset_root + "/" + os.path.join(os.path.dirname(this_usd_path), "gap.yaml")
                if os.path.exists(gap_yaml_path):
                    with open(gap_yaml_path, "r", encoding="utf-8") as file:
                        gap_data = OmegaConf.load(file)
                        gap = gap_data.get("gap", None)
                        obj_cfg["gap"] = gap
                obj_cfg["path"] = this_usd_path
                tmp_category = this_usd_path.split("/")[-3]

                # Update category for languages
                replace_texts = ["google_scan-", "omniobject3d-", "phocal-", "real-"]
                for replace_text in replace_texts:
                    if replace_text in tmp_category:
                        tmp_category = tmp_category.replace(replace_text, "")
                tmp_category = tmp_category.replace("_", " ")
                obj_cfg["category"] = tmp_category

    return cfg


def get_category_euler(category):
    if category not in CATEGORIES:
        available_categories = list(CATEGORIES.keys())
        print(
            f"[get_category_euler] Category '{category}' not found in CATEGORIES. "
            f"Available categories: {available_categories}. Using [0, 0, 0] as default."
        )
        return [0.0, 0.0, 0.0]

    euler = np.zeros(3, dtype=float)
    if "x" in CATEGORIES[category]:
        euler[0] = random.choice(CATEGORIES[category]["x"])
    if "y" in CATEGORIES[category]:
        euler[1] = random.choice(CATEGORIES[category]["y"])
    if "z" in CATEGORIES[category]:
        euler[2] = random.choice(CATEGORIES[category]["z"])

    return euler.tolist()
