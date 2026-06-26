#!/usr/bin/env python3
"""
Batch generate YAML config files for pick_and_place tasks.
Replaces asset names and USD paths based on available assets.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
ASSETS_DIR = Path("workflows/simbox/assets/pick_and_place/pre-train-pick/assets")
FUNCTIONAL_ASSETS_DIR = Path("workflows/simbox/assets/pick_and_place/functional-pick-assets")
CONFIG_BASE_DIR = Path("workflows/simbox/core/configs/tasks/pick_and_place")

# Exclude these asset folders from pre-train-pick
EXCLUDE_ASSETS = ["google_scan-book", "google_scan-box", "omniobject3d-rubik_cube-old"]

# Template configs: (robot, task_type, arm_side, template_path, template_asset_name, assets_dir_type)
# task_type: "single_pick", "single_pnp", "single_func_pick", etc.
# arm_side: "left", "right", or None (for single-arm robots like franka)
# assets_dir_type: "pre-train" or "functional"
TEMPLATE_CONFIGS = [
    # lift2 - single_pick (pre-train assets)
    ("lift2", "single_pick", "left", "lift2/single_pick/left/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    ("lift2", "single_pick", "right", "lift2/single_pick/right/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    # split_aloha - single_pick (pre-train assets)
    ("split_aloha", "single_pick", "left", "split_aloha/single_pick/left/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    ("split_aloha", "single_pick", "right", "split_aloha/single_pick/right/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    # franka - single_pick (pre-train assets, single arm, no left/right)
    ("franka", "single_pick", None, "franka/single_pick/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    # genie1 - single_pick (pre-train assets)
    ("genie1", "single_pick", "left", "genie1/single_pick/left/omniobject3d-lemon.yaml", "omniobject3d-lemon", "pre-train"),
    ("genie1", "single_pick", "right", "genie1/single_pick/right/omniobject3d-lemon.yaml", "omniobject3d-lemon", "pre-train"),
    # genie1 - single_pnp (pre-train assets, pick and place)
    ("genie1", "single_pnp", "left", "genie1/single_pnp/left/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
    ("genie1", "single_pnp", "right", "genie1/single_pnp/right/omniobject3d-banana.yaml", "omniobject3d-banana", "pre-train"),
]

# Functional pick template configs (uses functional-pick-assets)
FUNCTIONAL_TEMPLATE_CONFIGS = [
    # genie1 - single_func_pick (functional assets)
    ("genie1", "single_func_pick", "left", "genie1/single_func_pick/left/omniobject3d-hammer.yaml", "omniobject3d-hammer", "functional"),
    ("genie1", "single_func_pick", "right", "genie1/single_func_pick/right/omniobject3d-hammer.yaml", "omniobject3d-hammer", "functional"),
]


def get_all_assets(assets_dir: Path, exclude: List[str] = None) -> List[str]:
    """Get all asset folder names, excluding specified ones."""
    if exclude is None:
        exclude = []
    assets = []
    for item in sorted(assets_dir.iterdir()):
        if item.is_dir() and item.name not in exclude:
            assets.append(item.name)
    return assets


def get_first_usd_path(assets_dir: Path, asset_name: str, assets_dir_type: str = "pre-train") -> Optional[str]:
    """Get the first USD file path for an asset."""
    asset_folder = assets_dir / asset_name
    if not asset_folder.exists():
        return None
    
    # Get all subfolders
    subfolders = sorted([f for f in asset_folder.iterdir() if f.is_dir()])
    if not subfolders:
        return None
    
    # Find the first subfolder with Aligned_obj.usd
    for subfolder in subfolders:
        usd_file = subfolder / "Aligned_obj.usd"
        if usd_file.exists():
            if assets_dir_type == "functional":
                return f"pick_and_place/functional-pick-assets/{asset_name}/{subfolder.name}/Aligned_obj.usd"
            else:
                return f"pick_and_place/pre-train-pick/assets/{asset_name}/{subfolder.name}/Aligned_obj.usd"
    
    return None


def replace_in_yaml(content: str, old_asset: str, new_asset: str, new_usd_path: str) -> str:
    """Replace asset name and USD path in YAML content.
    
    Note: Does NOT modify distractors section - it should remain unchanged.
    Only replaces paths that end with .usd (object paths, not directory paths).
    """
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Replace path line ONLY if it's a USD file path (ends with .usd)
        # This distinguishes object paths from distractor directory paths
        # Support both pre-train-pick/assets and functional-pick-assets
        if "path:" in line and "pick_and_place/" in line and ".usd" in line:
            # Replace the USD path
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f"path: {new_usd_path}")
        elif "category:" in line and old_asset in line:
            # Replace category
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f'category: "{new_asset}"')
        elif "task_dir:" in line and old_asset in line:
            # Replace task_dir
            indent = len(line) - len(line.lstrip())
            # Extract the directory structure before the asset name
            old_task_dir = line.split('"')[1]
            new_task_dir = old_task_dir.replace(old_asset, new_asset)
            new_lines.append(" " * indent + f'task_dir: "{new_task_dir}"')
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def generate_configs(
    assets: List[str],
    template_configs: List[tuple],
    config_base_dir: Path,
    assets_dir: Path,
    dry_run: bool = True
):
    """Generate YAML config files for all assets."""
    
    for robot, task_type, arm_side, template_rel_path, template_asset_name, assets_dir_type in template_configs:
        template_path = config_base_dir / template_rel_path
        
        if not template_path.exists():
            print(f"[WARNING] Template not found: {template_path}")
            continue
        
        # Read template content
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Determine output directory based on task_type and arm_side
        if arm_side is None:
            # Single-arm robot (like franka)
            output_dir = config_base_dir / robot / task_type
        else:
            # Dual-arm robot with left/right
            output_dir = config_base_dir / robot / task_type / arm_side
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for asset_name in assets:
            # Get USD path for this asset
            usd_path = get_first_usd_path(assets_dir, asset_name, assets_dir_type)
            if usd_path is None:
                print(f"[SKIP] No USD found for {asset_name}")
                continue
            
            # Generate new content
            new_content = replace_in_yaml(
                template_content,
                template_asset_name,
                asset_name,
                usd_path
            )
            
            # Output file path
            output_file = output_dir / f"{asset_name}.yaml"
            
            if dry_run:
                print(f"[DRY-RUN] Would create: {output_file}")
                print(f"          USD path: {usd_path}")
            else:
                with open(output_file, 'w') as f:
                    f.write(new_content)
                print(f"[CREATED] {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate YAML config files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without creating files")
    parser.add_argument("--assets", nargs="+", help="Specific assets to process (default: all)")
    parser.add_argument("--robots", nargs="+", help="Specific robots to process (default: all)")
    parser.add_argument("--task-types", nargs="+", help="Specific task types to process (default: all)")
    parser.add_argument("--functional", action="store_true", help="Generate configs for functional pick assets")
    args = parser.parse_args()
    
    # Change to repo root
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    
    # Select template configs based on --functional flag
    if args.functional:
        template_configs = FUNCTIONAL_TEMPLATE_CONFIGS
        assets_dir = FUNCTIONAL_ASSETS_DIR
        exclude = []  # Don't exclude anything for functional assets
    else:
        template_configs = TEMPLATE_CONFIGS
        assets_dir = ASSETS_DIR
        exclude = EXCLUDE_ASSETS
    
    # Get assets
    if args.assets:
        assets = args.assets
    else:
        assets = get_all_assets(assets_dir, exclude)
    
    # Filter templates by robots and task_types if specified
    if args.robots:
        template_configs = [t for t in template_configs if t[0] in args.robots]
    if args.task_types:
        template_configs = [t for t in template_configs if t[1] in args.task_types]
    
    print(f"Assets directory: {assets_dir}")
    print(f"Found {len(assets)} assets to process")
    print(f"Assets: {assets[:5]}..." if len(assets) > 5 else f"Assets: {assets}")
    print(f"Templates: {[(t[0], t[1], t[2]) for t in template_configs]}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)
    
    generate_configs(
        assets=assets,
        template_configs=template_configs,
        config_base_dir=CONFIG_BASE_DIR,
        assets_dir=assets_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()