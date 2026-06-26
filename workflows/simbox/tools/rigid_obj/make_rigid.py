import argparse

from pxr import Usd, UsdPhysics


def process(
    usd_file,
    add_rigid_body=True,
):
    """Add rigid body physics properties to a USD file.

    Args:
        usd_file: Path to the USD file.
        add_rigid_body: Whether to add rigid body properties.
    """
    stage = Usd.Stage.Open(usd_file)
    editor = Usd.NamespaceEditor(stage)
    root_prim = stage.GetDefaultPrim()
    aligned_prim = root_prim.GetAllChildren()[1]
    try:
        editor.RenamePrim(aligned_prim, "Aligned")
        editor.ApplyEdits()
    except Exception:
        # If rename fails (e.g., already renamed), continue with existing prim
        aligned_prim = stage.GetPrimAtPath("/World/Aligned")

    aligned_prim = stage.GetPrimAtPath("/World/Aligned")

    if add_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(aligned_prim)

        UsdPhysics.MassAPI.Apply(aligned_prim)
        mass_attr = aligned_prim.GetAttribute("physics:mass")
        mass_attr.Set(0.1)

    stage.Save()
    print(f"Successfully processed: {usd_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add rigid body physics properties to USD file")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to the USD file")
    args = parser.parse_args()

    process(args.usd_path, True)
