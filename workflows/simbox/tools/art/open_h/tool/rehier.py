# pylint: skip-file
# flake8: noqa
import os
import json
import argparse
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from pdb import set_trace

def remove_articulation_root(prim: Usd.Prim):
    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)  
    allchildren = prim.GetAllChildren()
    if len(allchildren) == 0:
        return 
    else:
        for child in allchildren:
            remove_articulation_root(child)

def remove_rigidbody(prim: Usd.Prim):
    prim.RemoveAPI(UsdPhysics.RigidBodyAPI) 
    allchildren = prim.GetAllChildren()
    if len(allchildren) == 0:
        return
    else:
        for child in allchildren:
            remove_rigidbody(child)

def remove_mass(prim: Usd.Prim):
    prim.RemoveAPI(UsdPhysics.MassAPI)
    allchildren = prim.GetAllChildren()
    if len(allchildren) == 0:
        return
    else:
        for child in allchildren:
            remove_mass(child)

def add_rigidbody(prim: Usd.Prim):
    UsdPhysics.RigidBodyAPI.Apply(prim)

def add_mass(prim: Usd.Prim):
    UsdPhysics.MassAPI.Apply(prim)
    mass = prim.GetAttribute("physics:mass")
    mass.Clear()

def get_args():
    parser = argparse.ArgumentParser(description="USD Hierarchy and Physics Editor")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def safe_rename_prim(stage, old_path, new_path):
    editor = Usd.NamespaceEditor(stage)
    old_p = stage.GetPrimAtPath(old_path)
    editor.RenamePrim(old_p, new_path.split('/')[-1])

    if editor.CanApplyEdits():
        editor.ApplyEdits()
        return True
    else:
        return False

def modify_hierarchy(stage, config):
    editor = Usd.NamespaceEditor(stage)

    """ Modify USD hierarchy structure """
    base_path = config["base_initial_prim_path"]
    link0_path = config["link0_initial_prim_path"]
    revolute_joint_path = config["revolute_joint_initial_prim_path"]

    # Get original root node
    old_root_path = f"/{link0_path.split('/')[1]}"
    instance_path = "/root/instance"
    safe_rename_prim(stage, '/{}'.format(stage.GetDefaultPrim().GetName()), "/instance")

    return instance_path

def modify_physics(stage, instance_root_path, config):
    """ Modify physics properties and ensure colliders are uniformly set to Convex Hull """
    print(f"Applying physics modifications and setting colliders to ConvexHull...")

    # Traverse all nodes for processing
    for prim in stage.Traverse():
        # 1. Clear instancing (if editing physics properties of individual instances)
        if prim.IsInstanceable():
            prim.ClearInstanceable()

        # 2. Process Mesh colliders
        if prim.IsA(UsdGeom.Mesh):
            # Ensure base collision API is applied
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

            # Force apply MeshCollisionAPI to set collision approximation
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
            if not mesh_collision_api:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

            # Set collision approximation to 'convexHull'
            # Optional values include: 'none', 'convexHull', 'convexDecomposition', 'meshSimplification', etc.
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

            # Ensure physics collision is enabled
            col_enabled_attr = prim.GetAttribute("physics:collisionEnabled")
            if not col_enabled_attr.HasValue():
                col_enabled_attr.Set(True)

def create_fixed_joint(stage, joint_path, body0_path, body1_path):
    """
    Create a FixedJoint at the specified path and connect two rigid bodies.
    """
    # 1. Define FixedJoint node
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # 2. Set Body0 and Body1 path references
    # Note: Paths must be of Sdf.Path type
    fixed_joint.GetBody0Rel().SetTargets([Sdf.Path(body0_path)])
    fixed_joint.GetBody1Rel().SetTargets([Sdf.Path(body1_path)])

    # 3. (Optional) Set local offset (Local Pose)
    # If not set, the joint defaults to the origin of both objects
    # fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    # fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))

    print(f"Successfully created FixedJoint: {joint_path}")
    print(f"  Connected: {body0_path} <---> {body1_path}")
    return fixed_joint

def process_joints(stage, revolute_joint_initial_prim_path):
    # 1. Collect paths to process
    paths_to_delete = []
    joints_to_convert = []

    # Use TraverseAll to ensure no defined nodes are missed
    for prim in stage.Traverse():
        # Check if it is a physics joint
        if prim.IsA(UsdPhysics.Joint):
            path = prim.GetPath()

            # Logic A: If FixedJoint -> delete
            if prim.IsA(UsdPhysics.FixedJoint):
                paths_to_delete.append(path)

            # Logic B: If not FixedJoint and path does not contain 'contact_link' -> convert to FixedJoint
            # elif "contact_link" not in str(path).lower():
            #     joints_to_convert.append(path)
            elif str(path) != revolute_joint_initial_prim_path:
                joints_to_convert.append(path)

            print(str(path))

    # 2. Get current edit layer
    layer = stage.GetEditTarget().GetLayer()
    edit = Sdf.BatchNamespaceEdit()

    # Execute deletion logic
    for path in paths_to_delete:
        edit.Add(path, Sdf.Path.emptyPath)
        print(f"[Delete] FixedJoint: {path}")

    # 3. Apply deletion edits
    if paths_to_delete:
        layer.Apply(edit)

    # 4. Execute type conversion logic
    # In USD, changing type usually means re-Defining the new type at that path
    for path in joints_to_convert:
        # Record original Body0 and Body1 relationships to prevent loss after conversion
        prim = stage.GetPrimAtPath(path)
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()

        # Redefine as FixedJoint
        new_fixed_joint = UsdPhysics.FixedJoint.Define(stage, path)

        # Restore relationships
        if body0: new_fixed_joint.GetBody0Rel().SetTargets(body0)
        if body1: new_fixed_joint.GetBody1Rel().SetTargets(body1)

        safe_rename_prim(stage, str(new_fixed_joint.GetPath()),  "/FixedJoint")

        print(f"[Convert] Regular joint -> FixedJoint: {path}")

    return stage

def final_refine(stage, output_usd_path, revolute_joint_initial_prim_path):
    root_prim = stage.GetPrimAtPath("/root")
    instance_prim = stage.GetPrimAtPath("/root/instance")  
    ### remove articulation root ###
    remove_articulation_root(root_prim)

    ### remove rigid body ###   
    remove_rigidbody(root_prim) 

    ### remove mass ###
    # remove_mass(root_prim)

    ### add rigid body and mass ###
    for child in instance_prim.GetAllChildren():   
        
        if child.GetTypeName() == "PhysicsRevoluteJoint" or child.GetTypeName() == "PhysicsPrismaticJoint":
            continue

        if child.GetTypeName() == "Xform" :
            print('name:', child.GetTypeName())
            add_rigidbody(child)

    stage = process_joints(stage, revolute_joint_initial_prim_path)

    ### add articulation root ###
    UsdPhysics.ArticulationRootAPI.Apply(instance_prim)
    stage.SetDefaultPrim(root_prim)
    
    for child in instance_prim.GetAllChildren():
        try:
            attr = child.GetAttribute('physics:jointEnabled')
        except:
            continue
        if attr.Get() is not None:
            print(child)
            attr.Set(True)

    modify_physics(stage, "/root/instance", 11)
    stage.Export(output_usd_path)

    return stage

def import_as_copy(source_usd_path, output_usd_path, root_name="root", sub_node_name="instance"):
    """
    Create a new USD and copy the content from source_usd_path to /root/sub_node_name.
    """
    # 1. Create target Stage and root node
    stage = Usd.Stage.CreateNew(output_usd_path)
    root_path = Sdf.Path(f"/{root_name}")
    UsdGeom.Xform.Define(stage, root_path)

    # 2. Define copy destination path (e.g., /root/model_copy)
    dest_path = root_path.AppendChild(sub_node_name)

    # 3. Open source file layer
    source_layer = Sdf.Layer.FindOrOpen(source_usd_path)

    if not source_layer:
        print(f"Error: Cannot find source file {source_usd_path}")
        return

    # 4. Get source file's default prim (DefaultPrim) as copy target
    # If source file has no default prim, use the first root prim
    source_root_name = list(source_layer.rootPrims)[0].name
    source_path = Sdf.Path(f"/{source_root_name}")

    # 5. Execute core copy operation (Sdf.CopySpec)
    # This copies all attributes, topology, and properties from source file to new file
    Sdf.CopySpec(source_layer, source_path, stage.GetRootLayer(), dest_path)

    # 6. Set default prim and save
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))
    stage.GetRootLayer().Save()
    print(f"Success! Content copied to: {output_usd_path}")

    return stage, output_usd_path

def main():
    args = get_args()
    config = load_config(args.config)

    dir_name = config["DIR"]
    usd_path = os.path.join(dir_name, config["USD_NAME"])
    output_path = os.path.join(dir_name, "instance.usd")
    revolute_joint_initial_prim_path = (config["revolute_joint_initial_prim_path"]).replace("root", "root/instance")

    # --- Key improvement: Open Stage using Flatten ---
    # This writes all reference data directly into the main layer, preventing reference loss after path changes
    base_stage = Usd.Stage.Open(usd_path)

    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().TransferContent(base_stage.Flatten())

    # 1. Modify hierarchy
    instance_path = modify_hierarchy(stage, config)

    # 3. Export
    print(f"Exporting to: {output_path}")
    stage.GetRootLayer().Export(output_path)
    
    stage, output_usd_path = import_as_copy(output_path, output_path.replace('.usd', '_refiened.usd'))
    stage = final_refine(stage, output_usd_path.replace('.usd', '_final.usd'), revolute_joint_initial_prim_path)

    stage.Export(output_path)

    os.remove(output_path.replace('.usd', '_refiened.usd'))
    os.remove(output_usd_path.replace('.usd', '_final.usd'))

    print("Done.")


if __name__ == "__main__":
    main()