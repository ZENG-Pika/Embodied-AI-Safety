"""Convert MANO URDF hand model to USD for use in Isaac Sim.

Must run with Isaac Sim's python:
    cd /home/pika/Workspace/pika/InternDataEngine
    /home/pika/Software/isaacsim4.5/python.sh scripts/convert_mano_to_usd.py
"""
import os


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    # Isaac Sim 4.5 URDF importer
    from omni.importer.urdf.urdf_importer import UrdfImporter

    urdf_path = os.path.abspath("InternDataAssets/assets/mano_urdf/mano.urdf")
    output_dir = os.path.abspath("InternDataAssets/assets/mano_urdf/usd")
    output_path = os.path.join(output_dir, "mano_hand.usd")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Importing URDF: {urdf_path}")
    print(f"Output USD: {output_path}")

    importer = UrdfImporter()
    importer.import_urdf(
        urdf_path,
        output_path,
        merge_fixed_joints=False,
        import_inertia_tensor=True,
        fix_base=True,
    )

    if os.path.exists(output_path):
        print(f"OK MANO hand USD saved to: {output_path}")
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size} bytes)")
    else:
        print("FAIL Conversion failed - trying alternative method")
        # Fallback: use omni.kit commands
        try:
            import omni.kit.app
            from omni.kit.commands import execute

            result = execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config={
                    "merge_fixed_joints": False,
                    "import_inertia_tensor": True,
                    "fix_base": True,
                },
                dest_path=output_path,
            )
            if result:
                print(f"OK via commands: {output_path}")
        except Exception as e:
            print(f"FAIL Alternative method failed: {e}")

    app.close()


if __name__ == "__main__":
    main()
