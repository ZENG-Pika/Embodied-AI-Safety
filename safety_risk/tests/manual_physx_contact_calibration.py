"""Standalone Isaac/PhysX contact-unit calibration.

Runs a 0.5 kg cube at rest on a ground plane with physics_dt=1/30 s and
prints the theoretical support force, tensor contact values, and native
contact-report impulses.  This is a diagnostic script, not a pytest test.
"""

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidContactView
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import physicsUtils


DT = 1.0 / 30.0
MASS_KG = 0.5

world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/calibration_cube",
        name="calibration_cube",
        position=np.array([0.0, 0.0, 0.05]),
        scale=np.array([0.1, 0.1, 0.1]),
        mass=MASS_KG,
    )
)
contact_view = RigidContactView(
    prim_paths_expr=cube.prim_path,
    filter_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"],
    disable_stablization=False,
    max_contact_count=100,
)

pending_impulses = []


def on_contact_report(headers, data):
    for header in headers:
        actor0 = str(physicsUtils.PhysicsSchemaTools.intToSdfPath(header.actor0))
        actor1 = str(physicsUtils.PhysicsSchemaTools.intToSdfPath(header.actor1))
        if "calibration_cube" not in actor0 and "calibration_cube" not in actor1:
            continue
        start = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        if count <= 0:
            continue
        impulse = np.zeros(3, dtype=float)
        for index in range(start, start + count):
            impulse += np.asarray(data[index].impulse, dtype=float)
        pending_impulses.append(impulse)


subscription = get_physx_simulation_interface().subscribe_contact_report_events(
    on_contact_report
)
world.reset()
contact_view.initialize()

samples = []
for step in range(90):
    world.step(render=False)
    default_value = np.asarray(contact_view.get_contact_force_matrix(dt=1.0))[0, 0]
    force_value = np.asarray(contact_view.get_contact_force_matrix(dt=DT))[0, 0]
    report_impulse = np.sum(pending_impulses, axis=0) if pending_impulses else np.zeros(3)
    pending_impulses.clear()
    if step >= 60:
        samples.append(
            (
                float(np.linalg.norm(default_value)),
                float(np.linalg.norm(force_value)),
                float(np.linalg.norm(report_impulse)),
            )
        )

gravity_direction, gravity_magnitude = world.get_physics_context().get_gravity()
values = np.asarray(samples, dtype=float)
print(
    {
        "physics_dt_s": world.get_physics_dt(),
        "gravity_direction": gravity_direction,
        "gravity_magnitude_mps2": gravity_magnitude,
        "mass_kg": float(cube.get_mass()),
        "theoretical_support_force_n": MASS_KG * float(gravity_magnitude),
        "tensor_default_median": float(np.median(values[:, 0])),
        "tensor_dt_median": float(np.median(values[:, 1])),
        "callback_impulse_median_ns": float(np.median(values[:, 2])),
        "callback_impulse_over_dt_median_n": float(np.median(values[:, 2]) / DT),
    },
    flush=True,
)

subscription = None
simulation_app.close()
