# Isaac Sim 5.0 Compatibility

## Scope

This checkout of Embodied-AI-Safety has been adapted to run the
`hand_avoidance` Split ALOHA workload with Isaac Sim 5.0. The validated setup
is:

- Isaac Sim: `5.0.0-rc.45+release.23960.184afb15.gl`
- Compatibility userspace: Ubuntu 22.04.5 LTS
- Isaac Sim installation: `/home/zxw/isaacsim-5.0`
- Ubuntu 22.04 rootfs: `/home/zxw/isaacsim-5.0-rootfs`
- Shared InternData-A1 assets: `/home/zxw/InternDataEngine/InternDataAssets`

The compatibility userspace is launched with bubblewrap. The host NVIDIA
device nodes and driver libraries are passed through, so the workload still
uses the host GPU.

## Run

From this repository:

```bash
cd /home/zxw/InternDataEngine/3rd/Embodied-AI-Safety
scripts/isaac50/run_hand_avoidance.sh
```

Set a deterministic seed with:

```bash
RANDOM_SEED=1 scripts/isaac50/run_hand_avoidance.sh
```

The launcher uses `configs/simbox/de_hand_avoidance_isaac50.yaml` and sets
`INTERNDATA_ISAAC5_COMPAT=1`. It does not replace the existing Isaac Sim 4.5
installation or launcher.

## Verified Result

The seed-0 end-to-end run completed with `success rate: 1/1` on 2026-08-05.
Both arms picked their assigned objects, lifted them by more than 2 cm, moved
them to the target bin, released them, and passed detached placement checks.

Run log:

```text
output/hand_avoidance_isaac50/de_time_profile_20260805_032338.log
```

Complete episode:

```text
output/hand_avoidance_isaac50/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance/2026-08-05_03_26_41_911901
```

The episode contains:

- 1,234 time steps and 6,207 LMDB entries
- a 244,305,920-byte LMDB data file
- `sim_raw_gt.json`, `sim_features.json`, and `sim_labels.json`
- an automatic JSON safety report
- left-hand, right-hand, and head-camera MP4 files
- 1,234 decoded frames per video at 640x480 and 15 FPS

The generated risk result is HS L3, PT L3, RS L2, IR L0, overall L3, with
data quality A and no missing fields. PT drop/damage labels in this report are
rule-based proxy labels, not readings from a physical fracture sensor.

## Compatibility Changes

1. Added an Ubuntu 22.04/Isaac Sim 5 launcher under `scripts/isaac50/`, backed
   by the engine-level bubblewrap launcher.
2. Added an Isaac Sim 5 pipeline config with a writable portable Kit root and
   metrics-assembler listeners disabled during tensor-view execution.
3. Replaced unavailable absolute asset paths with the local InternData-A1
   asset link and a portable MANO USD wrapper.
4. Pointed Split ALOHA CuRobo configuration at the local robot YAML files and
   explicitly initialized the mobile-body, arm, and gripper joints.
5. Reordered scene initialization so robot and object prims are stable before
   PhysX tensor views are created. Added controlled warmup and world/local
   frame corrections for sampled task regions.
6. Updated CuRobo planning for live PhysX/Fabric transforms, deterministic
   chained preplanning, stale-plan rejection, and sequential dual-arm skills.
7. Replaced imageio's unavailable optional FFmpeg writer with OpenCV MP4
   output and verified all three generated videos can be decoded.

## Boundaries

- Grasp and placement semantics intentionally remain those of the upstream
  repository: Pick uses the original CuRobo attachment path and its original
  contact-based success criterion.
- Deprecation notices for legacy `omni.isaac` aliases and asset warnings such
  as dynamic triangle-mesh fallback remain, but they did not stop this run.
- Only the Split ALOHA `hand_avoidance` workload has completed end-to-end
  validation in this imported repository. Other task configurations still
  require task-specific validation before they should be called compatible.

## Post-audit Verification

After removing non-upstream grasp tracking, release delays, unused planner
options, and diagnostic-only branches, a fresh Isaac Sim 5 run succeeded on
2026-08-10 with random seed `1868503102`. The generated episode is:

```text
output/hand_avoidance_isaac50/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance/2026-08-10_13_10_25_435451
```

It contains 340 simulation steps, a 74,256,384-byte LMDB, the raw GT/features/
labels JSON files, an automatic risk report, and three decodable 640x480 MP4
videos with 340 frames each.
