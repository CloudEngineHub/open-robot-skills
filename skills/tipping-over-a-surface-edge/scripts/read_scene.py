"""Which hand is on which side, and where the crate is, read live."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap import NodeContext
import tipcommon as T


def run(ctx: NodeContext) -> dict:
    arms = []
    for arm_id in (0, 1):
        info = ctx.tool("robot.describe_arm", arm_id=arm_id)
        ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]["position"]
        arms.append({"arm_id": arm_id, "name": info["arm"], "sign": 1.0 if ee["y"] >= 0 else -1.0, "home_ee": [ee["x"], ee["y"], ee["z"]]})
    if len({a["sign"] for a in arms}) != 2:
        raise RuntimeError(f"both hands on one side of the bench: {arms}")
    grip = ctx.tool("robot.describe_gripper", arm_id=0)
    st = T.crate_state(ctx)
    scene = {"x0": st["c"][0], "y0": st["c"][1], "z0": st["c"][2], "z_table": st["c"][2] - T.HZ, "theta0": st["theta"]}
    T.log("read_scene", arms=arms, scene=scene, gripper={k: grip[k] for k in ("span_m", "min_grasp_width_m", "finger", "housing", "width_fit")})
    return {"arms": arms, "scene": scene, "route": "ok" if abs(st["theta"]) < 5.0 else "bad_start"}
