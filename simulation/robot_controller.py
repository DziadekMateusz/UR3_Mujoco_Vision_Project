import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class RobotController:
    
    """Drives the UR3 arm to place object at its target site"""

    def __init__(self):

        root = Path(__file__).resolve().parent.parent
        scene_xml = root / "simulation" / "scene.xml"

        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

        self.tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site"
        )

        """Reusable Jacobian buffers - their shape never changes between
        calls, so allocate them once instead of every control-loop tick."""
        
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def move_tcp(
        self,
        target_pos,
        viewer,
        max_iters=3000,
        stall_window=50,
        stall_eps=1e-4,
        damping=0.05
    ):
        """Closed-loop Cartesian IK toward target_pos.
        Returns True if the target was reached, False if the move
        timed out (e.g. target outside the reachable workspace)"""

        n_arm = 6
        jnt_range = self.model.jnt_range[:n_arm]

        recent_dist = []

        for _ in range(max_iters):

            if not viewer.is_running():
                return False

            current = self.data.site_xpos[self.tcp_id].copy()
            error = target_pos - current
            distance = np.linalg.norm(error)

            print("DIST:", round(distance, 4))

            if distance < 0.03:
                print("TARGET REACHED")
                return True

            mujoco.mj_jacSite(
                self.model, self.data, self._jacp, self._jacr, self.tcp_id
            )

            J = self._jacp[:, :n_arm]

            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(
                JJt + (damping ** 2) * np.eye(3),
                error * 2.0
            )
            dq = np.clip(dq, -0.05, 0.05)

            target_q = self.data.qpos[:n_arm] + dq

            target_q = np.clip(target_q, jnt_range[:, 0], jnt_range[:, 1])

            self.data.ctrl[:n_arm] = target_q

            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)

            """Stall watchdog: if distance hasn't moved in a while, the
            arm is pinned against a limit/singularity. Nudge the
            joints with a small random perturbation and keep trying,
            instead of looping in place forever."""
            
            recent_dist.append(distance)
            if len(recent_dist) > stall_window:
                recent_dist.pop(0)
                if max(recent_dist) - min(recent_dist) < stall_eps:
                    print("STALLED - nudging joints to escape limit/singularity")
                    nudge = np.random.uniform(-0.05, 0.05, size=n_arm)
                    self.data.qpos[:n_arm] = np.clip(
                        self.data.qpos[:n_arm] + nudge,
                        jnt_range[:, 0], jnt_range[:, 1]
                    )
                    mujoco.mj_forward(self.model, self.data)
                    recent_dist.clear()

        print("MOVE TIMED OUT - target may be outside the reachable workspace")
        return False

    def pick_and_place(self, object_class):

        print(f"\nDetected object: {object_class}")

        target_name = f"{object_class}_target"
        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, target_name
        )

        target_pos = self.data.site_xpos[target_id].copy()
        print("TARGET POSITION:", target_pos)

        approach = target_pos.copy()
        approach[2] += 0.15
        print("APPROACH POSITION:", approach)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            viewer.cam.azimuth = 180
            viewer.cam.elevation = -25
            viewer.cam.distance = 2.0

            print("\nMOVING ABOVE TARGET...")
            if not self.move_tcp(approach, viewer):
                print(f"\nFAILED to reach approach position for {object_class}")
                return

            print("\nMOVING TO TARGET...")
            if not self.move_tcp(target_pos, viewer):
                print(f"\nFAILED to reach target position for {object_class}")
                return

        print(f"\n{object_class} placed")
