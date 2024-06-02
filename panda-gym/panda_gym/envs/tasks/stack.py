from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

class Stack(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        get_ee_velocity,
        get_fingers_width,
        reward_type="dense", # SAC + dense, DDPG + sparse
        distance_threshold=0.04, # 0.1
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_ee_velocity = get_ee_velocity
        self.get_fingers_width = get_fingers_width
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=5, width=5, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position, object2_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        goal2 += noise
        return np.concatenate((goal1, goal2))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, 3 * self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        
        reward = 0.0
        
        EE_position = self.get_ee_position() - np.array([0.0, 0.0, 0.05])
        EE_Width = self.get_fingers_width()
        EE_speed = self.get_ee_velocity()
        
        blue_speed = self.get_obs()[2]
        blue_position = achieved_goal[:3] 
        blue_goal = desired_goal[:3] 
        
        green_speed = self.get_obs()[6]
        green_position = achieved_goal[3:] 
        green_goal = desired_goal[3:]
        
        if self.is_success(blue_goal, blue_position):
            reward += self.compute_reward_approach_xy(EE_position, green_position)
            reward += self.compute_reward_approach_xy(green_goal, green_position)
            reward += self.compute_reward_speed(EE_speed, green_speed)
            reward += self.compute_reward_approach_z(EE_position)
            reward += self.compute_reward_grasp(EE_Width)
            reward += 3
            reward += int(blue_position[2] < green_position[2]) * 3

        else:
            reward += self.compute_reward_approach_xy(EE_position, blue_position)
            reward += self.compute_reward_approach_xy(blue_goal, blue_position)
            reward += self.compute_reward_approach_z(EE_position)
            reward += self.compute_reward_speed(EE_speed, blue_speed)
            reward += self.compute_reward_grasp(EE_Width)
            
        return np.array([reward], dtype=np.float32)
    
    def compute_reward_approach_xy(self, EE_position, goal) -> np.ndarray:
        d = distance(EE_position[:2], goal[:2])
        if d > self.distance_threshold:
            reward = int(-25 * d.astype(np.float32))
        else:
            reward = 0
        return np.array([reward], dtype=np.float32)
    
    def compute_reward_approach_z(self, EE_position) -> np.ndarray:
        if EE_position[2] > 0.1 or EE_position[2] < 0.01:
            reward = -7
        else:
            reward = 0
        return np.array([reward], dtype=np.float32)
    
    def compute_reward_grasp(self, Width) -> np.ndarray:
        if Width > 0.045 or Width < 0.04:
            reward = -3
        else:
            reward = 0
        return np.array([reward], dtype=np.float32)
    
    def compute_reward_speed(self, Speed1, Speed2) -> np.ndarray:
        rel_speed = np.linalg.norm(Speed1 - Speed2)
        if rel_speed > 0.1:
            reward = int(-20 * rel_speed)
        else:
            reward = 0
        return np.array([reward], dtype=np.float32)

        
        
        
        