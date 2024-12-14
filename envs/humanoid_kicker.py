# import numpy as np
# from agents.agent import Agent
# from agents.humanoid import Humanoid
# import six
# from gym import spaces
import os

from brax.io import mjcf
from brax import base
from brax import actuator
from brax.envs.base import PipelineEnv, State

import jax
from jax import numpy as jp

def stabilization_reward(pipeline_state: base.State, action: jp.ndarray, target_height=1.0, dt=0.05) -> float:
    torso_height = pipeline_state.x.pos[0, 2]
    height_error = abs(target_height - torso_height)
    height_reward = -height_error / dt

    control_cost = 0.01 * jp.sum(jp.square(action))

    reward = height_reward + 1 - control_cost

    fallen_threshold = 0.5
    done = torso_height < fallen_threshold
    
    return reward, done

def calculate_kick_reward(pipeline_state: base.State, action: jp.ndarray):

    humanoid_pos = pipeline_state.x.pos[0]  # Humanoid
    ball_pos = pipeline_state.x.pos[1]  # Ball
    goal_pos = pipeline_state.x.pos[2]  # Goal

    ball_to_goal_dist = jp.norm(ball_pos - goal_pos)  # Distance from ball to goal
    humanoid_to_ball_dist = jp.norm(humanoid_pos - ball_pos)  # Distance from humanoid to ball
    ball_speed = jp.norm(pipeline_state.xd.vel[1])  # Ball velocity

    reward = -humanoid_to_ball_dist  # Reward approaching the ball
    reward += jp.exp(-ball_to_goal_dist)  # Bonus for moving ball closer to goal
    reward += 10.0 if ball_to_goal_dist < 0.1 else 0.0  # Big reward for scoring

    ctrl_cost = 0.01 * jp.sum(jp.square(action))
    reward -= ctrl_cost

    done = jp.array(1.0) if ball_to_goal_dist < 0.1 else jp.array(0.0)

    return reward, done


class HumanoidKicker(PipelineEnv):
    def __init__(self, **kwargs):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "kicking", "combined.xml" )
        sys = mjcf.load(str(path))

        super().__init__(sys, backend='mjx', **kwargs)
    
    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'stabilizeReward': zero,
            'kickReward': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    
    def step(self, state: State, action: jp.ndarray) -> State:
        # Scale action to actuator range
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Simulate physics
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Extract positions

        # Calculate reward
        kickReward, doneKick = calculate_kick_reward(pipeline_state, action)
        standReward, doneStand = stabilization_reward(pipeline_state, action)
        
        done = doneKick + doneStand

        #Control cost already calculated in both functions^
        reward = kickReward + standReward #need to linearly decrease one and increase the other.

        # Compute new observations
        obs = self._get_obs(pipeline_state, action)

        # Update metrics
        metrics = {
            'stabilizeReward': standReward,
            'kickReward': kickReward,
        }
        state.metrics.update(metrics)

        # Return updated state
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State):
        """Get complete state observation for humanoid kicker environment."""
        # Get center of mass data using class method
        com_pos, com_vel = self._com(pipeline_state)
        
        # Core body state
        qpos = pipeline_state.qp.pos
        qvel = pipeline_state.qp.vel
        cinr = pipeline_state.qp.cinr
        
        # Torso state (root body)
        torso_pos = qpos[0, :3]
        torso_rot = qpos[0, 3:7]
        torso_vel = qvel[0, :3]
        ang_vel = qvel[0, 3:6]
        
        # Joint and actuator states
        joint_pos = qpos[1:]
        joint_vel = qvel[1:]
        qfrc_actuator = pipeline_state.qf.qfrc_actuator
        
        # Relative body positions to COM
        xd_i = qpos[:, :3] - com_pos[None]
        
        # Ball state (second to last body)
        ball_pos = pipeline_state.qp.pos[-2, :3]
        ball_vel = pipeline_state.qp.vel[-2, :3]
        
        # Target state (last body)
        target_pos = pipeline_state.qp.pos[-1, :3]
        
        obs = jp.concatenate([
            com_pos,           # 3 - COM position from _com()
            com_vel,          # 3 - COM velocity from _com()
            cinr.flatten(),   # 9 - Centroidal inertia
            torso_pos,        # 3 - Torso position
            torso_rot,        # 4 - Torso rotation
            torso_vel,        # 3 - Torso linear velocity
            ang_vel,          # 3 - Torso angular velocity
            joint_pos,        # n - Joint positions
            joint_vel,        # n - Joint velocities
            qfrc_actuator,    # n - Actuator forces
            xd_i.flatten(),   # 3*n_bodies - Body positions relative to COM
            ball_pos,         # 3 - Ball position
            ball_vel,         # 3 - Ball velocity
            target_pos,       # 3 - Target position
        ])
        
        return obs
    
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
    
