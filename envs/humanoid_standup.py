import brax
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax import base
from brax.base import Transform
from brax import actuator

import brax.math
import jax
from jax import numpy as jp

import math
import pathlib

class HumanoidStandup(PipelineEnv):
    """
    ### Action Space

    ### Observation Space

    ### Rewards

    Reward consists of two parts:

    - *time-variant reward*: only activated for the last x seconds of the episode
        - *base height*: the height of the base of the humanoid
        - *joint position*: how closely the joints are to their default stance positions
        - *base orientation*: how closely the base is to being upright
    - *time-invariant reward*: activated for the entire episode
        - *body collision*: penalize high body collision
        - *momentum change*: penalize high momentum change
        - *body yank*: penalize high body yank
        - *joint velocity*: penalize high joint velocities
        - *action range*: penalize high actions rate
        - *torques*: penalize high torques
        - *accelerations*: penalize high accelerations

    ### Reset

    ### Episode Termination
    """

    def __init__(self, path, **kwargs):
        self.stance_height = 1.2    # height of torso when standing
        self.h_sensitivity = kwargs.get('h_sensitivity', 0.01)  # height sensitivity scaling factor
        self.q_sensitivity = kwargs.get('q_sensitivity', 0.01)  # joint position sensitivity scaling factor
        self.reward_weight = kwargs.get('reward_weight', [1, 1, 1, 1, 1, 1])  # reward weight for each reward component

        sys = mjcf.load(str(path))
        super().__init__(sys, backend='mjx', **kwargs)

        # array of key link indices
        self.B = ['torso', 'head', 'pelvis', 'left_foot', 'right_foot']
        self.B_links = [self.sys.link[i] for i in range(self.sys.num_links) if self.sys.link_names[i] in self.B]

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Randomize Joint Positions
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-1, maxval=1
        )

        # Randomize Joint Velocities
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-1, maxval=1
        )

        # Reset actuator activation states
        act = jp.zeros(self.sys.act_size())

        pipeline_state = self.pipeline_init(qpos, qvel, act)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_linup': zero,
            'reward_quadctrl': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jp.ndarray) -> State:
        # Scale action from [-1, 1] to actuators control range
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]

        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # update pipeline state
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Calculate reward
        reward = self.calculate_reward(pipeline_state, pipeline_state0)
        reward = jp.dot(reward, self.reward_weight)

        # Update observation
        obs = self._get_obs(pipeline_state, action)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    
    def _get_obs(self, pipeline_state: base.State, action: jp.ndarray) -> jp.ndarray:
        # Get joint states
        q = pipeline_state.q
        qd = pipeline_state.qd

        # Get base states
        x = pipeline_state.x
        xd_ang = pipeline_state.xd.ang

        return jp.concatenate([q, qd, x.pos, x.rot, xd_ang, action])

    def _base_height_reward(self, x: Transform) -> jp.ndarray:
        return math.exp(-math.pow(max(self.stance_height - x.pos[0, 2], 0), 2) / self.h_sensitivity)

    def _joint_pos_reward(self, q: jax.Array) -> jp.ndarray:
        return math.exp(-jp.sum(jp.square(self.sys.init_q - q)) / (self.q_sensitivity * self.sys.q_size()))
    
    def _base_orient_reward(self, x: Transform) -> jp.ndarray:
        up = jp.array([0, 0, 1])
        rot_up = brax.math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _momentum_change_reward(self, qdd: jax.Array) -> jp.ndarray:
        return jp.sum(jp.square(qdd * self.sys.link.inertia.mass[:, None]))
    
    def _joint_vel_reward(self, qd: jax.Array) -> jp.ndarray:
        return jp.sum(jp.square(qd))
    
    def _joint_acc_reward(self, qdd: jax.Array) -> jp.ndarray:
        return jp.sum(jp.square(qdd))

    def calculate_reward(self, pipeline_state: base.State, pipeline_state0: base.State) -> jp.ndarray:
        qdd = (pipeline_state.qd - pipeline_state0.qd) / self.dt

        base_height = self._base_height_reward(pipeline_state.x)
        joint_pos = self._joint_pos_reward(pipeline_state.q)
        base_orient = self._base_orient_reward(pipeline_state.x)
        momentum_change = self._momentum_change_reward(qdd)
        joint_vel = self._joint_vel_reward(pipeline_state.qd)
        joint_acc = self._joint_acc_reward(qdd)

        return jp.array([base_height, joint_pos, base_orient, momentum_change, joint_vel, joint_acc])
