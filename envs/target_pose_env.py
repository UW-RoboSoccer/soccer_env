import jax
import jax.numpy as jp

from brax import actuator
from brax import base
from brax import envs
from brax import math
from brax.envs.base import PipelineEnv, State

import xml.etree.ElementTree as ET

from brax.io import mjcf

class OP3Stand(PipelineEnv):
    def __init__(self, xml_path='../assets/op3/op3_simplified.xml', **kwargs):
        self.weights = {
            'reward_base_orient': 1.0,
            'reward_height': 1.0,
            'reward_ctrl': 0.1,
            'reward_up_vel': 1.0,
        }
        # define any other necessary values here

        if xml_path is None:
            raise ValueError('xml_path is required')
        
        self.mocap_rng = kwargs.pop('rng_key', jax.random.PRNGKey(0))

        # load brax system from xml file
        sys = mjcf.load(xml_path)
        super().__init__(sys, backend='mjx', **kwargs)

        # Load target q poses from poses.xml
        self.target_q_poses = self._load_target_q_poses('../poses.xml')
        self.steps_till_switch = 0

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng_pos, rng_vel, self.mocap_rng = jax.random.split(rng, 4)
        
        qp = jax.random.uniform(rng_pos, (self.sys.q_size(),), jp.float32, 0.4, 1)
        qv = jax.random.uniform(rng_vel, (self.sys.qd_size(),), jp.float32, -0.01, 0.01)

        act = jp.zeros(self.sys.act_size(), jp.float32)
        pipeline_state = self.pipeline_init(qp, qv, act)
        obs = self._get_obs(pipeline_state, act)

        self.step_count = 0
        self._lambda = 100

        reward, done, zero = jp.zeros(3)

        metrics = {
            'reward_up_vel': zero,
            'reward_base_orient': zero,
            'reward_height': zero,
            'reward_ctrl': zero,
        }

        info = {
            'base_height': pipeline_state.q[2],
            'base_orientation': jp.array([0, 0, 1], jp.float32),
            # 'time_step': jp.array(0, dtype=jp.int32),
            'target_time': jp.random.exponential(self._rng, shape=(1,)),
            'target_pos_idx': jp.random.randint(self.mocap_rng, minval=0, maxval=len(self.target_q_poses), shape=(1,), dtype=jp.int32)
        }

        return State(pipeline_state, obs, reward, done, metrics, info)
    
    def step(self, state: State, action: jax.Array):
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state = self.pipeline_step(state.pipeline_state, action)



        # Track the number of steps until the next target pose switch
        # time_idx = pipeline_state.info['time_step']
        target_time = pipeline_state.info['target_time']
        target_time -= 1

        # Track pose 

        target_time = jp.where(target_time <= 0, jp.random.exponential(self._rng, shape=(1,))/self._lambda, target_time)
        target_pos_idx = jp.where(target_time <= 0, jp.random.randint(self.mocap_rng, minval=0, maxval=len(self.target_q_poses), shape=(1,), dtype=jp.int32), state.info['target_pos_idx'])

        target_q_pos = self.target_q_poses[target_pos_idx]

        # Calculate rewards
        joint_pos_reward = self.joint_pos_reward(pipeline_state.q, target_q_pos)
        orientation_reward = self.orientation_reward(pipeline_state.q, target_q_pos)


        # time_idx = jp.where(self.steps_till_switch == 0, jp.random.exponential(self._rng, shape=(1,), rate=1), 0)
        
        # height = pipeline_state.q[2]
        # reward_height = self.weights['reward_height'] * (jp.exp(1.5*(height-0.6)) - 1)

        # up_vel = pipeline_state.qd[2]
        # reward_up_vel = self.weights['reward_up_vel'] * up_vel

        # quad_ctrl_cost = self.weights['reward_ctrl'] * jp.sum(jp.square(action))
        
        # up = jp.array([0, 0, 1], jp.float32)
        # rot_up = math.rotate(up, pipeline_state.q[3:7])
        # reward_base_orient = self.weights['reward_base_orient'] * jp.dot(up, rot_up)

        # reward = reward_height + reward_base_orient + reward_up_vel

        obs = self._get_obs(pipeline_state, action)

        state.metrics.update(reward_base_orient=reward_base_orient, reward_height=reward_height, reward_ctrl=-quad_ctrl_cost, reward_up_vel=reward_up_vel)
        state.info.update(base_height=height, base_orientation=rot_up)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    
    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q[2:]
        velocity = pipeline_state.qd

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
        ])

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
    
    def _load_target_q_poses(self, poses_path):
        tree = ET.parse(poses_path)
        root = tree.getroot()
        target_q_poses = jp.array()

        for key in root.findall('key'):
            qpos = key.get('qpos')
            q_values = qpos.split()
            q_values = [float(q) for q in q_values]
            target_q_poses.append(q_values)

        return target_q_poses
    
    def joint_pos_reward(self, joint_pos: jax.Array, target_q_pos: jax.Array):
        joint_pos = joint_pos[7:]
        target_q_pos = target_q_pos[7:]

        error = (jp.pi - jp.norm(target_q_pos - joint_pos)) / jp.pi
        return error
    
    def orientation_reward(self, joint_pos: jax.Array, target_q_pos: jax.Array):
        joint_pos = joint_pos[3:7]
        target_q_pos = target_q_pos[3:7]

        error = (jp.pi - jp.arccos(jp.dot(joint_pos, target_q_pos))) / jp.pi
        return error