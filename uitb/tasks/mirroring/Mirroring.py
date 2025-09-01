import numpy as np
import math
import mujoco
from scipy.signal import sawtooth

from ...simulator import Simulator
from ..base import BaseTask
from .reward_functions import AbsDistance


class Mirroring(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector to be defined
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list with two elements: first defines the mujoco element type, and "
                         "second defines the name.")
    self._end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    if not isinstance(shoulder, list) and len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements: first defines the mujoco element type, and "
                         "second defines the name.")
    self._shoulder = shoulder

    # Load reward function from config
    if "distance" not in kwargs:
      self._reward_function = AbsDistance()
    else:
      distance_cls = Simulator.get_class("tasks.mirroring.reward_functions", kwargs['distance'].get('cls', 'AbsDistance'))
      self._reward_function = distance_cls(**(kwargs['distance'].get('kwargs', {})))

    # Define episode length
    self._episode_length_seconds = kwargs.get('episode_length_seconds', 10)
    self._max_episode_steps = self._action_sample_freq * self._episode_length_seconds

    # Print eval settings
    if kwargs.get('evaluate', False):
      print(self._reward_function)

    # Define some limits for target movement speed
    max_arm_length = 0.7
    offset_body_x = min(max_arm_length, max(0.0, kwargs.get('offset_body_x', 0.25)))

    self._frequency = np.array([0.1, 0.5])
    self._disable_x_offset = kwargs.get('disable_x', False)
    self._offset_limit_x = max_arm_length - offset_body_x
    self._offset_limit_y = kwargs.get('limit_y', None)
    self._offset_limit_z = kwargs.get('limit_z', None)

    # For logging
    self._info = {"terminated": False, "truncated": False, "termination": False, "inside_target": False, "dist_target": None}

    # Target radius
    self._target_radius = kwargs.get('target_radius', 0.035)
    model.geom("target").size[0] = self._target_radius
    model.geom("mirror").size[0] = self._target_radius

    # hide target during training
    model.geom("target").type = mujoco.mjtGeom.mjGEOM_NONE

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define target origin and furthest position: 0.20m-0.70m in front of the right of shoulder.
    # Note that the body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([offset_body_x, 0, 0])
    self._target_position = self._target_origin.copy()

    # 0.05m gap in center + limit_x offset => outer edge start
    self._mirror_origin = self._target_origin + np.array([0.05 + self._offset_limit_x * 2, 0, 0])
    self._mirror_position = self._mirror_origin.copy()

    # Generate trajectory
    self._generate_trajectory()

    model.cam("for_testing").pos = np.array([-0.8, -0.6, 1.5])
    model.cam("for_testing").quat = np.array([0.718027, 0.4371043, -0.31987, -0.4371043])

  def _update(self, model, data):
    # Set defaults
    terminated = False
    truncated = False
    self._info = {"termination": False}

    # Get end-effector position
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos.copy()

    # Distance to target origin
    dist = np.linalg.norm(self._target_position - ee_position)

    # Is fingertip inside target?
    if dist <= self._target_radius:
      self._info["inside_target"] = True
      self._info["dist_target"] = 0
    else:
      self._info["inside_target"] = False
      self._info["dist_target"] = dist - self._target_radius

    # Check if time limit has been reached
    if self._steps >= self._max_episode_steps:
      truncated = True
      self._info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self._reward_function.get(self, dist - self._target_radius, self._info.copy())

    # Update target location
    self._update_target_location(model, data)

    return reward, terminated, truncated, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    state.update(self._info)
    return state

  def _reset(self, model, data):
    self._info = {"terminated": False, "truncated": False, "termination": False, "inside_target": False, "dist_target": None}

    # Generate a new trajectory
    self._generate_trajectory()

    # Update target location
    self._update_target_location(model, data)

    return self._info

  def _generate_frequency_curve(self):
    t = np.linspace(0, self._episode_length_seconds, int(self._max_episode_steps+1), endpoint=False)

    # Smoothly interpolate between random values to avoid abrupt changes
    key_points = self._rng.uniform(self._frequency[0], self._frequency[1], size=5)
    key_times = np.linspace(0, self._episode_length_seconds, num=len(key_points))
    frequency_curve = np.interp(t, key_times, key_points)

    return t, frequency_curve

  def _generate_offset_x(self):
    t, frequency_curve = self._generate_frequency_curve()

    # Integrate frequency to get phase (φ(t) = ∫ f(t) dt), scaled to 2π
    phase = 2 * np.pi * np.cumsum(frequency_curve) * (t[1] - t[0])
    
    # Triangle wave using varying phase
    triangle_wave = sawtooth(phase, width=0.5)

    # Normalize from [-1, 1] to [0, 1]
    oscillation = 0.5 * (triangle_wave + 1)

    return oscillation * self._offset_limit_x

  def _generate_sine_wave(self, limits, num_components=5, min_amplitude=1, max_amplitude=5):
    # Generate a sine wave with multiple components
    t = np.arange(self._max_episode_steps+1) * self._dt
    sine = np.zeros((t.size,))
    sum_amplitude = 0
    for _ in range(num_components):
      amplitude = self._rng.uniform(min_amplitude, max_amplitude)
      sine +=  amplitude *\
              np.sin(self._rng.uniform(0.0, 0.5)*2*np.pi*t + self._rng.uniform(0, 2*np.pi))
      sum_amplitude += amplitude

    # Normalise to fit limits
    sine = (sine + sum_amplitude) / (2*sum_amplitude)
    sine = limits[0] + (limits[1] - limits[0])*sine

    return sine

  def _generate_trajectory(self):
    if not self._disable_x_offset:
      self._offset_x = self._generate_offset_x()

    if self._offset_limit_y is not None:
      self._offset_y = self._generate_sine_wave(np.array([-self._offset_limit_y, self._offset_limit_y]), num_components=5)

    if self._offset_limit_z is not None:
      self._offset_z = self._generate_sine_wave(np.array([-self._offset_limit_z, self._offset_limit_z]), num_components=5)

  def _update_target_location(self, model, data):
    offset_x = self._offset_x[self._steps] if getattr(self, '_offset_x', None) is not None else 0
    offset_y = self._offset_y[self._steps] if getattr(self, '_offset_y', None) is not None else 0
    offset_z = self._offset_z[self._steps] if getattr(self, '_offset_z', None) is not None else 0

    self._target_position = self._target_origin + np.array([ offset_x, offset_y, offset_z])
    model.body("target").pos = self._target_position

    self._mirror_position = self._mirror_origin + np.array([-offset_x, offset_y, offset_z])
    model.body("mirror").pos = self._mirror_position

    mujoco.mj_forward(model, data)
