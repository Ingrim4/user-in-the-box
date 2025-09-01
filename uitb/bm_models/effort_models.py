from abc import ABC, abstractmethod
import mujoco
import numpy as np

import sys

class BaseEffortModel(ABC):

  def __init__(self, bm_model, **kwargs):
    self._bm_model = bm_model

  @abstractmethod
  def cost(self, model, data):
    pass

  @abstractmethod
  def reset(self, model, data):
    pass

  @abstractmethod
  def update(self, model, data):
    # If needed to e.g. reduce max force output
    pass

  def _get_state(self, model, data):
    """ Return the state of the effort model. These states are used only for logging/evaluation, not for RL
    training

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()


class Zero(BaseEffortModel):

  def cost(self, model, data):
    return 0

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class Neural(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None

  def cost(self, model, data):
    self._effort_cost = self._weight * np.sum(data.ctrl ** 2)
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state


class EJK(BaseEffortModel):

  def __init__(self, bm_model, weight=0.8, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None
    self._previous_qacc = np.zeros(17)

  def cost(self, model, data):
    r_effort = np.mean(data.ctrl)

    r_jerk = (np.linalg.norm(data.qacc - self._previous_qacc) / model.opt.timestep)/100000
    self._previous_qacc = data.qacc    
      
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_rot")
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
      
    shoulder_ang_vel = data.qvel[shoulder_id]
    elbow_ang_vel = data.qvel[elbow_id]
    
    # Access torques applied to joints
    shoulder_torque = data.ctrl[shoulder_id] 
    elbow_torque = data.ctrl[elbow_id]

    r_work = (np.abs(shoulder_ang_vel*shoulder_torque) + np.abs(elbow_ang_vel*elbow_torque))/100

    self._effort_cost = self._weight*(r_effort + 8* r_jerk + r_work)/10
      
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state


class JAC(BaseEffortModel):

  def __init__(self, bm_model, weight=1, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None

  def cost(self, model, data): 
    r_effort = 0.0198*np.linalg.norm(data.ctrl)**2 
    r_jacc = 6.67e-5*np.linalg.norm(data.qacc[self._bm_model.independent_dofs])**2 

    self._effort_cost = self._weight*(r_effort + r_jacc)
      
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state


class DC(BaseEffortModel):

  def __init__(self, bm_model, weight=1, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None

  def cost(self, model, data):
    self._effort_cost = self._weight*0.1477*np.linalg.norm(data.ctrl)**2 
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state


class CTC(BaseEffortModel):

  def __init__(self, bm_model, weight=1, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None
    self._previous_torque_shoulder = 0
    self._previous_torque_elbow = 0
    self.lifting_muscles = ["DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed"]  
    self.elbow_muscles = ["ANC", "SUP", "BRA", "PT", "PQ"]
      
  def cost(self, model, data):
    r_effort = 0.649*np.linalg.norm(data.ctrl)**2 

    lifting_indices = [model.actuator(_i).id for _i in self.lifting_muscles]
    applied_shoulder_torque = np.linalg.norm(data.actuator_force[lifting_indices])

    elbow_indices = [model.actuator(_i).id for _i in self.elbow_muscles]
    applied_elbow_torque = np.linalg.norm(data.actuator_force[elbow_indices])
      
    r_ctc = 0.0177 * (np.abs(applied_shoulder_torque - self._previous_torque_shoulder) + np.abs(applied_elbow_torque - self._previous_torque_elbow))
      
    self._previous_torque_shoulder = applied_shoulder_torque
    self._previous_torque_elbow = applied_elbow_torque

    self._effort_cost = self._weight*(r_effort + r_ctc)
      
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state
