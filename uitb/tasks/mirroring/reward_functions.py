from abc import ABC, abstractmethod
import numpy as np

class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass

class HitBonus(BaseFunction):

  def __init__(self, bonus=1.0):
    self._bonus = bonus

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self._bonus
    else:
      return 0.0

  def __repr__(self):
    return "HitBonus"

class HitBonusNegative(BaseFunction):

  def __init__(self, bonus=1.0):
    self._bonus = bonus

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self._bonus
    else:
      return -self._bonus

  def __repr__(self):
    return "HitBonusNegative"

class ExpDistance(BaseFunction):

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0.1
    else:
      return np.exp(-dist * 10) / 10

  def __repr__(self):
    return "ExpDistance"


class ExpDistanceWithHitBonus(BaseFunction):

  def __init__(self, bonus=0.2):
    self._bonus = bonus

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self._bonus
    else:
      return np.exp(-dist * 10) / 10

  def __repr__(self):
    return "ExpDistanceWithHitBonus"

class AbsDistance(BaseFunction):

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0.0
    else:
      return -dist

  def __repr__(self):
    return "AbsDistance"

class AbsDistanceWithHitBonus(BaseFunction):

  def __init__(self, bonus=0.5):
    self._bonus = bonus

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self._bonus
    else:
      return -dist

  def __repr__(self):
    return "AbsDistanceWithHitBonus"
