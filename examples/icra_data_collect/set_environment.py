"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict
import numpy as np

import elastica as el
from coomm.callback_func import SphereCallBack
from elastica._calculus import _isnan_check

# import sys
# sys.path.append("../")          # include examples directory
from set_arm_environment import ArmEnvironment

def check_target_distance(pos, target, threshold=2e-3):
    target_vector = target[:,0][:,None] - pos
    dist = np.linalg.norm(target_vector, axis=0)
    if np.amin(dist) < threshold:
        return True
    return False

class Environment(ArmEnvironment):
    
    def get_data(self):
        return [self.rod_parameters_dict, self.sphere_parameters_dict]

    def setup(self):
        self.set_arm()
        self.set_target()

    def set_target(self):
        """ Set up a sphere object """
        target_radius = 0.006
        self.sphere = el.Sphere(
            center=np.array([0.01, 0.15, 0.06]),
            base_radius=target_radius,
            density=1000
        )
        self.sphere.director_collection[:, :, 0] = np.array(
            [[ 0, 0, 1],
             [ 0, 1, 0],
             [-1, 0, 0]]
        )
        self.simulator.append(self.sphere)
        self.sphere_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.sphere).using(
            SphereCallBack,
            step_skip=self.step_skip,
            callback_params=self.sphere_parameters_dict
        )
        
        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            el.OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,)
        )

    def step(self, time, muscle_activations):

        """ Set muscle activations """
        for muscle_group, activation in zip(self.muscle_groups, muscle_activations):
            muscle_group.apply_activation(activation)
        
        """ Run the simulation for one step """
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)
        target_dist_condition = check_target_distance(self.shearable_rod.position_collection, self.sphere.position_collection, self.sphere.radius[0])

        if invalid_values_condition == True:
            print("NaN detected in the simulation !!!!!!!!")
            done = True

        if target_dist_condition == True:
            print('Target has been reached!')
            done = True

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done