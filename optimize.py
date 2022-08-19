# === Generic continuous optimization approach applying the fcmaes library ====
# @author: Dietmar Wolz
# Idea is:
# - Use all vertices of the outer polygon and from all holes as demand points.
# - Add a grid of demand points filtered according to feasibility: Inside the outer polygon, outside the holes.
# - Uses matplotlib.path.contains_points to determine if a point is valid.
# - Uses https://numba.pydata.org/[numba] to speed up the fitness calculation .
# - Utilizes modern many-core CPUs, tested on the AMD 5950x 16 core CPU. 
# Compare with 'python vorheur.py -p 20 -o belle_outer -i belle_botany2,belle_dock,belle_pavillion1,belle_pond1,belle_pond3,belle_pond5,belle_botany,belle_playground,belle_pond2,belle_pond4,belle_tennis_court'
# 
# Using 
#   max_evaluations = 200000
#    opt = Bite_cpp(max_evaluations, popsize=500)
# 
# Computation needs time, but result is radius = 7.19 (https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/img/optimize_nd.pdf)
# compared to 8.667 for vorheur.py (https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/img/vorheur_sol.pdf).
#
# Using     
#   max_evaluations = 50000
#   opt = Bite_cpp(max_evaluations)
#
# computation takes less than a minute resulting in radius = 7.64

import numpy as np
import matplotlib.path as mpltPath
from numba import njit
from fcmaes.optimizer import Bite_cpp, crfmnes_bite, De_cpp, wrapper, logger
from fcmaes import retry
from scipy.optimize import Bounds 
from parse_kml import parse_kml, plot    

@njit(fastmath=True) # maximum of the minimal distances for all demand points
def fitness_(facilities_x, facilities_y, demands):
    max_r = 0
    for i in range(len(demands)):
        min_r = 1E99
        for j in range(len(facilities_x)):
            dx = demands[i,0] - facilities_x[j]
            dy = demands[i,1] - facilities_y[j]
            # we use the square of the distance because it is faster to compute
            r = dx*dx + dy*dy 
            if r < min_r: min_r = r 
        if min_r > max_r: max_r = min_r 
    return np.sqrt(max_r)

class Fitness():
    
    def __init__(self, p, corners, holes_corners, tolerance):
        self.p = p
        self.dim = self.p * 2
        cmax = np.amax(corners, axis=0)
        cmin = np.amin(corners, axis=0)
        lower = [cmin[0]]*p + [cmin[1]]*p
        upper = [cmax[0]]*p + [cmax[1]]*p
        self.generate_demands(tolerance, cmin, cmax, corners, holes_corners)
        #self.generate_minimal_demands(corners, holes_corners)
        self.bounds = Bounds(lower, upper) 
        
    def generate_demands(self, tolerance, cmin, cmax, corners, holes_corners):
        x = np.arange(cmin[0], cmax[0], tolerance)
        y = np.arange(cmin[1], cmax[1], tolerance)
        xs, ys = np.meshgrid(x, y)
        demands = np.vstack(list(zip(xs.ravel(), ys.ravel()))) # use grid demands    
        #demands = cmin + (cmax-cmin)*np.random.rand(num, 2) # use random demands
        path = mpltPath.Path(corners,closed=True)
        self.path = path
        self.pathes = []
        demands = demands[path.contains_points(demands)] # filter demands not in outer
        demands = np.concatenate((demands, corners))
        for hole_corners in holes_corners: # filter demands in holes
            path = mpltPath.Path(hole_corners, closed=True)
            demands = demands[np.logical_not(path.contains_points(demands))]
            demands = np.concatenate((demands, hole_corners))
            self.pathes.append(path)
        self.demands = demands
        print(len(demands))
        
    def get_facilities(self, x):
        facilities_x = x[:self.p]
        facilities_y = x[self.p:]
        return np.array([ [facilities_x[i], facilities_y[i]] \
                                    for i in range(self.p)])

    def fitness(self, x):
        facilities_x = x[:self.p]
        facilities_y = x[self.p:]
        facilities = [ [facilities_x[i], facilities_y[i]] for i in range(self.p)]
        penalty = 0
        for path in self.pathes: # penalty for facility in hole
            penalty += sum(path.contains_points(facilities))
        # penalty for facility outside outer
        penalty += sum(np.logical_not(self.path.contains_points(facilities)))
        if penalty > 0:
            return 1E10*penalty
        return fitness_(facilities_x, facilities_y, self.demands)
    
def optimize(fit, opt, num_retries = 32):
    ret = retry.minimize(wrapper(fit.fitness), 
                               fit.bounds, num_retries = num_retries, 
                               optimizer=opt, logger=logger())    
    print("facility locations = ", fit.get_facilities(ret.x))
    print("value = ", ret.fun)
    return fit.get_facilities(ret.x), ret.fun

def run_optimize(corners, holes_corners, tolerance = 0.5, ndepots=20):   
    fit = Fitness(ndepots, corners, holes_corners, tolerance)
    max_evaluations = 50000 # takes < 52 seconds on AMD 5950x
    opt = Bite_cpp(max_evaluations)
    # max_evaluations = 200000 # takes < 205 seconds on AMD 5950x
    # opt = Bite_cpp(max_evaluations, popsize=512)
   
    facilities, distance = optimize(fit, opt, num_retries = 32)
    for path in fit.pathes:
        print("in hole = ", path.contains_points(facilities))
    print("in outer = ", fit.path.contains_points(facilities))
    plot("optimize", facilities, distance, ndepots, fit.demands)
    plot("optimize_nd", facilities, distance, ndepots, None)
        
if __name__ == '__main__':
    outer_file = 'belle_outer.kml'
    hole_files =  ['belle_botany2', 'belle_dock', 'belle_pavillion1', 'belle_pond1', 'belle_pond3', 'belle_pond5',
                    'belle_botany', 'belle_playground', 'belle_pond2', 'belle_pond4', 'belle_tennis_court']
    corners, holes_corners = parse_kml(outer_file, hole_files)
    run_optimize(corners, holes_corners)
    
