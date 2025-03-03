from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pymulti as pm
from functools import partial


n_splits = 3

sampler = pm.Sampler()

laser_time = [0.000000,
	    0.380000,
	    0.510000,
	    0.240000,
	    0.250000,
	    0.320000,
	    0.410000,
	    0.260000,
	    0.360000,
	    0.62,2.21,0.1300]
laser_power=[0.000000,
	    4.490000,
	    0.000000,
	    0.00,
	    3.480000,
	    4.060000,
	    5.010000,
	    5.880000,
	    13.47000,
	    28.0,
	    0.000000]
laser_power1=[i*(1-0.2*i/28) for i in laser_power]
laser_max=[i*1.1 for i in laser_power1]
laser_min=[i*0.9 for i in laser_power1]

laser_max=laser_time+laser_max
laser_min=laser_time+laser_min

splits=[1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,3,3,3,3,3,3,1]
def multiply_sequence(seq):
    product = 1
    for num in seq:
        product *= num
    return product
splits_sum = multiply_sequence(splits)

laser_grid = sampler.uniform_sampling(laser_max, laser_min, splits)
rows, cols = laser_grid.shape


Laser_grid = np.zeros((rows, cols+1), dtype=laser_grid.dtype)
Laser_grid[:, :-1] = laser_grid[:, :]
Laser_grid[:, -2] = laser_grid[:, -2]
Laser_grid[:, -1] = laser_grid[:, -1]

print(Laser_grid[0,:])
print(np.size(Laser_grid, 0))
print(splits_sum)
#rows_to_keep = np.where(laser_grid[:, -2] == laser_grid[:, -3])[0]
#Laser_grid = laser_grid[rows_to_keep]

#program_name = 'Multi-1D-Test2'
program_name = 'Multi-1D-Sample'

def process_task(index, new_dir,Laser_grid):
    pm.generate_input_data1D(new_dir, index,Laser_grid[index])
    pm.run_command_1D(new_dir, index)

def thread_task(index, new_dir, Laser_grid):
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(process_task, index, new_dir, Laser_grid)


if __name__ == '__main__':
    new_dir = pm.init1D(program_name)
    with ProcessPoolExecutor(max_workers=50) as pool:
        pool.map(partial(thread_task, new_dir=new_dir, Laser_grid=Laser_grid), range(splits_sum))


