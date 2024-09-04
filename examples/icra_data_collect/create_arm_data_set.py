"""
Created on Jul 18, 2024
@author: Tixian Wang
"""

import matplotlib.pyplot as plt
import numpy as np
print(np.__version__)
from tqdm import tqdm
# from tools import pos_dir_to_input
import sys
sys.path.append("../") # include elastica-python directory
sys.path.append("../../")       # include ActuationModel directory
import pickle
# import json
# import joblib
import h5py


color = ["C" + str(i) for i in range(10)]
# np.random.seed(2024)

folder_name = 'Data/'
filename = "simulation"

## simulation data
n_cases = 64
step_skip = 1

# ## data point setup
# n_data_pts = 8  # exlude the initial point at base
# idx_data_pts = np.array(
#     [int(100 / (n_data_pts)) * i for i in range(1, n_data_pts)] + [-1]
# )

# print(idx_data_pts)

# input_data = []
true_pos = []
true_dir = []
true_kappa = []
true_shear = []

for i in tqdm(range(n_cases)):
	with open(folder_name+filename+"_data_%03d.pickle"%i, "rb") as f:
		data = pickle.load(f)
		rod_data = data['systems'][0]
		sphere_data = data['systems'][1]
		recording_fps = data['recording_fps']

	position = np.array([rod_data["position"]])[0]
	director = np.array(rod_data['director'])
	kappa = np.array(rod_data['kappa'])
	shear = np.array(rod_data['sigma']) + np.array([0, 0, 1])[None, :, None]
	# print(position.shape, director.shape, kappa.shape, shear.shape)
	# quit()

	if i == 0:
		with open(folder_name+filename+"_systems_%03d.pickle"%i, "rb") as f:
			data = pickle.load(f)
			rod = data['systems'][0]
		
		n_elem = rod.n_elems
		dl = rod.rest_lengths
		L = sum(dl)
		s = np.linspace(0, L, n_elem + 1)
		s_mean = 0.5 * (s[1:] + s[:-1])
		radius = rod.radius
		bend_matrix = rod.bend_matrix
		shear_matrix = rod.shear_matrix

	true_pos.append(position[::step_skip, ...])
	true_dir.append(director[::step_skip, ...])
	true_kappa.append(kappa[::step_skip, ...])
	true_shear.append(shear[::step_skip, ...])

true_pos = np.vstack(true_pos)
true_dir = np.vstack(true_dir)
true_kappa = np.vstack(true_kappa)
true_shear = np.vstack(true_shear)
print(true_pos.shape, true_dir.shape, true_kappa.shape, true_shear.shape)

np.random.seed()
idx_list = np.random.randint(
	len(true_kappa), size=10
)  # [i*250 for i in range(10)]
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
fig2, axes = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 5))
for ii in range(len(idx_list)):
	i = idx_list[ii]
	ax.plot(
		true_pos[i, 0, :],
		true_pos[i, 1, :],
		true_pos[i, 2, :],
		ls="-",
		color=color[ii],
	)
	# ax.scatter(
	#     input_data[i, 0, :],
	#     input_data[i, 1, :],
	#     input_data[i, 2, :],
	#     s=50,
	#     marker="o",
	#     color=color[ii],
	# )
	ax.set_xlim(0, L)
	ax.set_ylim(0, L)
	ax.set_zlim(0, L)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel('z')
	ax.set_aspect("equal")
	for j in range(3):
		axes[0][j].plot(s[1:-1], true_kappa[i, j, :])
		axes[1][j].plot(s_mean, true_shear[i, j, :])
		axes[0][j].set_ylabel('$\\kappa_%d$'%(j+1))
		axes[1][j].set_ylabel('$\\nu_%d$'%(j+1))

# plt.show()
# quit()

# model_data = {
# 	"n_elem": n_elem,
# 	"L": L,
# 	"radius": radius,
# 	"s": s,
# 	"dl": dl,
# 	'bend_matrix': bend_matrix,
# 	'shear_matrix': shear_matrix
# }

data = {
	# "model": model_data,
	# "n_data_pts": n_data_pts,
	# "idx_data_pts": idx_data_pts,
	# "input_data": input_data,
	"n_elem": n_elem,
	"L": L,
	"radius": radius,
	"s": s,
	"dl": dl,
	'bend_matrix': bend_matrix,
	'shear_matrix': shear_matrix,
	"true_pos": true_pos,
	"true_dir": true_dir,
	"true_kappa": true_kappa,
	"true_shear": true_shear,
}

# print(type(n_elem), type(L), type(radius), type(s), type(dl), type(bend_matrix), type(shear_matrix), type(model_data), type(true_pos), type(true_dir), type(true_kappa), type(true_shear))

arm_data_name = 'octopus_arm_data' # 
# np.save(folder_name + arm_data_name + '.npy, data)
# with open(folder_name + arm_data_name + '.json', 'w') as f:
#     json.dump(data, f)
# Saving
# joblib.dump(data, folder_name + arm_data_name + '.joblib')
# with open(folder_name + arm_data_name + '.pickle', 'wb') as f:
#     pickle.dump(data, f)
with h5py.File(folder_name + arm_data_name + '.h5', 'w') as f:
    for key, value in data.items():
        f.create_dataset(key, data=value)

plt.show()
