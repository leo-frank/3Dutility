from utils import *

colmap_poses = load_colmap_pose('blendedmvs/scene-Statues/sparse/0')
K = np.loadtxt('blendedmvs/scene-Statues/intrinsics.txt')
vis = gui_init()
show_mesh(vis, mesh=o3d.io.read_triangle_mesh('blendedmvs/scene-Statues/mesh.ply'))
show_cams(vis, colmap_poses, K, line_width=2)
gui_run()