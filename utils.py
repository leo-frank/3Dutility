import torch
import numpy as np
import tqdm
import os
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from loguru import logger
from easydict import EasyDict as edict

class Pose:
    """
    A class of operations on camera poses
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R, t):
        """construct a camera pose from the given R and/or t

        Args:
            R (torch.Tensor): shape [N, 3, 3]
            t (torch.Tensor): shape [N, 3] 

        Returns:
            pose (torch.Tensor): shape [N, 3, 4]
        """        
        assert isinstance(R, torch.Tensor) and isinstance(t, torch.Tensor)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [N, 3, 4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        """invert given pose

        Args:
            pose (torch.Tensor): shape [N, 3, 4]
            use_inverse (bool, optional): Defaults to False.

        Returns:
            pose_inv (torch.Tensor): shape [N, 3, 4]
        """        
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

pose_util = Pose()

class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
            ..., None, None
        ] % np.pi  # ln(R) will explode if theta==pi
        lnR = (
            1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
        )  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        # ret: (3,4 )
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack(
            [
                torch.stack([O, -w2, w1], dim=-1),
                torch.stack([w2, O, -w0], dim=-1),
                torch.stack([-w1, w0, O], dim=-1),
            ],
            dim=-2,
        )
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

lie_util = Lie()

def gui_init():
    """init a gui window

    Returns:
        visualizer: gui window with white background and some lightning effect
    """
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("img", width=1024, height=768)
    widget = gui.SceneWidget()
    widget.enable_scene_caching(True)
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -10, 0]
    )
    widget.scene.set_background([1, 1, 1, 1])  # White background
    window.add_child(widget)
    visualizer = widget.scene
    return visualizer


def create_frustum(K: np.ndarray, w2c: np.ndarray, ret_ray: bool = False):
    """create camera frustum with parameter [K|w2c], composed of lines

    Args:
        K (numpy.ndarray): intrinsic, (4,4)
        w2c (numpy.ndarray): camera pose (4,4)
        ret_ray (bool, optional): return a camera ray. Defaults to False.

    Returns:
        frustum (open3d.object)
        ray (open3d.object, optional)
    """
    near = 0.2  # near clipping plane
    far = 1000.0  # far clipping plane

    width = K[0, 2] * 2
    height = K[1, 2] * 2
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = np.array([0, width, width, 0])
    y = np.array([0, 0, height, height])
    z = np.array([1, 1, 1, 1])  # homogeneous coordinates
    x = (x - cx) * near / fx
    y = (y - cy) * near / fy
    z = z * near
    corners = np.stack([x, y, z]).T
    corners = np.concatenate((corners, np.array([[0, 0, 0]])), axis=0)
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]
        ),
    )
    c2w = np.linalg.inv(w2c)  # 4x4
    start = c2w[:3, 3]
    direction = c2w[:3, :3] @ np.array([0, 0, 1])
    step = 1
    end = start + direction * step
    ray = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array([start, end])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    if w2c is not None:
        frustum.transform(np.linalg.inv(w2c))
    if ret_ray == True:
        return frustum, ray
    else:
        return frustum

def show_pcd(visualizer, pcd_filename=None, name='pcd'):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    pcd_material = o3d.visualization.rendering.MaterialRecord()# 怎么都是0啊！怪不得显示不出来
    pcd_material.shader = "defaultLit"
    pcd_material.base_roughness = 0.15
    pcd_material.base_reflectance = 0.72
    pcd_material.point_size = 5
    color=[1, 0, 0]
    color = np.array(color).reshape(1, 3)
    color = np.repeat(color, np.asarray(pcd.points).shape[0], 0)
    pcd.colors = o3d.utility.Vector3dVector(color)
    visualizer.add_geometry(name, pcd, pcd_material)

def show_mesh(visualizer, mesh=None, verts: np.ndarray=None, faces:np.ndarray=None, name="mesh"):
    """show mesh if given, else contruct mesh from verts and faces. present with lightning
    """ 
    if mesh == None:  # construct mesh from vertices and faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    # special material setting of LevelS2FM
    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_roughness = 0.15
    mesh_material.base_reflectance = 0.72
    mesh_material.point_size = 2
    visualizer.add_geometry(name, mesh, mesh_material)

def show_cams(
    visualizer,
    w2c_poses: torch.Tensor,
    K: np.ndarray,
    color=[1, 0, 0],
    se3_type=False,
    name="cam",
    line_width=0.2,
) -> None:
    """show cameras with frustrum

    Args:
        visualizer: _description_
        w2c_poses (torch.Tensor): w2c, (N, 3, 4) if se3_type == False else (N, 6)
        K (torch.Tensor): intrinsic, (4,4)
        color (list, optional): Defaults to [1, 0, 0].
        se3_type (bool, optional): format of input poses. Defaults to False.
        name (str, optional): each geometry in open3d must has a unique name. Defaults to "cam".
        line_width (float, optional): line width. Defaults to 0.2.
    """

    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = line_width

    color = np.array(color).reshape(1, 3)
    for i, w2c in enumerate(w2c_poses):
        if se3_type == True:
            w2c = lie_util.se3_to_SE3(w2c)  # torch.tensor
        w2c = torch.cat([w2c, torch.tensor([[0, 0, 0, 1]], device=w2c.device)], dim=0)
        c2w = torch.inverse(w2c)
        frustum = create_frustum(K, w2c.detach().cpu().numpy())
        line_colors = np.repeat(color, 8, 0)
        frustum.colors = o3d.utility.Vector3dVector(line_colors)
        visualizer.add_geometry("{}_{}".format(name, i), frustum, line_material)

def gui_run():
    gui.Application.instance.run()


def load_colmap_pose(project_path: str):
    """load pose from colmap results

    Args:
        project_path (str): colmap result path, e.g., ./sparse/0

    Returns:
        poses(torch.Tensor): w2c, (N, 3, 4) note that poses are sorted by filename
    """    
    import pycolmap
    # Load the reconstruction project
    assert os.path.exists(project_path)
    logger.info("loading colmap pose from: {}".format(project_path))
    reconstruction = pycolmap.Reconstruction(project_path)
    poses_dict = {}
    for index, image in reconstruction.images.items():
        extrinsic = image.projection_matrix()  # (3,4) numpy w2c
        poses_dict[index] = [image.name, extrinsic]
    res = sorted(poses_dict.items(), key=lambda x: x[1][0], reverse=False)
    logger.info("image filenames: {}".format([i[1][0] for i in res])) # sorted filename 
    pose = [i[1][1] for i in res]  # from colmap
    pose = torch.tensor(np.stack(pose))
    return pose

def procrustes_analysis(X0,X1): # [N,3]                                  # 对齐操作
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    print(R)
    print(R.det())
    if R.det()<0: R[0] *= -1 # R[1] should multiply -1 # TODO: 这里检查一下！！！！
    print(R)
    print(R.det())
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def to_hom(X: torch.Tensor):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def cam2world(X: torch.Tensor, pose: torch.Tensor):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)

def prealign_cameras(pose: torch.Tensor, pose_GT: torch.Tensor):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3, device=pose.device)
    center_pred = cam2world(center, pose)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(
            center_GT, center_pred
        )  # 求解一个放射变换，将求解得到的pose与pose_gt进行对齐
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=pose.device))
    # align the camera poses
    center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    print("sim3: {}".format(sim3))
    R_aligned = pose[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    pose_aligned = pose_util(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3

def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle


def evaluate_camera_alignment(pose_aligned, pose_GT):
    """
    Inputs:
        pose_aligned:   (N, 3, 4) c2w
        pose_GT:        (N, 3, 4) c2w
    Outputs:
        R_error:        np.array
        t_error:        np.array
        t_scale_error:  np.array
        ATE_error:      np.array
    """
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)  # torch.Size([49, 3, 3])
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)  # t_GT: torch.Size([49, 3, 1])

    R_error = rotation_distance(R_aligned, R_GT)
    R_error = np.rad2deg(R_error.mean().cpu())

    t_error = (t_aligned - t_GT)[..., 0].norm(
        dim=-1
    )  # t_error ([49]) 每一项存放的是每一个点的位姿偏移量
    t_error = t_error.mean().cpu().numpy()

    ATE_error = (
        torch.sqrt(((t_aligned - t_GT)[..., 0] ** 2).sum(dim=-1).mean()).cpu().numpy()
    )

    avg_scale = (
        ((torch.norm(t_aligned[..., 0], dim=-1) + torch.norm(t_GT[..., 0], dim=-1)) / 2)
        .mean()
        .cpu()
        .numpy()
    )

    error = edict(R=R_error, t=t_error, t_scale=t_error / avg_scale, ATE=ATE_error)
    return error

def eval_poses(poses_all, poses_gt_all, mode="normal"):
    """
    Inputs:
        poses_all:      w2c, numpy | torch
        poses_gt_all:   w2c, numpy | torch
        mode:           'normal': return None
                        'ret_align': return aligned pose
    """
    if isinstance(poses_all, np.ndarray):
        poses_all = torch.tensor(poses_all)
    if isinstance(poses_gt_all, np.ndarray):
        poses_gt_all = torch.tensor(poses_gt_all)
        
    if poses_all.shape[0] > 2:
        pose_aligned, sim3 = prealign_cameras(poses_all.float(), poses_gt_all)
        error = evaluate_camera_alignment(
            pose_util.invert(
                pose_aligned
            ),  # convert w2c to c2w, c2w is suitable for compute loss
            pose_util.invert(poses_gt_all),
        )
        logger.info("rotation error: {}".format(error.R))
        logger.info("translation error: {}".format(error.t))
        # logger.info("translation/scale error: {}".format(error.t_scale))
        logger.info("ATE error: {}".format(error.ATE))
    else:
        logger.error("input poses has only {} images".format(poses_all.shape[0]))

    if mode == "ret_align":
        return pose_aligned
    elif mode == 'ret_sim3':
        return pose_aligned, sim3
    else:
        return error
