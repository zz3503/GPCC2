# # coding=utf-8
from tqdm import tqdm
import pyvista as pv
import numpy as np
import socket
import json
import struct
import sys
import cv2
import open3d as o3d
import os
import ctypes
import sophuspy as sp
import copy
import torch
import sys
import trimesh
import warnings
import nvdiffrast.torch as dr
from transformations import euler_matrix
from pytorch3d.transforms import so3_log_map, so3_exp_map
from bop_toolkit_lib.pose_error import re, te, vsd, mssd, mspd, add, adi, proj, cus
from Utils import nvdiffrast_render
from sampling import furthest_point_sample


def sampling_point(data, number):
    """
        Args:
            data: (M,3)
        Returns:
            data: (N,3)
    """
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, device='cuda', dtype=torch.float)
    if len(data) > number:
        idx = furthest_point_sample(data.unsqueeze(0).contiguous(), number).squeeze(0)
    else:
        idx = torch.multinomial(torch.ones(len(data)), number, replacement=True)
    return data[idx].contiguous()


# ==========================规范输出格式=========================== #
warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)

# ==========================连接至unity=========================== #
HOST = "127.0.0.1"
PORT = 18878


def call(fn: str, *args):
    """
    远程调用 Unity 服务器端 RPC 函数
    例：call("Add", 3, 5)
    """
    arg_json = [json.dumps({"value": arg}, ensure_ascii=False) for arg in args]
    req = {"fn": fn, "args": arg_json}
    body = json.dumps(req, ensure_ascii=False).encode("utf-8")
    head = struct.pack(">I", len(body))
    with socket.create_connection((HOST, PORT)) as sock:
        sock.sendall(head + body)
        head = sock.recv(4, socket.MSG_WAITALL)
        if len(head) != 4:
            raise RuntimeError("连接断开")
        (length,) = struct.unpack(">I", head)
        body = sock.recv(length, socket.MSG_WAITALL)
        resp = json.loads(body.decode("utf-8"))
        return resp


def getImage(downs=2):
    image_dict = json.loads(call("getImage", downs)['ret'])['image']
    rgb = np.array([[d[k] for k in ['r', 'g', 'b']] for d in image_dict])
    rgb = rgb.reshape(1440 // downs, 1920 // downs, 3)[:, :, ::-1] * 255  # rgb[0-1]->bgr[0-255]
    rgb = cv2.resize(rgb.astype(np.uint8), (1920, 1440))
    return rgb


def getPointClouds(downs=4):
    cloud_dict = json.loads(call("getPointClouds", downs)['ret'])['point']
    xyzrgb = np.array([[d[k] for k in ['x', 'y', 'z', 'r', 'g', 'b']] for d in cloud_dict])
    gray = ((xyzrgb[:, 3] * 0.299 + xyzrgb[:, 4] * 0.587 + xyzrgb[:, 5] * 0.114) * 255).astype(np.uint8)
    xyz = xyzrgb[:, :3][gray > 16]
    return xyz


def moveJ(joints):
    if type(joints) == np.ndarray:
        call("moveJ", joints.tolist())
    else:
        call("moveJ", joints)


def moveT(T):
    RTT = np.linalg.inv(np.array([[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    LTT = np.linalg.inv(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    T = LTT @ T @ RTT
    rx = np.arcsin(-T[1, 2])
    ry = np.arctan2(T[0, 2], T[2, 2])
    rz = np.arctan2(T[1, 0], T[1, 1])
    rx, ry, rz = rx * 180.0 / np.pi, ry * 180.0 / np.pi, rz * 180.0 / np.pi
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    return json.loads(call("moveT", rx, ry, rz, x * 10, y * 10, z * 10)['ret'])['value']


def getCurrentWaypoint():
    return list(map(float, json.loads(call("getCurrentWaypoint")['ret'])['value']))


def getTheta():
    return list(map(float, json.loads(call("getTheta")['ret'])['value']))


def setObjectId(idx):
    call("setObjectId", idx)


# ==========================手眼标定dll=========================== #
def xyz_rpy_to_T(rx, ry, rz, x, y, z):
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


class CPoseAndCloud(ctypes.Structure):
    _fields_ = [
        ("rx", ctypes.c_double), ("ry", ctypes.c_double), ("rz", ctypes.c_double),
        ("x", ctypes.c_double), ("y", ctypes.c_double), ("z", ctypes.c_double),
        ("cloudLen", ctypes.c_int),
        ("cloud", ctypes.POINTER(ctypes.c_double))
    ]


class CVectorPoseAndCloud(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int),
                ("items", ctypes.POINTER(CPoseAndCloud))]


dll = ctypes.CDLL(r'C:\Users\WRW\Desktop\jupyter\bylw\dll\ConsoleApplication2.dll')
dll.setHE.argtypes = [ctypes.c_double] * 6
dll.setHE.restype = None
dll.getHE.argtypes = [ctypes.POINTER(ctypes.c_double)] * 6
dll.getHE.restype = None
dll.setInputData.argtypes = [ctypes.POINTER(CVectorPoseAndCloud)]
dll.setInputData.restype = ctypes.c_bool


# ==========================新位姿生成=========================== #
def my_generate_spherical_points(num_points=42, r=1.):
    z = np.linspace(-1.0, 1.0, num_points)
    phi = np.arccos(z)
    theta = np.sqrt(num_points * np.pi) * phi
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.column_stack((x, y, z)) * r
    return points


def my_sample_views_icosphere(num_points=42, r=1.):
    points = my_generate_spherical_points(num_points, r)
    cam_in_obs = np.tile(np.eye(4)[None], (len(points), 1, 1))
    cam_in_obs[:, :3, 3] = points
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True)
    x_axis = np.cross(up, z_axis)
    x_axis = np.where(np.linalg.norm(x_axis, axis=-1, keepdims=True) > 0, x_axis, np.array([1, 0, 0]))
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True)
    cam_in_obs[:, :3, :3] = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return cam_in_obs


def make_rotation_grid(min_n_views=42, inplane_step=60, r=1.):
    cam_in_obs = my_sample_views_icosphere(min_n_views, r)
    rot_grid = []
    for i in range(len(cam_in_obs)):
        for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
            cam_in_ob = cam_in_obs[i]
            R_inplane = euler_matrix(0, 0, inplane_rot)
            cam_in_ob = cam_in_ob @ R_inplane
            ob_in_cam = np.linalg.inv(cam_in_ob)
            rot_grid.append(ob_in_cam)
    return np.asarray(rot_grid)


def 球面位姿生成(N=42, M=6, r=0.25):
    """
    :param N: 总采样点数
    :param M: 旋转次数
    :param r: 位姿半径
    :return:
        pose: (N*M,4,4) 位姿
    """
    I_i = np.linspace(-1.0, 1.0, N)
    ph_i = np.arccos(I_i)
    theta_i = np.sqrt(N * np.pi) * ph_i
    x_i = np.cos(theta_i) * np.sin(ph_i)
    y_i = np.sin(theta_i) * np.sin(ph_i)
    z_i = np.cos(ph_i)
    p_i = np.column_stack((x_i, y_i, z_i)) * r

    T_i = np.tile(np.eye(4)[None], (len(p_i), 1, 1))
    T_i[:, :3, 3] = p_i
    up = np.array([0, 0, 1])
    rz_i = -T_i[:, :3, 3]
    rz_i /= np.linalg.norm(rz_i, axis=-1, keepdims=True)
    rx_i = np.cross(up, rz_i)
    rx_i = np.where(np.linalg.norm(rx_i, axis=-1, keepdims=True) > 0, rx_i, np.array([1, 0, 0]))
    rx_i /= np.linalg.norm(rx_i, axis=-1, keepdims=True)
    ry_i = np.cross(rz_i, rx_i)
    ry_i /= np.linalg.norm(ry_i, axis=-1, keepdims=True)
    T_i[:, :3, :3] = np.stack((rx_i, ry_i, rz_i), axis=-1)

    TR_ij = []
    for i in range(len(T_i)):
        for j in np.deg2rad(np.linspace(0, 360, M, endpoint=False)):
            TR_ij.append(np.linalg.inv(T_i[i] @ euler_matrix(0, 0, j)))
    TR_ij = np.asarray(TR_ij)  # [N*M//2:,:,:]

    if False:
        pcds = []
        for i in range(len(TR_ij)):
            pcds.append(
                o3d.geometry.LineSet.create_camera_visualization(view_width_px=1920, view_height_px=1440, intrinsic=K,
                                                                 extrinsic=TR_ij[i], scale=0.075))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075, origin=[0, 0, 0]))
        o3d.visualization.draw_geometries([*pcds, o3d_mesh])
    return torch.as_tensor(TR_ij, dtype=torch.float, device='cuda')


def 可见点云生成(render_pose, num_point=1024):
    K = np.array([[2181.8, 0.0, 960.0], [0.0, 2181.8, 720.0], [0.0, 0.0, 1.0]])
    H, W = [1440, 1920]
    render_size = [320, 320]
    glctx = dr.RasterizeCudaContext()
    extra = {}
    nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=render_pose, get_normal=False, glctx=glctx, mesh=mesh,
                      output_size=render_size, use_light=True, extra=extra)
    render_point = []
    sparse = extra['xyz_map'].to_sparse(sparse_dim=3)
    value_buckets = torch.split(sparse.values(),
                                torch.bincount(sparse.indices()[0], minlength=len(render_pose)).tolist())
    for frame_point_cloud in value_buckets:
        render_point.append(sampling_point(frame_point_cloud, num_point))
    render_point = torch.stack(render_point)
    torch.cuda.empty_cache()
    return render_point


def 机械臂限制位姿(render_pose, T_base_obj, T_tool_cam):
    T_cam_obj = render_pose.cpu().numpy()
    T_cam_obj[:, :3, 3:] = T_cam_obj[:, :3, 3:] * 1000
    success_flag = []
    # print(len(T_cam_obj))
    for i in range(len(T_cam_obj)):
        T_base_tool = T_base_obj @ np.linalg.inv(T_cam_obj[i]) @ np.linalg.inv(T_tool_cam)
        T_base_tool[:3, 3:] = T_base_tool[:3, 3:] / 1000
        success_flag.append(moveT(T_base_tool))
        # print(getPointClouds(4).shape)
    T_cam_obj = T_cam_obj[success_flag]
    # print(len(T_cam_obj))
    return success_flag


def 最少位姿选择(render_pose, rate=0.9):
    def min_frames_to_cover_minimize_redundancy(complete_point_cloud, frame_point_cloud, rate):
        complete_set = set(complete_point_cloud)
        frame_sets = [set(frame) for frame in frame_point_cloud]
        selected_frames = []
        covered_set = set()
        index_seen_count = {idx: 0 for idx in complete_set}
        uncovered_set = complete_set.copy()
        while uncovered_set:
            max_covered = 0
            best_frame_idx = -1
            for i, frame_set in enumerate(frame_sets):
                uncovered_covered = len(frame_set & uncovered_set)
                if uncovered_covered > max_covered:
                    max_covered = uncovered_covered
                    best_frame_idx = i
            if best_frame_idx == -1:
                break
            selected_frames.append(best_frame_idx)
            new_covered_indices = frame_sets[best_frame_idx] - covered_set
            covered_set.update(new_covered_indices)
            uncovered_set = complete_set - covered_set
            for idx in frame_sets[best_frame_idx]:
                index_seen_count[idx] += 1
            if len(covered_set) >= len(complete_set) * rate:
                break
        uncovered_indices = uncovered_set
        return selected_frames, uncovered_indices, index_seen_count

    poseA = render_pose
    complete_point_clouds = 可见点云生成(render_pose, 8192)
    complete_point_clouds = complete_point_clouds.cpu().numpy()
    # print(complete_point_clouds.shape)
    pcd = []
    for i in range(len(complete_point_clouds)):
        pcd.append(o3d.geometry.PointCloud())
        pcd[-1].points = o3d.utility.Vector3dVector(complete_point_clouds[i])
        pcd[-1].transform(np.linalg.inv(poseA[i].cpu().numpy()))
    voxel_size = 0.002
    # voxel_size = np.mean([np.mean(i.compute_nearest_neighbor_distance()) for i in pcd]) * 2
    # o3d.io.write_point_cloud(rf'C:\Users\WRW\Desktop\111\temp.pcd', pcd[0], write_ascii=False)

    merged_pcd = o3d.geometry.PointCloud()
    for cloud in pcd:
        merged_pcd += cloud

    global_key2idx = {tuple(map(int, idx / voxel_size)): i for i, idx in enumerate(np.array(merged_pcd.points))}
    complete_point_clouds_tensor_idx = list(global_key2idx.values())
    # print(len(complete_point_clouds_tensor_idx))
    frame_point_clouds_tensor_idx = []
    for cloud in tqdm(pcd, desc='voxel idx'):
        idx = [global_key2idx.get(tuple(map(int, k / voxel_size)), -1) for k in np.array(cloud.points)]
        frame_point_clouds_tensor_idx.append(np.asarray(idx)[np.asarray(idx) != -1])

    selected_frames, uncovered_indices, index_seen_count = min_frames_to_cover_minimize_redundancy(
        complete_point_clouds_tensor_idx, frame_point_clouds_tensor_idx, rate)
    print("最少需要的帧数：{}, 索引为:：{}".format(len(selected_frames), selected_frames))
    print("覆盖率：", 1 - len(uncovered_indices) / len(complete_point_clouds_tensor_idx))
    if False:
        pcds = []
        for i in range(len(render_pose)):
            if i in selected_frames:
                pcds.append(o3d.geometry.LineSet.create_camera_visualization(view_width_px=1920, view_height_px=1440,
                                                                             intrinsic=K,
                                                                             extrinsic=render_pose[i].cpu().numpy(),
                                                                             scale=0.075))
                pcds[-1].paint_uniform_color([1, 0, 0])
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075, origin=[0, 0, 0]))
        o3d.visualization.draw_geometries([*pcds, o3d_mesh])
    return render_pose[selected_frames]


def 真实位姿估计(gt_point, temp=None):
    """
    :param gt_point: 真实点云 (Nx3)
    :return:
        gt_pose: 真实位姿（粗配准 + 精配准）
    """
    radius = mesh.bounding_sphere.primitive.radius
    render_pose = 球面位姿生成(r=max(0.25, radius * 3))
    center = np.mean(gt_point, axis=0, keepdims=True).T
    render_pose[:, :3, 3:] = torch.as_tensor(center, dtype=torch.float, device='cuda')

    if temp is not None:
        scores = []
        for i in range(len(render_pose)):
            scores.append(re(render_pose[i].cpu().numpy()[:3, :3], temp[:3, :3]))
        flag = torch.zeros(len(render_pose), dtype=torch.bool)
        for i in np.argsort(scores)[:len(render_pose) // 2]:
            flag[i] = True
        render_pose = render_pose[flag]

    coarse_num_point = 2 ** 10
    coarse_gt_point = sampling_point(gt_point, coarse_num_point).cpu().numpy()
    A = o3d.geometry.PointCloud()
    A.points = o3d.utility.Vector3dVector(coarse_gt_point)
    voxel_size = np.mean(A.compute_nearest_neighbor_distance()) * 10
    A.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    render_point = 可见点云生成(render_pose, coarse_num_point).cpu().numpy()
    render_pcds = []
    for pts in render_point:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        render_pcds.append(pcd)

    scores = []
    for i in range(len(render_pcds)):
        B = render_pcds[i]
        B.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
        icp = o3d.pipelines.registration.registration_icp(
            B, A,
            max_correspondence_distance=voxel_size * 5,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
        )
        scores.append(1 if icp.inlier_rmse == 0 else icp.inlier_rmse)
        render_pose[i] = torch.as_tensor(icp.transformation, dtype=torch.float, device='cuda') @ render_pose[i]

    render_pose = render_pose[np.argsort(scores)[:8]]

    fine_num_point = 2 ** 15
    fine_gt_point = sampling_point(gt_point, fine_num_point).cpu().numpy()
    A = o3d.geometry.PointCloud()
    A.points = o3d.utility.Vector3dVector(fine_gt_point)
    voxel_size = np.mean(A.compute_nearest_neighbor_distance()) * 10
    A.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=60))
    render_point = 可见点云生成(render_pose, fine_num_point).cpu().numpy()
    render_pcds = []
    for pts in render_point:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        render_pcds.append(pcd)
    scores = []
    for i in range(len(render_pcds)):
        B = render_pcds[i]
        B.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=60))
        icp = o3d.pipelines.registration.registration_icp(
            B, A,
            max_correspondence_distance=voxel_size * 5,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        scores.append(1 if icp.inlier_rmse == 0 else icp.inlier_rmse)
        render_pose[i] = torch.as_tensor(icp.transformation, dtype=torch.float, device='cuda') @ render_pose[i]
    return render_pose[np.argmin(scores)].cpu().numpy()


def 机械臂末端位姿(render_pose, T_base_obj, T_tool_cam):
    T_cam_obj = render_pose.cpu().numpy()
    T_cam_obj[:, :3, 3:] = T_cam_obj[:, :3, 3:] * 1000
    ans = []
    for i in range(len(T_cam_obj)):
        T_base_tool = T_base_obj @ np.linalg.inv(T_cam_obj[i]) @ np.linalg.inv(T_tool_cam)
        T_base_tool[:3, 3:] = T_base_tool[:3, 3:] / 1000
        ans.append(T_base_tool)
    return np.array(ans)


class draw_point:
    def __init__(self, gt_points, render_pose):
        from pyvistaqt import BackgroundPlotter
        # ---------- 新增全局变量 ----------
        self.gt_points = gt_points
        self.render_pose = render_pose
        self.all_points = []
        self.hist_points = []  # 所有历史点（绿）
        self.new_points = []  # 本轮新点（红）
        self.hist_actor = None  # 绿色历史点 actor
        self.new_actor = None  # 红色新点 actor
        self.frame = 0
        self.success_num = 0
        self.cam_actor = None  # 相机框 actor
        self.K = np.array([[2181.8, 0.0, 960.0], [0.0, 2181.8, 720.0], [0.0, 0.0, 1.0]])  # 你的内参
        self.cam_scale = 100  # 相机框大小
        # ---------- 启动 ----------
        self.plotter = BackgroundPlotter()
        self.plotter.set_background('white')
        self.plotter.camera_position = [(372.137111608413, 380.97708403767683, 419.2962991521736),  # 相机 eye
                                        (-58.166995955375285, -49.32702352611199, -11.007808411615382),  # look-at 中心
                                        (0.0, 0.0, 1.0)]
        time.sleep(1)
        self.update_cloud()
        self.plotter.add_callback(self.update_cloud, 3000)
        self.plotter.app.exec_()

    def make_camera_lineset(self, temp_T_cam_obj):
        """
        T_cam_world: 4×4，OpenCV/Open3D 格式（世界系→相机系）
        返回: pv.PolyData（lines）
        """
        cam_lines = o3d.geometry.LineSet.create_camera_visualization(1920, 1440, self.K, temp_T_cam_obj, self.cam_scale)
        pts = np.asarray(cam_lines.points)  # 8×3
        idx = np.asarray(cam_lines.lines)  # 24×2
        cells = []
        for i, j in idx:
            cells.extend([2, i, j])
        return pv.PolyData(pts, lines=np.array(cells))

    def update_cloud(self):
        if self.frame >= len(self.gt_points):
            return
        self.new_pts = np.asarray(self.gt_points[self.frame])
        self.all_points.append(self.new_pts.copy())
        self.new_points = self.new_pts
        self.hist_points.extend(self.new_points)
        self.hist_points = sampling_point(np.asarray(self.hist_points), 2048).cpu().tolist()
        self.hist_cloud = pv.PolyData(np.asarray(self.hist_points))
        self.new_cloud = pv.PolyData(np.asarray(self.new_points))
        for self.actor in [self.hist_actor, self.new_actor]:
            if self.actor is not None:
                self.plotter.remove_actor(self.actor)
        self.hist_actor = self.plotter.add_mesh(self.hist_cloud, color=(0, 255, 0), point_size=4,
                                                render_points_as_spheres=True)
        self.new_actor = self.plotter.add_mesh(self.new_cloud, color=(255, 0, 0), point_size=4,
                                               render_points_as_spheres=True)
        if self.cam_actor is not None:
            self.plotter.remove_actor(self.cam_actor)
        self.render_pose[self.frame][:3, 3:] *= 1000
        cam_pv = self.make_camera_lineset(self.render_pose[self.frame])
        self.cam_actor = self.plotter.add_mesh(cam_pv, color='yellow', line_width=3, render_lines_as_tubes=True)
        self.plotter.reset_camera_clipping_range()
        self.frame += 1


import time

sum_CD_robot = 0
sum_HD_robot = 0
sum_CD_pose = 0
sum_HD_pose = 0
if __name__ == '__main__':
    for idx in range(1, 19):
        if idx in [3, 16, 17, 18]:
            continue
        setObjectId(idx)
        mesh_ply_path = rf'E:\毕业论文\GPCC2\data\obj_{idx:06d}.obj'
        mesh = trimesh.load(mesh_ply_path)
        mesh.apply_scale(0.001)
        # 左手坐标系转右手坐标系
        mesh.apply_transform(xyz_rpy_to_T(np.pi / 2, 0, np.pi / 2, 0, 0, 0))
        mesh.apply_transform(xyz_rpy_to_T(0, -np.pi / 2, 0, 0, 0, 0))
        radius = mesh.bounding_sphere.primitive.radius
        mesh_center = mesh.bounds.mean(axis=0)
        mesh.apply_transform(trimesh.transformations.translation_matrix(-mesh_center))

        o3d_mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
        o3d_mesh.scale(1 / 1000.0, center=[0, 0, 0])
        o3d_mesh.transform(xyz_rpy_to_T(np.pi / 2, 0, np.pi / 2, 0, 0, 0))
        o3d_mesh.transform(xyz_rpy_to_T(0, -np.pi / 2, 0, 0, 0, 0))
        o3d_mesh.translate(-mesh_center, relative=True)

        moveJ([-83, -30, -158, 156, 102, 28])
        gt_point = getPointClouds(4)
        T_base_tool = xyz_rpy_to_T(*getCurrentWaypoint())
        # T_tool_cam = sp.SE3.exp(np.array(
        #     [-24.85568346396407, 24.313245342898707, 49.021704346935046, 0.0008666744879287994, 0.0032395116075079485,
        #      -1.5745586789349082])).matrix()
        T_tool_cam = sp.SE3.exp(np.array(
            [-23.226367168247968, 24.061234563984087, 50.06320345011153, -0.00018035573675309984,
             -0.00013002876111078876, -1.5732829669179795])).matrix()

        T_cam_obj = 真实位姿估计(gt_point)
        TT = T_cam_obj.copy()
        T_cam_obj[:3, 3:] *= 1000
        T_base_obj = T_base_tool @ T_tool_cam @ T_cam_obj
        # T_base_obj[:3, 3:] = T_base_obj[:3, 3:] - mesh_center.reshape(3, 1) * 1000

        # print(T_base_obj)
        # render_point = 可见点云生成(torch.as_tensor(TT, device='cuda').unsqueeze(0), len(gt_point)).cpu().numpy()[0]
        #
        # pcds = []
        # pcds.append(o3d.geometry.PointCloud())
        # pcds[-1].points = o3d.utility.Vector3dVector(render_point)
        # pcds[-1].colors = o3d.utility.Vector3dVector(np.ones_like(render_point))
        # pcds[-1].transform(np.linalg.inv(TT))
        # pcds.append(o3d.geometry.PointCloud())
        # pcds[-1].points = o3d.utility.Vector3dVector(gt_point)
        # pcds[-1].transform(np.linalg.inv(TT))
        # o3d.visualization.draw_geometries([*pcds, o3d_mesh])

        render_pose = 球面位姿生成(r=max(0.25, radius * 3))[126:, :, :]
        # render_pose = render_pose @ torch.as_tensor(
        #     trimesh.transformations.translation_matrix(-mesh_center)).cuda().float().unsqueeze(0)

        # TR_ij = render_pose.cpu().numpy()
        # pcds = []
        # K = np.array([[2181.8, 0.0, 960.0], [0.0, 2181.8, 720.0], [0.0, 0.0, 1.0]])
        # for i in range(len(TR_ij)):
        #     pcds.append(
        #         o3d.geometry.LineSet.create_camera_visualization(view_width_px=1920, view_height_px=1440, intrinsic=K,
        #                                                          extrinsic=TR_ij[i], scale=0.075))
        # pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075, origin=[0, 0, 0]))
        # o3d.visualization.draw_geometries([*pcds, o3d_mesh])

        success_flag = 机械臂限制位姿(render_pose, T_base_obj, T_tool_cam)
        render_pose = render_pose[success_flag]
        render_pose = 最少位姿选择(render_pose, 0.9)
        T_base_tool = 机械臂末端位姿(render_pose, T_base_obj, T_tool_cam)
        # print(T_base_tool.shape)

        all_points = []
        pcds = o3d.geometry.PointCloud()
        method = '机械臂拼接'
        for i in range(len(T_base_tool)):
            moveT(T_base_tool[i])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(getPointClouds())
            if method == '机械臂拼接':  # 模拟机械臂拼接，只引入了手眼标定的误差，忽略了机械臂运动的误差
                pcd.transform(np.linalg.inv(render_pose[i].cpu().numpy()))
            elif method == '位姿估计拼接':
                pcd.transform(np.linalg.inv(真实位姿估计(getPointClouds(4), render_pose[i].cpu().numpy())))
            all_points.append(np.array(pcd.points) * 1000.0)
            pcds += pcd
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcds, o3d_mesh, mesh_frame])

        # temp = o3d_mesh.sample_points_uniformly(2 ** 16)

        target_spacing = 0.4 / 1000.0
        surface_area = o3d_mesh.get_surface_area()
        num_points = int(surface_area / (target_spacing ** 2))
        temp = o3d_mesh.sample_points_uniformly(number_of_points=num_points)

        print(rf'机械臂拼接id：{idx}')
        print('平均点间距：', np.mean(temp.compute_nearest_neighbor_distance()) * 1000, 'mm')
        print('CD：', np.mean(pcds.compute_point_cloud_distance(temp)) * 1000, 'mm')
        print('HD：', np.max(pcds.compute_point_cloud_distance(temp)) * 1000, 'mm')
        sum_CD_robot += np.mean(pcds.compute_point_cloud_distance(temp)) * 1000
        sum_HD_robot += np.max(pcds.compute_point_cloud_distance(temp)) * 1000

        # draw_point(all_points, render_pose.cpu().numpy())

        all_points = []
        pcds = o3d.geometry.PointCloud()
        method = '位姿估计拼接'
        for i in range(len(T_base_tool)):
            moveT(T_base_tool[i])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(getPointClouds())
            if method == '机械臂拼接':
                pcd.transform(np.linalg.inv(render_pose[i].cpu().numpy()))
            elif method == '位姿估计拼接':
                pcd.transform(np.linalg.inv(真实位姿估计(getPointClouds(4), render_pose[i].cpu().numpy())))
            all_points.append(np.array(pcd.points) * 1000.0)
            pcds += pcd
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcds, o3d_mesh, mesh_frame])

        print(rf'位姿估计拼接id：{idx}')
        print('平均点间距：', np.mean(temp.compute_nearest_neighbor_distance()) * 1000, 'mm')
        print('CD：', np.mean(pcds.compute_point_cloud_distance(temp)) * 1000, 'mm')
        print('HD：', np.max(pcds.compute_point_cloud_distance(temp)) * 1000, 'mm')
        sum_CD_pose += np.mean(pcds.compute_point_cloud_distance(temp)) * 1000
        sum_HD_pose += np.max(pcds.compute_point_cloud_distance(temp)) * 1000
        # draw_point(all_points, render_pose.cpu().numpy())
    print('sum_CD_robot：', sum_CD_robot / 14)
    print('sum_HD_robot：', sum_HD_robot / 14)
    print('sum_CD_pose：', sum_CD_pose / 14)
    print('sum_HD_pose：', sum_HD_pose / 14)

# voxel idx: 100%|██████████| 118/118 [00:03<00:00, 38.88it/s]
# 最少需要的帧数：5, 索引为:：[25, 51, 5, 42, 8]
# 覆盖率： 0.9057250599931437
# 机械臂拼接id：1
# 平均点间距： 0.20291409512358913 mm
# CD： 0.4431522796010788 mm
# HD： 1.5323040950327074 mm
# 位姿估计拼接id：1
# 平均点间距： 0.20291409512358913 mm
# CD： 0.26414138281591143 mm
# HD： 0.9362326411244677 mm
# voxel idx: 100%|██████████| 92/92 [00:02<00:00, 36.48it/s]
# 最少需要的帧数：13, 索引为:：[18, 73, 4, 33, 17, 19, 36, 31, 2, 21, 71, 12, 20]
# 覆盖率： 0.9058462325700396
# 机械臂拼接id：2
# 平均点间距： 0.2041442309633137 mm
# CD： 0.5802506910679315 mm
# HD： 3.008992359319238 mm
# 位姿估计拼接id：2
# 平均点间距： 0.2041442309633137 mm
# CD： 0.35837351921445826 mm
# HD： 2.2814664513195453 mm
# voxel idx: 100%|██████████| 111/111 [00:03<00:00, 36.92it/s]
# 最少需要的帧数：8, 索引为:：[76, 4, 24, 39, 40, 17, 6, 30]
# 覆盖率： 0.9106385671792925
# 机械臂拼接id：4
# 平均点间距： 0.20276209579690624 mm
# CD： 0.45095519787560384 mm
# HD： 1.7221304114729765 mm
# 位姿估计拼接id：4
# 平均点间距： 0.20276209579690624 mm
# CD： 0.31810283408294465 mm
# HD： 1.4311832146141277 mm
# voxel idx: 100%|██████████| 101/101 [00:02<00:00, 36.66it/s]
# 最少需要的帧数：9, 索引为:：[35, 10, 39, 27, 2, 23, 34, 13, 40]
# 覆盖率： 0.9049783210573056
# 机械臂拼接id：5
# 平均点间距： 0.2031017635651993 mm
# CD： 0.5056397889001765 mm
# HD： 2.206042518954413 mm
# 位姿估计拼接id：5
# 平均点间距： 0.2031017635651993 mm
# CD： 0.34896265130697174 mm
# HD： 1.7806688281850165 mm
# voxel idx: 100%|██████████| 118/118 [00:02<00:00, 39.53it/s]
# 最少需要的帧数：8, 索引为:：[98, 19, 39, 5, 51, 14, 32, 4]
# 覆盖率： 0.9036156918979822
# 机械臂拼接id：6
# 平均点间距： 0.20517191806352172 mm
# CD： 0.4281241235617821 mm
# HD： 1.6923534293328946 mm
# 位姿估计拼接id：6
# 平均点间距： 0.20517191806352172 mm
# CD： 0.29495656829436473 mm
# HD： 1.1343325031937408 mm
# voxel idx: 100%|██████████| 117/117 [00:03<00:00, 38.38it/s]
# 最少需要的帧数：10, 索引为:：[111, 18, 50, 59, 41, 112, 107, 4, 27, 98]
# 覆盖率： 0.9086158089569573
# 机械臂拼接id：7
# 平均点间距： 0.20336247118059556 mm
# CD： 0.46589400850606116 mm
# HD： 1.944123565569594 mm
# 位姿估计拼接id：7
# 平均点间距： 0.20336247118059556 mm
# CD： 0.33497916518768217 mm
# HD： 1.608086779522663 mm
# voxel idx: 100%|██████████| 90/90 [00:02<00:00, 36.53it/s]
# 最少需要的帧数：10, 索引为:：[53, 71, 7, 33, 27, 2, 70, 24, 77, 14]
# 覆盖率： 0.9104492380763902
# 机械臂拼接id：8
# 平均点间距： 0.20147921343728473 mm
# CD： 0.7320383956867824 mm
# HD： 3.584137830763202 mm
# 位姿估计拼接id：8
# 平均点间距： 0.20147921343728473 mm
# CD： 0.3644568122428872 mm
# HD： 2.109473262171204 mm
# voxel idx: 100%|██████████| 117/117 [00:02<00:00, 39.48it/s]
# 最少需要的帧数：6, 索引为:：[67, 46, 2, 26, 49, 34]
# 覆盖率： 0.9007711334206314
# 机械臂拼接id：9
# 平均点间距： 0.20289688225533703 mm
# CD： 0.4343925042828171 mm
# HD： 1.5926687869368599 mm
# 位姿估计拼接id：9
# 平均点间距： 0.20289688225533703 mm
# CD： 0.2710543873556925 mm
# HD： 0.9090830856401502 mm
# voxel idx: 100%|██████████| 118/118 [00:03<00:00, 38.07it/s]
# 最少需要的帧数：8, 索引为:：[35, 4, 50, 20, 39, 8, 24, 49]
# 覆盖率： 0.9043809523809524
# 机械臂拼接id：10
# 平均点间距： 0.20340460684070544 mm
# CD： 0.496076888056589 mm
# HD： 2.137847026473672 mm
# 位姿估计拼接id：10
# 平均点间距： 0.20340460684070544 mm
# CD： 0.2890957047639954 mm
# HD： 1.302410446924812 mm
# voxel idx: 100%|██████████| 111/111 [00:02<00:00, 40.97it/s]
# 最少需要的帧数：4, 索引为:：[29, 46, 26, 1]
# 覆盖率： 0.9058726673984632
# 机械臂拼接id：11
# 平均点间距： 0.20279947501730772 mm
# CD： 0.47509327022925996 mm
# HD： 2.1540890144703755 mm
# 位姿估计拼接id：11
# 平均点间距： 0.20279947501730772 mm
# CD： 0.2395101033430541 mm
# HD： 1.3700299906067503 mm
# voxel idx: 100%|██████████| 118/118 [00:02<00:00, 39.62it/s]
# 最少需要的帧数：8, 索引为:：[37, 3, 21, 44, 63, 31, 99, 38]
# 覆盖率： 0.9090498758745205
# 机械臂拼接id：12
# 平均点间距： 0.20286053160098722 mm
# CD： 0.4813814890305525 mm
# HD： 1.9558840481695563 mm
# 位姿估计拼接id：12
# 平均点间距： 0.20286053160098722 mm
# CD： 0.317297328961123 mm
# HD： 1.1931784381101855 mm
# voxel idx: 100%|██████████| 87/87 [00:02<00:00, 36.75it/s]
# 最少需要的帧数：11, 索引为:：[86, 14, 66, 5, 23, 68, 19, 38, 30, 67, 11]
# 覆盖率： 0.9014108889465939
# 机械臂拼接id：13
# 平均点间距： 0.20174783728269294 mm
# CD： 0.5519138699334224 mm
# HD： 2.5856519780744467 mm
# 位姿估计拼接id：13
# 平均点间距： 0.20174783728269294 mm
# CD： 0.3633298451235082 mm
# HD： 2.0811092638060473 mm
# voxel idx: 100%|██████████| 87/87 [00:02<00:00, 37.41it/s]
# 最少需要的帧数：11, 索引为:：[47, 32, 38, 24, 71, 31, 0, 20, 63, 36, 30]
# 覆盖率： 0.9012016021361816
# 机械臂拼接id：14
# 平均点间距： 0.20254095042613765 mm
# CD： 0.6535196819261435 mm
# HD： 3.13636518591924 mm
# 位姿估计拼接id：14
# 平均点间距： 0.20254095042613765 mm
# CD： 0.500594679628056 mm
# HD： 3.822120050371192 mm
# voxel idx: 100%|██████████| 97/97 [00:02<00:00, 38.79it/s]
# 最少需要的帧数：9, 索引为:：[60, 5, 77, 15, 33, 0, 20, 39, 31]
# 覆盖率： 0.9088485464009141
# 机械臂拼接id：15
# 平均点间距： 0.20237310067959485 mm
# CD： 0.507425656522697 mm
# HD： 2.515998698596933 mm
# 位姿估计拼接id：15
# 平均点间距： 0.20237310067959485 mm
# CD： 0.30177554708330645 mm
# HD： 2.0503038484313594 mm
# sum_CD_robot： 0.5147041317986355
# sum_HD_robot： 2.2691849249347213
# sum_CD_pose： 0.32618789495742545
# sum_HD_pose： 1.7149770574300902
