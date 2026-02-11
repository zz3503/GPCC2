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
import time


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
    for cloud in pcd:
        idx = [global_key2idx.get(tuple(map(int, k / voxel_size)), -1) for k in np.array(cloud.points)]
        frame_point_clouds_tensor_idx.append(np.asarray(idx)[np.asarray(idx) != -1])

    selected_frames, uncovered_indices, index_seen_count = min_frames_to_cover_minimize_redundancy(
        complete_point_clouds_tensor_idx, frame_point_clouds_tensor_idx, rate)
    return render_pose[selected_frames], len(selected_frames), 1 - len(uncovered_indices) / len(
        complete_point_clouds_tensor_idx)


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


def 固定位姿(all_render_pose, select_render_pose):
    poseA = all_render_pose
    complete_point_clouds = 可见点云生成(all_render_pose, 8192)
    complete_point_clouds = complete_point_clouds.cpu().numpy()
    # print(complete_point_clouds.shape)
    pcd = []
    for i in range(len(complete_point_clouds)):
        pcd.append(o3d.geometry.PointCloud())
        pcd[-1].points = o3d.utility.Vector3dVector(complete_point_clouds[i])
        pcd[-1].transform(np.linalg.inv(poseA[i].cpu().numpy()))
    voxel_size = 0.002
    # voxel_size = np.mean([np.mean(i.compute_nearest_neighbor_distance()) for i in pcd]) * 2
    merged_pcdA = o3d.geometry.PointCloud()
    for cloud in pcd:
        merged_pcdA += cloud

    poseB = select_render_pose
    complete_point_clouds = 可见点云生成(select_render_pose, 8192)
    complete_point_clouds = complete_point_clouds.cpu().numpy()
    # print(complete_point_clouds.shape)
    pcd = []
    for i in range(len(complete_point_clouds)):
        pcd.append(o3d.geometry.PointCloud())
        pcd[-1].points = o3d.utility.Vector3dVector(complete_point_clouds[i])
        pcd[-1].transform(np.linalg.inv(poseB[i].cpu().numpy()))
    merged_pcdB = o3d.geometry.PointCloud()
    for cloud in pcd:
        merged_pcdB += cloud

    global_key2idx = {tuple(map(int, idx / voxel_size)): i for i, idx in enumerate(np.array(merged_pcdA.points))}
    all_num = len(global_key2idx)
    idx = [global_key2idx.get(tuple(map(int, k / voxel_size)), -1) for k in np.array(merged_pcdB.points)]
    select_num = len(set((np.asarray(idx)[np.asarray(idx) != -1])))

    return select_num / all_num


mean_4_gd = 0
mean_4_bz = 0
mean_8_gd = 0
mean_8_bz = 0
mean_12_gd = 0
mean_12_bz = 0
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
        T_tool_cam = sp.SE3.exp(np.array(
            [-24.85568346396407, 24.313245342898707, 49.021704346935046, 0.0008666744879287994, 0.0032395116075079485,
             -1.5745586789349082])).matrix()
        T_cam_obj = 真实位姿估计(gt_point)
        TT = T_cam_obj.copy()
        T_cam_obj[:3, 3:] *= 1000
        T_base_obj = T_base_tool @ T_tool_cam @ T_cam_obj

        render_pose = 球面位姿生成(r=max(0.25, radius * 3))[126:, :, :]
        success_flag = 机械臂限制位姿(render_pose, T_base_obj, T_tool_cam)
        render_pose = render_pose[success_flag]
        rate = 0.5
        frame = 0
        while frame < 12:
            _, frame, rate = 最少位姿选择(render_pose, rate)
            if frame in [4, 8, 12]:
                print(idx, frame, rate)
                if frame == 4:
                    mean_4_bz += rate
                if frame == 8:
                    mean_8_bz += rate
                if frame == 12:
                    mean_12_bz += rate
            rate += 0.001

        for view_num in [4, 8, 12]:
            temp_T = [np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, max(0.25, radius * 3)], [0, 0, 0, 1.]])] * view_num
            if view_num == 4:
                temp_T[0] = temp_T[0] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 0 / 3, 0, 0, 0, 0)
                temp_T[1] = temp_T[1] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 1 / 3 + np.pi / 8, 0, 0, 0, 0)
                temp_T[2] = temp_T[2] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 2 / 3, 0, 0, 0, 0)
                temp_T[3] = temp_T[3] @ xyz_rpy_to_T(np.pi, 0, 0, 0, 0, 0)

            if view_num == 8:
                temp_T[0] = temp_T[0] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 0 / 4 + np.pi / 4 - np.pi / 8, 0, 0, 0, 0)
                temp_T[1] = temp_T[1] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 1 / 4 + np.pi / 4 + np.pi / 8, 0, 0, 0, 0)
                temp_T[2] = temp_T[2] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 2 / 4 + np.pi / 4, 0, 0, 0, 0)
                temp_T[3] = temp_T[3] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 3 / 4 + np.pi / 4, 0, 0, 0, 0)
                temp_T[4] = temp_T[4] @ xyz_rpy_to_T(np.pi / 4 * 3, 0, 0, 0, 0, 0)
                temp_T[5] = temp_T[5] @ xyz_rpy_to_T(np.pi / 4 * 5, 0, 0, 0, 0, 0)
                temp_T[6] = temp_T[6] @ xyz_rpy_to_T(0, np.pi / 4 * 3, 0, 0, 0, 0)
                temp_T[7] = temp_T[7] @ xyz_rpy_to_T(0, np.pi / 4 * 5, 0, 0, 0, 0)

            if view_num == 12:
                temp_T[0] = temp_T[0] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 0 / 3, 0, 0, 0, 0)
                temp_T[1] = temp_T[1] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 0 / 4 + np.pi / 4 - np.pi / 8, 0, 0, 0, 0)
                temp_T[2] = temp_T[2] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 1 / 4 + np.pi / 4 + np.pi / 8, 0, 0, 0, 0)
                temp_T[3] = temp_T[3] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 1 / 3 + np.pi / 8, 0, 0, 0, 0)
                temp_T[4] = temp_T[4] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 2 / 4 + np.pi / 4, 0, 0, 0, 0)
                temp_T[5] = temp_T[5] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 2 / 3, 0, 0, 0, 0)
                temp_T[6] = temp_T[6] @ xyz_rpy_to_T(np.pi / 2, 2 * np.pi * 3 / 4 + np.pi / 4, 0, 0, 0, 0)
                temp_T[7] = temp_T[7] @ xyz_rpy_to_T(np.pi / 4 * 3, 0, 0, 0, 0, 0)
                temp_T[8] = temp_T[8] @ xyz_rpy_to_T(np.pi / 4 * 5, 0, 0, 0, 0, 0)
                temp_T[9] = temp_T[9] @ xyz_rpy_to_T(0, np.pi / 4 * 3, 0, 0, 0, 0)
                temp_T[10] = temp_T[10] @ xyz_rpy_to_T(0, np.pi / 4 * 5, 0, 0, 0, 0)
                temp_T[11] = temp_T[11] @ xyz_rpy_to_T(np.pi, 0, 0, 0, 0, 0)
            select_render_pose = torch.as_tensor(temp_T.copy(), dtype=torch.float, device='cuda')
            rate = 固定位姿(render_pose, select_render_pose)
            print(idx, view_num, rate)
            if view_num == 4:
                mean_4_gd += rate
            if view_num == 8:
                mean_8_gd += rate
            if view_num == 12:
                mean_12_gd += rate

print(mean_4_gd / 14)
print(mean_4_bz / 14)
print(mean_8_gd / 14)
print(mean_8_bz / 14)
print(mean_12_gd / 14)
print(mean_12_bz / 14)
# 1 4 0.8827562564278368
# 1 8 0.9348645869043538
# 1 12 0.9573191635241687
# 1 4 0.878813849845732
# 1 8 0.9249228659581762
# 1 12 0.947891669523483
# 2 4 0.7133738989830221
# 2 8 0.8456235904855113
# 2 12 0.8966001446746947
# 2 4 0.6789498319220458
# 2 8 0.8324326624398962
# 2 12 0.8818348155397643
# 4 4 0.8429846212265046
# 4 8 0.9106385671792925
# 4 12 0.937915321815075
# 4 4 0.8343775710398076
# 4 8 0.8979178533004241
# 4 12 0.926650212011898
# 5 4 0.8012850650368281
# 5 8 0.8935903463406989
# 5 12 0.9295826150551115
# 5 4 0.7498824635637047
# 5 8 0.8764561458496578
# 5 12 0.9078514339445228
# 6 4 0.8184984123732459
# 6 8 0.9036156918979822
# 6 12 0.933422103861518
# 6 4 0.8157328689951859
# 6 8 0.8868175765645806
# 6 12 0.9184676841134897
# 8 4 0.7682564812982386
# 8 8 0.8873936275479913
# 8 12 0.9269245992479715
# 8 4 0.7685038590936077
# 8 8 0.8738867999208391
# 8 12 0.9125272115574906
# 9 4 0.8667248654153936
# 9 8 0.9241961297832096
# 9 12 0.9483486105048742
# 9 4 0.8649789029535865
# 9 8 0.9109559144478394
# 9 12 0.9364178670158592
# 10 4 0.8214857142857143
# 10 8 0.9043809523809524
# 10 12 0.9337904761904762
# 10 4 0.8330666666666666
# 10 8 0.8800761904761905
# 10 12 0.9187809523809524
# 11 4 0.9058726673984632
# 11 8 0.9528055974756482
# 11 12 0.9681668496158068
# 11 4 0.8854281009879253
# 11 8 0.9338729592536699
# 11 12 0.9470362239297475
# 12 4 0.8401414278191529
# 12 8 0.9090498758745205
# 12 12 0.94004363198676
# 12 4 0.8037312871436094
# 12 8 0.8848266004664109
# 12 12 0.913337846987136
# 13 4 0.7274733835367437
# 13 8 0.8641911191898208
# 13 12 0.9100666493551458
# 13 4 0.7250497706223492
# 13 8 0.8589976629446897
# 13 12 0.9006318705098243
# 14 4 0.7304871012939915
# 14 8 0.8665622681941811
# 14 12 0.9120580235720761
# 14 4 0.719278002142916
# 14 8 0.8529217835654825
# 14 12 0.8984999587900766
# 15 4 0.8088169996828417
# 15 8 0.8983190612115446
# 15 12 0.9307960672375515
# 15 4 0.7935299714557564
# 15 8 0.8922296225816683
# 15 12 0.9225499524262607
# 0.79566969
# 0.80811034
# 0.87935118
# 0.89891643
# 0.91678546
# 0.93186158
