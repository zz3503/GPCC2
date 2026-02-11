# coding=utf-8
from PointCloudStitching import *
setObjectId(1)
n = 18
joints = []
# for i in range(n):
joints.append(np.array([-70, 7, -75, -19, 87, -54]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-50, 0, -65, -33, 78, -33]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-100, 2, -71, -14, 105, -83]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-76, -28, -27, -45, 95, -62]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-39, 37, -97, -22, 71, -24]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-70, 7, -75, -19, 87, 90 - 54]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-50, 0, -65, -33, 78, 90 - 33]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-100, 2, -71, -14, 105, 90 - 83]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-76, -28, -27, -45, 95, 90 - 62]) + np.random.uniform(-1, 1, size=6))
joints.append(np.array([-70, 0, -66, -25, 90, 0]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, 0, -66, -25, 90, 90]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, 0, -66, -25, 90, -90]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, 0, -95, 0, 90, 0]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, 0, -95, 0, 90, 90]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, 0, -95, 0, 90, -90]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([-70, -17, -28, -50, 90, 0]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([95, 30, 31, 43, -103, 90]) + np.random.uniform(-5, 5, size=6))
joints.append(np.array([110, -47, 129, -23, -115, -90]) + np.random.uniform(-5, 5, size=6))
joints = np.array(joints)
print(joints)

T_base_tool_s = []
pcds = []
for i in range(n):
    moveJ(joints[i])
    T_base_tool_s.append(getCurrentWaypoint())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(getPointClouds(1))
    pcds.append(pcd)
print(np.sort([np.mean(pcds[i].compute_nearest_neighbor_distance()) for i in range(n)]))

T_base_tool_s = np.array(T_base_tool_s)
print(f"T_base_tool_s=\n{T_base_tool_s}")
p_cam_s = []
citems = (CPoseAndCloud * n)()
for i in range(n):
    pcd = pcds[i]
    pcd.scale(1000., np.array([0, 0, 0]))
    p_cam_s.append(pcd)  # 防止释放内存
    point = np.asarray(pcd.points, dtype=np.float64)
    citems[i] = CPoseAndCloud(
        rx=T_base_tool_s[i][0], ry=T_base_tool_s[i][1], rz=T_base_tool_s[i][2],  # 单位:弧度,顺规:zyx
        x=T_base_tool_s[i][3], y=T_base_tool_s[i][4], z=T_base_tool_s[i][5],  # 单位:毫米
        cloudLen=len(point),  # 数量:1w左右
        cloud=point.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))  # 单位:毫米

if True:
    dll.setHE(-23.226367168247968, 24.061234563984087, 50.06320345011153, -0.00018035573675309984, -0.00013002876111078876, -1.5732829669179795)

cdata = CVectorPoseAndCloud(num=n, items=citems)
reslut = dll.setInputData(ctypes.byref(cdata))
print(f'reslut = {reslut}')
n1, n2, n3, n4, n5, n6 = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
dll.getHE(ctypes.byref(n1), ctypes.byref(n2), ctypes.byref(n3), ctypes.byref(n4), ctypes.byref(n5), ctypes.byref(n6))
T_tool_cam = sp.SE3.exp(np.array([n1.value, n2.value, n3.value, n4.value, n5.value, n6.value])).matrix()
print([n1.value, n2.value, n3.value, n4.value, n5.value, n6.value])
print(f"T_tool_cam=\n{T_tool_cam}")
