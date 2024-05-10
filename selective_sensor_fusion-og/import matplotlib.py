import matplotlib.pyplot as plt
import numpy as np

# 读取数据
truth_poses = np.loadtxt('E:/SLAM/results/hard/truth_pose_seq5.txt')
result_poses = np.loadtxt('E:/SLAM/results/hard/result_seq5.txt')
gt_poses = np.loadtxt('E:/SLAM/Dataset/dataset_test/05/poses.txt',skiprows=1)
# 提取x, z坐标
gt_x = gt_poses[:, 3]
gt_z = gt_poses[:, 11]
truth_x = truth_poses[:, 0] + gt_x
truth_z = truth_poses[:, 2] + gt_z
result_x = result_poses[:, 0] + gt_x
result_z = result_poses[:, 2] + gt_z

# 创建图形
plt.figure()

# 绘制真实轨迹
plt.plot(truth_x, truth_z, label='Hard')

# 绘制结果轨迹
plt.plot(result_x, result_z, label='Turth')

# 添加图例
plt.legend()

# 显示图形
plt.show()