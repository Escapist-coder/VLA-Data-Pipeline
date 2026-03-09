import h5py
import numpy as np
import os

def create_fake_aloha_hdf5(filename, num_frames, qpos_data, action_data):
    """按照真实的 ALOHA 格式生成 HDF5 文件"""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('action', data=action_data)
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos_data)
        # 顺便塞点假图像进去充数
        obs.create_dataset('images/top', data=np.zeros((num_frames, 224, 224, 3), dtype=np.uint8))
    print(f"📦 生成测试文件: {filename} (帧数: {num_frames})")

# 准备测试目录
os.makedirs("test_data", exist_ok=True)

# ---------------------------------------------------------
# Case 1: 完美的正常轨迹 (应该 ✅ 质检通过)
# ---------------------------------------------------------
frames = 150
# 模拟机械臂平滑移动：利用 np.linspace 生成渐变的数据
qpos_good = np.linspace(0, 1.5, frames).reshape(-1, 1) * np.ones((1, 14))
action_good = np.copy(qpos_good) 
create_fake_aloha_hdf5("test_data/case1_perfect.hdf5", frames, qpos_good, action_good)

# ---------------------------------------------------------
# Case 2: 轨迹过短，刚启动就断开 (应该 ❌ 质检失败)
# ---------------------------------------------------------
frames_short = 10
qpos_short = np.zeros((frames_short, 14))
action_short = np.zeros((frames_short, 14))
create_fake_aloha_hdf5("test_data/case2_too_short.hdf5", frames_short, qpos_short, action_short)

# ---------------------------------------------------------
# Case 3: 原地发呆，没有产生位移 (应该 ❌ 质检失败)
# ---------------------------------------------------------
frames_lazy = 100
# 数据全是 0，代表机械臂根本没动
qpos_lazy = np.zeros((frames_lazy, 14)) 
action_lazy = np.zeros((frames_lazy, 14))
create_fake_aloha_hdf5("test_data/case3_lazy.hdf5", frames_lazy, qpos_lazy, action_lazy)

# ---------------------------------------------------------
# Case 4: 传感器跳变，速度爆炸 (应该 ❌ 质检失败)
# ---------------------------------------------------------
frames_spike = 100
qpos_spike = np.linspace(0, 0.5, frames_spike).reshape(-1, 1) * np.ones((1, 14))
# 💥 投毒：在第 50 帧，某个关节的数值突然暴增！
qpos_spike[50, 3] = 10.0 
action_spike = np.copy(qpos_spike)
create_fake_aloha_hdf5("test_data/case4_spike.hdf5", frames_spike, qpos_spike, action_spike)

print("\n🎯 4个极限测试用例已生成在 test_data/ 目录下！")