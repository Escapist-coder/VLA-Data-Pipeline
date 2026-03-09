import numpy as np
from scipy.interpolate import interp1d

def synchronize_multimodal_data():
    print("⏳ 开始多模态时间戳对齐模拟...")

    # ==========================================
    # 1. 制造一组极度不对齐的原始数据
    # ==========================================
    duration = 1.0 # 录制了 1 秒的轨迹
    cam_fps = 30   # 相机 30Hz
    robot_hz = 50  # 机器人 50Hz

    # 生成时间戳 (相机有30个时间点，机器人有50个时间点)
    cam_timestamps = np.linspace(0, duration, int(duration * cam_fps))
    robot_timestamps = np.linspace(0, duration, int(duration * robot_hz))

    # 模拟机械臂的连续动作：7个关节都在做正弦波运动
    robot_qpos = np.sin(robot_timestamps * 2 * np.pi).reshape(-1, 1) * np.ones((1, 7))

    # 模拟夹爪的离散动作：前0.5秒张开(0)，后0.5秒闭合(1)
    robot_gripper = np.where(robot_timestamps < 0.5, 0.0, 1.0).reshape(-1, 1)

    print(f"🚨 对齐前灾难现场：")
    print(f"   📷 相机帧数: {len(cam_timestamps)} 帧")
    print(f"   🦾 关节帧数: {len(robot_qpos)} 帧")
    print(f"   🤏 夹爪帧数: {len(robot_gripper)} 帧")

    # ==========================================
    # 2. 核心算法：SciPy 时间轴重采样与插值
    # ==========================================
    
    # 针对连续物理量 (关节角 qpos)：使用【线性插值 linear】
    # 原理：如果在 0.1秒和0.2秒之间有个相机帧，那就取两帧动作的平均渐变值
    interp_func_qpos = interp1d(
        robot_timestamps, 
        robot_qpos, 
        kind='linear', 
        axis=0, 
        fill_value="extrapolate" # 如果超出边界就外推，防止报错
    )
    synced_qpos = interp_func_qpos(cam_timestamps)

    # 针对离散指令 (夹爪 gripper)：使用【最近邻插值 nearest】
    # 面试加分项：绝对不能对夹爪用线性插值！否则 0 和 1 之间会插出 0.5 的半开状态，导致机械臂硬件报错！
    interp_func_gripper = interp1d(
        robot_timestamps, 
        robot_gripper, 
        kind='nearest', 
        axis=0, 
        fill_value="extrapolate"
    )
    synced_gripper = interp_func_gripper(cam_timestamps)

    # ==========================================
    # 3. 验证对齐结果
    # ==========================================
    print(f"\n✅ 对齐后完美状态：(以相机时间轴为准)")
    print(f"   📷 相机帧数: {len(cam_timestamps)} 帧")
    print(f"   🦾 同步后关节: {len(synced_qpos)} 帧")
    print(f"   🤏 同步后夹爪: {len(synced_gripper)} 帧")
    
    # 拼接成最终送入 VLA 模型的数据集格式 [30, 8]
    final_robot_state = np.hstack([synced_qpos, synced_gripper])
    print(f"\n🎉 最终拼接完成的机器人状态矩阵形状: {final_robot_state.shape}")

if __name__ == "__main__":
    synchronize_multimodal_data()