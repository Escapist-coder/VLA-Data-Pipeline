import numpy as np
from scipy.interpolate import interp1d

class TimeSynchronizer:
    def __init__(self, target_fps=30):
        # 目标频率，通常对齐到相机的频率 (30Hz)
        self.target_fps = target_fps

    def synchronize(self, images, qpos, actions, original_robot_hz=50):
        """
        通用的多模态对齐函数
        images: 视觉数组 [相机帧数, H, W, C]
        qpos: 关节角度数组 [机器人帧数, 关节维数]
        actions: 动作指令数组 [机器人帧数, 动作维数 (包含夹爪)]
        original_robot_hz: 原始机器人数据采集频率
        """
        num_cam_frames = images.shape[0]
        num_robot_frames = qpos.shape[0]

        # 如果帧数天然一致，直接放行（节省算力）
        if num_cam_frames == num_robot_frames:
            return images, qpos, actions

        # 1. 根据帧数反推时间轴
        duration = num_cam_frames / self.target_fps
        cam_timestamps = np.linspace(0, duration, num_cam_frames)
        robot_timestamps = np.linspace(0, duration, num_robot_frames)

        # 2. 分离连续动作(关节)和离散动作(夹爪)
        # 假设 actions 的最后一列是夹爪 (0 或 1)
        continuous_actions = actions[:, :-1]
        gripper_actions = actions[:, -1:]

        # 3. 对本体状态 (qpos) 进行线性插值
        interp_qpos = interp1d(robot_timestamps, qpos, kind='linear', axis=0, fill_value="extrapolate")
        synced_qpos = interp_qpos(cam_timestamps)

        # 4. 对连续动作 进行线性插值
        interp_actions = interp1d(robot_timestamps, continuous_actions, kind='linear', axis=0, fill_value="extrapolate")
        synced_continuous_actions = interp_actions(cam_timestamps)

        # 5. 对夹爪指令 进行最近邻插值 (极其重要，保持 0/1 状态不被破坏)
        interp_gripper = interp1d(robot_timestamps, gripper_actions, kind='nearest', axis=0, fill_value="extrapolate")
        synced_gripper = interp_gripper(cam_timestamps)

        # 重新拼接动作
        synced_actions = np.hstack([synced_continuous_actions, synced_gripper])

        return images, synced_qpos, synced_actions