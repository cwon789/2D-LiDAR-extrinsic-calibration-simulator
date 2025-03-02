#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import math
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# SciPy 최적화
from scipy.optimize import minimize

############################################################
# SE(2) Helper Functions
############################################################
def se2_compose(a, b):
    """
    합성: a, b는 (theta, x, y) 형태
     a ⊕ b = (a_th + b_th,  a_xy + R(a_th)*b_xy)
    """
    (th_a, x_a, y_a) = a
    (th_b, x_b, y_b) = b
    th = th_a + th_b
    cos_a = math.cos(th_a)
    sin_a = math.sin(th_a)
    x = x_a + cos_a*x_b - sin_a*y_b
    y = y_a + sin_a*x_b + cos_a*y_b
    return (th, x, y)

def se2_invert(a):
    """
    역변환: a^-1
    if a = (th, x, y), then a^-1 = (-th, R(-th)*(-x, -y))
    """
    (th, x, y) = a
    th_inv = -th
    cos_th = math.cos(th_inv)
    sin_th = math.sin(th_inv)
    # apply rotation to (-x, -y)
    x_inv = cos_th*(-x) - sin_th*(-y)
    y_inv = sin_th*(-x) + cos_th*(-y)
    return (th_inv, x_inv, y_inv)

def se2_translation(t):
    """
    변환 (theta, x, y)에서 평행이동 (x, y) 부분만 리턴
    """
    return (t[1], t[2])


############################################################
# ROS Node
############################################################
class LidarCalibrationNode(Node):
    def __init__(self):
        super().__init__('lidar_calibration_node')

        self.qos_profile_sensor_data = QoSProfile(depth=1)
        self.qos_profile_sensor_data.reliability = ReliabilityPolicy.BEST_EFFORT
        self.qos_profile_sensor_data.durability = DurabilityPolicy.VOLATILE

        # for absolute odom
        self.prev_pose = None
        self.odom_poses = []   # list of (theta, x, y) -> LiDAR 누적 pose
        self.delta_odom = []   # list of increments Δ_i in LiDAR frame

        # 현재 Extrinsic
        self.theta_ext = 0.0
        self.x_ext = 0.0
        self.y_ext = 0.0

        self.sub_odom = None

    def subscribe_odometry(self):
        if self.sub_odom is None:
            self.sub_odom = self.create_subscription(
                Odometry,
                '/odometry',
                self.odom_callback,
                self.qos_profile_sensor_data
            )
            self.get_logger().info("Subscribed to /odometry")

    def odom_callback(self, msg):
        # 절대 pose (x, y, yaw)
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        # quat -> yaw
        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w
        siny = 2.0 * (ow * oz + ox * oy)
        cosy = 1.0 - 2.0 * (oy * oy + oz * oz)
        yaw = math.atan2(siny, cosy)

        current_pose = (yaw, px, py)

        if self.prev_pose is None:
            # 첫 번째 pose
            self.prev_pose = current_pose
            self.odom_poses = [current_pose]
            return

        # 이전 포즈와의 증분 (LiDAR 좌표계 상)
        delta = self.get_increment(self.prev_pose, current_pose)
        self.delta_odom.append(delta)

        # 누적 포즈 갱신
        last_accum = self.odom_poses[-1]
        new_accum = se2_compose(last_accum, delta)
        self.odom_poses.append(new_accum)

        self.prev_pose = current_pose

    def get_increment(self, poseA, poseB):
        """
        poseA->poseB로 가는 증분 transform (Δ = A^-1 ⊕ B)
        """
        invA = se2_invert(poseA)
        delta = se2_compose(invA, poseB)
        return delta

    ############################################################
    # cost: sum of translations of T⊕Δ_i⊕T^-1
    ############################################################
    def cost_extrinsic(self, theta, tx, ty):
        T = (theta, tx, ty)
        sse = 0.0
        for d in self.delta_odom:
            # T ⊕ d ⊕ T^-1
            Td = se2_compose(T, d)
            TdT = se2_compose(Td, se2_invert(T))
            # translation만 추출
            (dx, dy) = se2_translation(TdT)
            # ideally (dx, dy) = (0, 0)
            sse += (dx*dx + dy*dy)
        return sse

    def optimize_extrinsic_once(self, init_th, init_x, init_y):
        """
        SciPy minimize로 한 번 최적화
        """
        if len(self.delta_odom) < 1:
            return (init_th, init_x, init_y, 9999.9)

        def cost_fn(params):
            th, xx, yy = params
            return self.cost_extrinsic(th, xx, yy)

        x0 = [init_th, init_x, init_y]
        res = minimize(cost_fn, x0, method='BFGS')
        (th_opt, x_opt, y_opt) = res.x
        cost_val = res.fun
        # theta를 0~2π 범위로 정규화
        th_opt = (th_opt + 2*math.pi) % (2*math.pi)
        return (th_opt, x_opt, y_opt, cost_val)

    def calibrate_translation_step(self, alpha):
        """
        (1) 현재 extrinsic = (theta_ext, x_ext, y_ext)를 초기값으로 최적화
        (2) 나온 결과와 alpha만큼 보간
        (3) 갱신 후 오차(평균 거리) 반환
        """
        (th_star, x_star, y_star, cost_val) = self.optimize_extrinsic_once(
            self.theta_ext, self.x_ext, self.y_ext
        )

        old_th = self.theta_ext
        old_x = self.x_ext
        old_y = self.y_ext

        # theta 보간 (수치적 단순 보간)
        new_th = old_th + alpha * (th_star - old_th)
        new_th = (new_th + 2*math.pi) % (2*math.pi)

        new_x = old_x + alpha * (x_star - old_x)
        new_y = old_y + alpha * (y_star - old_y)

        self.theta_ext = new_th
        self.x_ext = new_x
        self.y_ext = new_y

        # cost_val은 sum of squared distances
        # 평균 오차(RMS) 계산
        N = len(self.delta_odom)
        if N > 0:
            mean_sq = cost_val / N
            mean_dist = math.sqrt(mean_sq)  # RMS
        else:
            mean_dist = 9999.9

        return mean_dist

    ############################################################
    # 시각화를 위해 "누적 포즈"를 extrinsic으로 base_link에 변환
    ############################################################
    def get_transformed_points(self):
        """
        self.odom_poses: LiDAR 누적포즈들
        T = (theta_ext, x_ext, y_ext)

        base_link_pose_i = T ⊕ p_i ⊕ T^-1
        (그중 (x,y)만 반환)
        """
        result = []
        T = (self.theta_ext, self.x_ext, self.y_ext)
        T_inv = se2_invert(T)
        for p in self.odom_poses:
            Tp = se2_compose(T, p)
            basePose = se2_compose(Tp, T_inv)
            (bx, by) = se2_translation(basePose)
            result.append((bx, by))
        return result


############################################################
# Tkinter GUI
############################################################
class CalibrationGUI:
    def __init__(self, node):
        self.node = node

        # GUI 메인 윈도우
        self.root = tk.Tk()
        self.root.title("Incremental Delta-based Calibration")

        # 상태
        self.calibrating_translation = False
        self.trans_error = 0.0

        # extrinsic 시간 변화 기록 (그래프용)
        self.t_data = []
        self.x_ext_data = []
        self.y_ext_data = []
        self.plot_time = 0.0  # 0.1초 간격으로 누적

        # -----------------------------
        # (1) 상단 프레임: 구독, Threshold, LR
        # -----------------------------
        frame_top = ttk.Frame(self.root)
        frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 구독 버튼
        btn_sub = ttk.Button(frame_top, text="Subscribe Odom", command=self.subscribe_odom)
        btn_sub.pack(side=tk.LEFT, padx=5)

        self.label_odom_status = ttk.Label(frame_top, text="Not Subscribed", foreground="red")
        self.label_odom_status.pack(side=tk.LEFT, padx=5)

        # Threshold & LR
        ttk.Label(frame_top, text="Trans Thr:").pack(side=tk.LEFT, padx=2)
        self.entry_trans_threshold = ttk.Entry(frame_top, width=6)
        self.entry_trans_threshold.insert(0, "0.002")
        self.entry_trans_threshold.pack(side=tk.LEFT, padx=2)

        ttk.Label(frame_top, text="LR:").pack(side=tk.LEFT, padx=2)
        self.entry_trans_lr = ttk.Entry(frame_top, width=6)
        self.entry_trans_lr.insert(0, "0.01")
        self.entry_trans_lr.pack(side=tk.LEFT, padx=2)

        btn_trans_cal = ttk.Button(frame_top, text="Start Trans Cal", command=self.start_translation_cal)
        btn_trans_cal.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (2) Extrinsic 표시
        # -----------------------------
        frame_ext = ttk.Frame(self.root)
        frame_ext.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.label_xy_ext = ttk.Label(frame_ext, text="x_ext=0.0, y_ext=0.0")
        self.label_xy_ext.pack(side=tk.LEFT, padx=5)

        self.label_theta_ext = ttk.Label(frame_ext, text="theta_ext=0.0°")
        self.label_theta_ext.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (3) 에러 표시
        # -----------------------------
        frame_err = ttk.Frame(self.root)
        frame_err.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.label_trans_error = ttk.Label(frame_err, text="Trans Error=0.0")
        self.label_trans_error.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (4) Matplotlib Figure
        # -----------------------------
        # 1행 3열: (Path) - (x_ext vs Time) - (y_ext vs Time)
        self.fig = plt.Figure(figsize=(12, 4), dpi=100)

        # (1,1) Base Link 경로
        self.ax_path = self.fig.add_subplot(1, 3, 1)
        self.ax_path.set_aspect('equal', 'box')
        self.ax_path.set_title("Base Link Trajectory")
        self.ax_path.grid(True)

        # (1,2) x_ext vs Time
        self.ax_x_ext = self.fig.add_subplot(1, 3, 2)
        self.ax_x_ext.set_title("x_ext vs. Time")
        self.ax_x_ext.set_xlabel("Time [s]")
        self.ax_x_ext.set_ylabel("x_ext [m]")
        self.ax_x_ext.grid(True)

        # (1,3) y_ext vs Time
        self.ax_y_ext = self.fig.add_subplot(1, 3, 3)
        self.ax_y_ext.set_title("y_ext vs. Time")
        self.ax_y_ext.set_xlabel("Time [s]")
        self.ax_y_ext.set_ylabel("y_ext [m]")
        self.ax_y_ext.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 주기적 업데이트
        self.update_plot()

    ############################################################
    # 버튼 콜백
    ############################################################
    def subscribe_odom(self):
        self.node.subscribe_odometry()
        self.label_odom_status.config(text="Subscribed", foreground="green")

    def start_translation_cal(self):
        self.calibrating_translation = True
        self.trans_error = 9999.9

    ############################################################
    # 주기적(100ms) 업데이트
    ############################################################
    def update_plot(self):
        # (1) 캘리브레이션 진행
        if self.calibrating_translation:
            thr = float(self.entry_trans_threshold.get())
            alpha = float(self.entry_trans_lr.get())
            err = self.node.calibrate_translation_step(alpha)
            self.trans_error = err
            # 예: 오차가 threshold 이하일 때 중단
            # if err < thr:
            #     self.calibrating_translation = False

        # (2) 시간축 데이터 기록
        self.plot_time += 0.1  # 100ms마다 0.1초 증가
        self.t_data.append(self.plot_time)
        self.x_ext_data.append(self.node.x_ext)
        self.y_ext_data.append(self.node.y_ext)

        # (3) 플롯 업데이트
        # 3-1) Path Plot
        self.ax_path.clear()
        self.ax_path.set_title("Base Link Trajectory")
        self.ax_path.set_aspect('equal', 'box')
        self.ax_path.grid(True)

        pts = self.node.get_transformed_points()
        if len(pts) > 0:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.ax_path.plot(xs, ys, 'b.-', label="Trajectory")
            self.ax_path.legend()

        # 3-2) x_ext vs Time
        self.ax_x_ext.clear()
        self.ax_x_ext.set_title("x_ext vs. Time")
        self.ax_x_ext.set_xlabel("Time [s]")
        self.ax_x_ext.set_ylabel("x_ext [m]")
        self.ax_x_ext.grid(True)
        self.ax_x_ext.plot(self.t_data, self.x_ext_data, 'r.-')

        # 3-3) y_ext vs Time
        self.ax_y_ext.clear()
        self.ax_y_ext.set_title("y_ext vs. Time")
        self.ax_y_ext.set_xlabel("Time [s]")
        self.ax_y_ext.set_ylabel("y_ext [m]")
        self.ax_y_ext.grid(True)
        self.ax_y_ext.plot(self.t_data, self.y_ext_data, 'g.-')

        self.canvas.draw()

        # (4) 라벨 업데이트
        self.label_xy_ext.config(
            text=f"x_ext={self.node.x_ext:.4f}, y_ext={self.node.y_ext:.4f}"
        )
        deg = math.degrees(self.node.theta_ext)
        deg_signed = (deg + 180.0) % 360.0 - 180.0
        self.label_theta_ext.config(text=f"theta_ext={deg_signed:.2f}°")
        self.label_trans_error.config(text=f"Trans Error={self.trans_error:.4f}")

        # (5) 100ms 후 재호출
        self.root.after(100, self.update_plot)

    def run(self):
        self.root.mainloop()


############################################################
# main
############################################################
def main(args=None):
    rclpy.init(args=args)
    node = LidarCalibrationNode()

    gui = CalibrationGUI(node)

    def ros_spin():
        rclpy.spin(node)

    t = threading.Thread(target=ros_spin, daemon=True)
    t.start()

    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
