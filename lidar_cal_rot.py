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
matplotlib.use("TkAgg")  # Tkinter 연동
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

############################################################
# ROS2 Node
############################################################
class LidarCalibrationNode(Node):
    def __init__(self):
        super().__init__('lidar_calibration_node')

        # QoS 설정: sensor_data
        self.qos_profile_sensor_data = QoSProfile(depth=1)
        self.qos_profile_sensor_data.reliability = ReliabilityPolicy.BEST_EFFORT
        self.qos_profile_sensor_data.durability = DurabilityPolicy.VOLATILE

        self.sub_odom = None
        self.odometry_data = []  # (x, y, yaw) 리스트

        # Extrinsic 파라미터 (라디안)
        self.x_ext = 0.0
        self.y_ext = 0.0
        self.theta_ext = 0.0  # 라디안

    def subscribe_odometry(self):
        """ /odometry 구독 시작 """
        if self.sub_odom is None:
            self.sub_odom = self.create_subscription(
                Odometry,
                '/odometry',
                self.odom_callback,
                self.qos_profile_sensor_data
            )
            self.get_logger().info("Subscribed to /odometry with sensor_data QoS")

    def odom_callback(self, msg):
        """Odometry 콜백: (x, y, yaw) 2D 추출"""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w

        # 쿼터니언 -> yaw (2D)
        siny_cosp = 2.0 * (ow * oz + ox * oy)
        cosy_cosp = 1.0 - 2.0 * (oy * oy + oz * oz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.odometry_data.append((px, py, yaw))

    ############################################################
    # 한 번의 캘리브레이션 스텝
    # (반복 호출하여 오차 < threshold 되면 종료)
    ############################################################
    def calibrate_translation_step(self, alpha):
        """
        제자리 회전 구간 사용 가정:
         - (x,y)의 평균값을 0으로 수렴시키는 간단 예시
         - alpha(학습률)
         - return: 현재 스텝의 에러
        """
        if len(self.odometry_data) < 2:
            return 9999.9

        xs_base = []
        ys_base = []
        for (lx, ly, _) in self.odometry_data:
            # 라이다 odom -> base_link
            bx = math.cos(self.theta_ext) * lx - math.sin(self.theta_ext) * ly + self.x_ext
            by = math.sin(self.theta_ext) * lx + math.cos(self.theta_ext) * ly + self.y_ext
            xs_base.append(bx)
            ys_base.append(by)

        mean_x = np.mean(xs_base)
        mean_y = np.mean(ys_base)

        # 오차: (mean_x, mean_y) => (0,0)에서의 거리
        error = math.sqrt(mean_x**2 + mean_y**2)

        # 업데이트
        self.x_ext -= alpha * mean_x
        self.y_ext -= alpha * mean_y

        return error

    def calibrate_theta_step(self, alpha):
        """
        직진 구간 사용 가정:
         - y 편차가 0이 되어야 한다는 단순 예시
         - alpha(학습률)
         - return: 현재 스텝의 에러
        """
        if len(self.odometry_data) < 2:
            return 9999.9

        xs_base = []
        ys_base = []
        for (lx, ly, _) in self.odometry_data:
            bx = math.cos(self.theta_ext)*lx - math.sin(self.theta_ext)*ly + self.x_ext
            by = math.sin(self.theta_ext)*lx + math.cos(self.theta_ext)*ly + self.y_ext
            xs_base.append(bx)
            ys_base.append(by)

        mean_y = np.mean(ys_base)
        # 오차 = y 편차
        error = abs(mean_y)

        # theta 업데이트
        self.theta_ext -= alpha * mean_y
        # 0 ~ 2π 범위로 정규화
        self.theta_ext = (self.theta_ext + math.pi*2) % (math.pi*2)

        return error

    ############################################################
    # 시각화용: 변환된 점들
    ############################################################
    def get_transformed_points(self):
        """
        현재 (x_ext, y_ext, theta_ext)로 라이다 odom 좌표 -> base_link 좌표로 변환
        """
        result = []
        for (lx, ly, _) in self.odometry_data:
            bx = math.cos(self.theta_ext)*lx - math.sin(self.theta_ext)*ly + self.x_ext
            by = math.sin(self.theta_ext)*lx + math.cos(self.theta_ext)*ly + self.y_ext
            result.append((bx, by))
        return result

############################################################
# Tkinter GUI
############################################################
class CalibrationGUI:
    def __init__(self, node):
        self.node = node

        # 캘리브레이션 상태
        self.calibrating_translation = False
        self.calibrating_theta = False
        self.trans_error = 0.0
        self.theta_error = 0.0

        # 메인 윈도우
        self.root = tk.Tk()
        self.root.title("LiDAR Extrinsic Calibration")

        # -----------------------------
        # (1) 상단 Frame: 구독, Threshold, LR
        # -----------------------------
        frame_top = ttk.Frame(self.root)
        frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Odometry 구독 버튼
        btn_sub = ttk.Button(frame_top, text="Subscribe Lidar Odometry", command=self.subscribe_odometry)
        btn_sub.pack(side=tk.LEFT, padx=5)

        self.label_odom_status = ttk.Label(frame_top, text="Not Subscribed", foreground="red")
        self.label_odom_status.pack(side=tk.LEFT, padx=5)

        # Translation Threshold & LR
        ttk.Label(frame_top, text="Trans Thr:").pack(side=tk.LEFT, padx=2)
        self.entry_trans_threshold = ttk.Entry(frame_top, width=6)
        self.entry_trans_threshold.insert(0, "0.01")
        self.entry_trans_threshold.pack(side=tk.LEFT, padx=2)

        ttk.Label(frame_top, text="Trans LR:").pack(side=tk.LEFT, padx=2)
        self.entry_trans_lr = ttk.Entry(frame_top, width=6)
        self.entry_trans_lr.insert(0, "0.5")
        self.entry_trans_lr.pack(side=tk.LEFT, padx=2)

        btn_trans_cal = ttk.Button(frame_top, text="Start Trans Cal", command=self.start_translation_cal)
        btn_trans_cal.pack(side=tk.LEFT, padx=5)

        # Theta Threshold & LR
        ttk.Label(frame_top, text="Theta Thr:").pack(side=tk.LEFT, padx=2)
        self.entry_theta_threshold = ttk.Entry(frame_top, width=6)
        self.entry_theta_threshold.insert(0, "0.001")
        self.entry_theta_threshold.pack(side=tk.LEFT, padx=2)

        ttk.Label(frame_top, text="Theta LR:").pack(side=tk.LEFT, padx=2)
        self.entry_theta_lr = ttk.Entry(frame_top, width=6)
        self.entry_theta_lr.insert(0, "0.001")
        self.entry_theta_lr.pack(side=tk.LEFT, padx=2)

        btn_theta_cal = ttk.Button(frame_top, text="Start Theta Cal", command=self.start_theta_cal)
        btn_theta_cal.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (2) Extrinsic 표시 라벨
        # -----------------------------
        frame_ext = ttk.Frame(self.root)
        frame_ext.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.label_xy_ext = ttk.Label(frame_ext, text="x_ext: 0.0, y_ext: 0.0")
        self.label_xy_ext.pack(side=tk.LEFT, padx=5)

        # theta_ext(deg) & 방향 표시
        self.label_theta_ext = ttk.Label(frame_ext, text="theta_ext: 0.0° (direction)")
        self.label_theta_ext.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (3) 에러 표시
        # -----------------------------
        frame_err = ttk.Frame(self.root)
        frame_err.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.label_trans_error = ttk.Label(frame_err, text="Trans Error: 0.0")
        self.label_trans_error.pack(side=tk.LEFT, padx=5)

        self.label_theta_error = ttk.Label(frame_err, text="Theta Error: 0.0")
        self.label_theta_error.pack(side=tk.LEFT, padx=5)

        # -----------------------------
        # (4) Matplotlib Figure
        # -----------------------------
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)

        # 왼쪽 - Translation
        self.ax_translation = self.fig.add_subplot(1, 2, 1)
        self.ax_translation.set_title("Translation Calibration")
        self.ax_translation.set_aspect('equal', 'box')
        self.ax_translation.grid(True)

        # 오른쪽 - Theta
        self.ax_theta = self.fig.add_subplot(1, 2, 2)
        self.ax_theta.set_title("Theta Calibration")
        self.ax_theta.set_aspect('equal', 'box')
        self.ax_theta.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 주기적 업데이트
        self.update_plot()

    ############################################################
    # 버튼 콜백
    ############################################################
    def subscribe_odometry(self):
        self.node.subscribe_odometry()
        self.label_odom_status.config(text="Subscribed", foreground="green")

    def start_translation_cal(self):
        self.calibrating_translation = True
        self.calibrating_theta = False
        self.trans_error = 9999.9

    def start_theta_cal(self):
        self.calibrating_theta = True
        self.calibrating_translation = False
        self.theta_error = 9999.9

    ############################################################
    # 주기적(100ms) 업데이트
    ############################################################
    def update_plot(self):
        # (1) 캘리브레이션 반복
        if self.calibrating_translation:
            thr = float(self.entry_trans_threshold.get())
            lr = float(self.entry_trans_lr.get())
            err = self.node.calibrate_translation_step(lr)
            self.trans_error = err
            if err < thr:
                self.calibrating_translation = False

        if self.calibrating_theta:
            thr = float(self.entry_theta_threshold.get())
            lr = float(self.entry_theta_lr.get())
            err = self.node.calibrate_theta_step(lr)
            self.theta_error = err
            if err < thr:
                self.calibrating_theta = False

        # (2) 플롯 갱신
        self.ax_translation.clear()
        self.ax_translation.set_title("Translation Calibration")
        self.ax_translation.set_aspect('equal', 'box')
        self.ax_translation.grid(True)

        self.ax_theta.clear()
        self.ax_theta.set_title("Theta Calibration")
        self.ax_theta.set_aspect('equal', 'box')
        self.ax_theta.grid(True)

        points = self.node.get_transformed_points()
        if len(points) > 0:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # 왼쪽 Plot
            self.ax_translation.plot(xs, ys, 'b.-', label="Trajectory")
            if len(points) > 1:
                bx, by = points[-1]
                last_yaw = self.node.odometry_data[-1][2] + self.node.theta_ext
                hx = bx + 0.5 * math.cos(last_yaw)
                hy = by + 0.5 * math.sin(last_yaw)
                self.ax_translation.arrow(bx, by, hx-bx, hy-by,
                                          head_width=0.05, head_length=0.1, fc='r', ec='r')
            self.ax_translation.legend()

            # 오른쪽 Plot
            self.ax_theta.plot(xs, ys, 'g.-', label="Trajectory")
            if len(points) > 1:
                bx, by = points[-1]
                last_yaw = self.node.odometry_data[-1][2] + self.node.theta_ext
                hx = bx + 0.5 * math.cos(last_yaw)
                hy = by + 0.5 * math.sin(last_yaw)
                self.ax_theta.arrow(bx, by, hx-bx, hy-by,
                                    head_width=0.05, head_length=0.1, fc='r', ec='r')
            self.ax_theta.legend()

        self.canvas.draw()

        # (3) 라벨 업데이트
        # 3-1) x_ext, y_ext
        self.label_xy_ext.config(
            text=f"x_ext: {self.node.x_ext:.4f}, y_ext: {self.node.y_ext:.4f}"
        )

        # 3-2) theta_ext (라디안 -> -180~+180도)
        theta_deg = math.degrees(self.node.theta_ext)  # 0~360 범위일 수도
        theta_deg_signed = (theta_deg + 180.0) % 360.0 - 180.0  # -180 ~ +180
        if abs(theta_deg_signed) < 1e-2:
            # 거의 0도
            direction_str = "Aligned"
        elif theta_deg_signed > 0.0:
            direction_str = f"Left {theta_deg_signed:.2f}°"
        else:
            direction_str = f"Right {abs(theta_deg_signed):.2f}°"

        self.label_theta_ext.config(
            text=f"theta_ext: {theta_deg_signed:.2f}° → {direction_str}"
        )

        # 3-3) 오차 표시
        self.label_trans_error.config(text=f"Trans Error: {self.trans_error:.4f}")
        self.label_theta_error.config(text=f"Theta Error: {self.theta_error:.4f}")

        # (4) 100ms 뒤 다시
        self.root.after(100, self.update_plot)

    def run(self):
        self.root.mainloop()


############################################################
# 메인
############################################################
def main(args=None):
    rclpy.init(args=args)
    node = LidarCalibrationNode()

    gui = CalibrationGUI(node)

    def ros_spin():
        rclpy.spin(node)

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
