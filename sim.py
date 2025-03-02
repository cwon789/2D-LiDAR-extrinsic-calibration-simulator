#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import math
import time
import threading
import tkinter as tk
from tkinter import ttk

import random  # 노이즈 추가용
import matplotlib
matplotlib.use("TkAgg")  # Tkinter와 연동
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def euler_to_quaternion(roll, pitch, yaw):
    """yaw만 중요한 경우 roll=pitch=0, yaw만 사용"""
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) \
        - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) \
        + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) \
        - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) \
        + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return (qx, qy, qz, qw)


class SimpleOdomSimulator(Node):
    def __init__(self):
        super().__init__('simple_odom_simulator')

        # QoS 설정
        self.qos_profile = QoSProfile(depth=1)
        self.qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        self.qos_profile.durability = DurabilityPolicy.VOLATILE

        # 오도메트리 퍼블리셔
        self.odom_pub = self.create_publisher(Odometry, '/odometry', self.qos_profile)

        # 시나리오 상태
        self.running_scenario1 = False
        self.running_scenario2 = False

        # Scenario1(제자리 회전)
        self.x_ext = 0.0
        self.y_ext = 0.0
        self.rot_vel = 0.2
        self.start_time_s1 = None

        # Scenario2(직진, 각도편이)
        self.angle_offset = 0.0
        self.lin_vel = 0.1
        self.start_time_s2 = None

        # ----- 추가: 노이즈 표준편차(필요 시 조절 가능) -----
        self.noise_std_pos = 0.001  # x, y 위치 노이즈 (m 단위)
        self.noise_std_yaw = 0.001  # yaw 노이즈 (rad 단위)

        # 시뮬레이션 궤적 저장
        self.sim_positions = []  # [(x, y, yaw)]

        # 30Hz 타이머
        self.timer = self.create_timer(1.0/30.0, self.publish_odom)

    def start_scenario1(self, x_ext, y_ext):
        self.x_ext = x_ext
        self.y_ext = y_ext
        self.start_time_s1 = time.time()
        self.running_scenario1 = True
        self.running_scenario2 = False
        self.sim_positions.clear()
        self.get_logger().info("Scenario1 (제자리 회전) Start")

    def stop_scenario1(self):
        self.running_scenario1 = False
        self.get_logger().info("Scenario1 (제자리 회전) Stop")

    def start_scenario2(self, angle_offset):
        self.angle_offset = angle_offset
        self.start_time_s2 = time.time()
        self.running_scenario2 = True
        self.running_scenario1 = False
        self.sim_positions.clear()
        self.get_logger().info("Scenario2 (직진) Start")

    def stop_scenario2(self):
        self.running_scenario2 = False
        self.get_logger().info("Scenario2 (직진) Stop")

    def publish_odom(self):
        now = time.time()
        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "lidar"
        odom_msg.header.stamp = self.get_clock().now().to_msg()

        if self.running_scenario1:
            dt = now - self.start_time_s1
            base_link_heading = self.rot_vel * dt
            base_link_x = 0.0
            base_link_y = 0.0

            lidar_x = (math.cos(base_link_heading)*self.x_ext
                       - math.sin(base_link_heading)*self.y_ext
                       + base_link_x)
            lidar_y = (math.sin(base_link_heading)*self.x_ext
                       + math.cos(base_link_heading)*self.y_ext
                       + base_link_y)
            lidar_yaw = base_link_heading

            odom_msg.pose.pose.position.x = lidar_x
            odom_msg.pose.pose.position.y = lidar_y

            q = euler_to_quaternion(0.0, 0.0, lidar_yaw)
            odom_msg.pose.pose.orientation.x = q[0]
            odom_msg.pose.pose.orientation.y = q[1]
            odom_msg.pose.pose.orientation.z = q[2]
            odom_msg.pose.pose.orientation.w = q[3]

            self.sim_positions.append((lidar_x, lidar_y, lidar_yaw))
            self.odom_pub.publish(odom_msg)

        elif self.running_scenario2:
            dt = now - self.start_time_s2
            heading = self.angle_offset  # 고정
            dist = self.lin_vel * dt

            # --- (1) 이상적인 위치/방향 ---
            ideal_x = dist * math.cos(heading)
            ideal_y = dist * math.sin(heading)
            ideal_yaw = heading

            # --- (2) 노이즈 추가 ---
            noisy_x = ideal_x + random.gauss(0, self.noise_std_pos)
            noisy_y = ideal_y + random.gauss(0, self.noise_std_pos)
            noisy_yaw = ideal_yaw + random.gauss(0, self.noise_std_yaw)

            odom_msg.pose.pose.position.x = noisy_x
            odom_msg.pose.pose.position.y = noisy_y
            q = euler_to_quaternion(0.0, 0.0, noisy_yaw)
            odom_msg.pose.pose.orientation.x = q[0]
            odom_msg.pose.pose.orientation.y = q[1]
            odom_msg.pose.pose.orientation.z = q[2]
            odom_msg.pose.pose.orientation.w = q[3]

            # Trajectory 저장(플롯 용도)
            self.sim_positions.append((noisy_x, noisy_y, noisy_yaw))

            self.odom_pub.publish(odom_msg)
        else:
            # Stop 상태이면 퍼블리시 안 함
            pass


class SimpleOdomSimGUI:
    def __init__(self, node):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Simple Odom Simulator")

        frame_top = ttk.Frame(self.root)
        frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Scenario1
        scenario1_frame = ttk.LabelFrame(frame_top, text="Scenario1: 제자리 회전")
        scenario1_frame.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(scenario1_frame, text="x_ext:").grid(row=0, column=0, padx=2, pady=2)
        self.entry_x_ext = ttk.Entry(scenario1_frame, width=6)
        self.entry_x_ext.insert(0, "0.0")
        self.entry_x_ext.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(scenario1_frame, text="y_ext:").grid(row=1, column=0, padx=2, pady=2)
        self.entry_y_ext = ttk.Entry(scenario1_frame, width=6)
        self.entry_y_ext.insert(0, "0.0")
        self.entry_y_ext.grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(scenario1_frame, text="rot_vel(rad/s):").grid(row=2, column=0, padx=2, pady=2)
        self.entry_rot_vel = ttk.Entry(scenario1_frame, width=6)
        self.entry_rot_vel.insert(0, "0.2")
        self.entry_rot_vel.grid(row=2, column=1, padx=2, pady=2)

        btn_start_s1 = ttk.Button(scenario1_frame, text="Start Rotation", command=self.on_start_scenario1)
        btn_start_s1.grid(row=3, column=0, padx=2, pady=2)

        btn_stop_s1 = ttk.Button(scenario1_frame, text="Stop Rotation", command=self.on_stop_scenario1)
        btn_stop_s1.grid(row=3, column=1, padx=2, pady=2)

        # Scenario2
        scenario2_frame = ttk.LabelFrame(frame_top, text="Scenario2: 직진(각도 편이)")
        scenario2_frame.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(scenario2_frame, text="angle_offset(deg):").grid(row=0, column=0, padx=2, pady=2)
        self.entry_angle_offset = ttk.Entry(scenario2_frame, width=6)
        self.entry_angle_offset.insert(0, "0.0")
        self.entry_angle_offset.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(scenario2_frame, text="lin_vel(m/s):").grid(row=1, column=0, padx=2, pady=2)
        self.entry_lin_vel = ttk.Entry(scenario2_frame, width=6)
        self.entry_lin_vel.insert(0, "0.1")
        self.entry_lin_vel.grid(row=1, column=1, padx=2, pady=2)

        btn_start_s2 = ttk.Button(scenario2_frame, text="Start Straight", command=self.on_start_scenario2)
        btn_start_s2.grid(row=2, column=0, padx=2, pady=2)

        btn_stop_s2 = ttk.Button(scenario2_frame, text="Stop Straight", command=self.on_stop_scenario2)
        btn_stop_s2.grid(row=2, column=1, padx=2, pady=2)

        # Matplotlib Figure & Canvas
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Odometry Trajectory")
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # 주기적 플롯 업데이트
        self.update_plot()

    def on_start_scenario1(self):
        x_ext = float(self.entry_x_ext.get())
        y_ext = float(self.entry_y_ext.get())
        rot_vel = float(self.entry_rot_vel.get())
        self.node.rot_vel = rot_vel
        self.node.start_scenario1(x_ext, y_ext)

    def on_stop_scenario1(self):
        self.node.stop_scenario1()

    def on_start_scenario2(self):
        angle_deg = float(self.entry_angle_offset.get())
        angle_rad = math.radians(angle_deg)
        lin_vel = float(self.entry_lin_vel.get())
        self.node.lin_vel = lin_vel
        self.node.start_scenario2(angle_rad)

    def on_stop_scenario2(self):
        self.node.stop_scenario2()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Odometry Trajectory")
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)

        if len(self.node.sim_positions) > 0:
            xs = [p[0] for p in self.node.sim_positions]
            ys = [p[1] for p in self.node.sim_positions]
            self.ax.plot(xs, ys, 'b.-', label="Trajectory")

            # 마지막 위치에 헤딩 화살표 표시
            bx, by, byaw = self.node.sim_positions[-1]
            heading_len = 0.5
            hx = bx + heading_len * math.cos(byaw)
            hy = by + heading_len * math.sin(byaw)
            self.ax.arrow(
                bx, by,
                hx - bx, hy - by,
                head_width=0.05, head_length=0.08,
                fc='r', ec='r'
            )
            self.ax.legend()

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleOdomSimulator()

    gui = SimpleOdomSimGUI(node)

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


if __name__ == "__main__":
    main()
