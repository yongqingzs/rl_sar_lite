/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_GO1_HPP
#define RL_REAL_GO1_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_go1.hpp"

#include "hardware_interface_base.hpp"
#include "hardware_interface_free_dog_sdk.hpp"
#ifdef UNITREE_ROS2_AVAILABLE
#include "hardware_interface_unitree_ros2.hpp"
#endif

#include <csignal>
#include <chrono>

#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#endif

#if defined(CONTROL_EXTRA) && defined(USE_ROS)
#include <control_input_msgs/msg/inputs.hpp>
#endif

#ifdef PLOT
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

#if defined(USE_ROS2) && defined(USE_ROS)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_udpSend;
    std::shared_ptr<LoopFunc> loop_udpRecv;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // Hardware interface (supports free_dog_sdk and unitree_ros2)
    enum class HardwareProtocol {
        FREE_DOG_SDK,
        UNITREE_ROS2
    };
    
    HardwareProtocol hardware_protocol_;
    std::unique_ptr<HardwareInterfaceBase> hardware_interface_;
    
    void InitializeHardwareInterface();
    void UDPSend();  // For backward compatibility
    void UDPRecv();  // For backward compatibility

    // print timing
    std::chrono::steady_clock::time_point last_print_time;

    // others
    std::vector<float> mapped_joint_positions;
    std::vector<float> mapped_joint_velocities;

    // debug print
    void PrintDebugInfo(const RobotCommand<float> *command);

#if defined(USE_ROS1) && defined(USE_ROS)
    geometry_msgs::Twist cmd_vel;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
#elif defined(USE_ROS2) && defined(USE_ROS)
    // realtime buffer for cmd_vel
    realtime_tools::RealtimeBuffer<geometry_msgs::msg::Twist> buffer_;
    int twist_count = 0;

    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
#endif

#if defined(CONTROL_EXTRA) && defined(USE_ROS)
    control_input_msgs::msg::Inputs control_input;
    rclcpp::Subscription<control_input_msgs::msg::Inputs>::SharedPtr control_input_subscriber;
    void ControlInputsCallback(const control_input_msgs::msg::Inputs::SharedPtr msg);
#endif
};

#endif // RL_REAL_GO1_HPP
