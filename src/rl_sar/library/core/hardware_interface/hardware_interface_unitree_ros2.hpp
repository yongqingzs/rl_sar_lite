/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_UNITREE_ROS2_HPP
#define HARDWARE_INTERFACE_UNITREE_ROS2_HPP

#include "hardware_interface_base.hpp"
#include <rclcpp/rclcpp.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <mutex>
#include <atomic>

/**
 * Hardware interface implementation using unitree_ros2 messages
 * This adapter communicates via ROS2 topics with the DDS-ROS2 bridge
 */
class HardwareInterfaceUnitreeRos2 : public HardwareInterfaceBase
{
public:
    explicit HardwareInterfaceUnitreeRos2(rclcpp::Node::SharedPtr node);
    ~HardwareInterfaceUnitreeRos2() override;

    bool Initialize() override;
    void Start() override;
    void Stop() override;

    bool GetState(
        std::vector<float>& joint_positions,
        std::vector<float>& joint_velocities,
        std::vector<float>& joint_efforts,
        std::vector<float>& imu_quaternion,
        std::vector<float>& imu_gyroscope,
        std::vector<float>& imu_accelerometer) override;

    void SetCommand(
        const std::vector<float>& joint_positions,
        const std::vector<float>& joint_velocities,
        const std::vector<float>& joint_torques,
        const std::vector<float>& joint_kp,
        const std::vector<float>& joint_kd) override;

    bool IsReady() const override;

private:
    void lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg);

    rclcpp::Node::SharedPtr node_;
    
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_pub_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
    
    unitree_go::msg::LowState::SharedPtr latest_state_;
    unitree_go::msg::LowCmd low_cmd_;
    
    mutable std::mutex state_mutex_;
    mutable std::mutex cmd_mutex_;
    
    std::atomic<bool> active_{false};
    std::atomic<bool> ready_{false};
    
    static constexpr int NUM_JOINTS = 12;
    static constexpr char LOW_CMD_TOPIC[] = "/unitree_go/low_cmd";
    static constexpr char LOW_STATE_TOPIC[] = "/unitree_go/low_state";
};

#endif // HARDWARE_INTERFACE_UNITREE_ROS2_HPP
