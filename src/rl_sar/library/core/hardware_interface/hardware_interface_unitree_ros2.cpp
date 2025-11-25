/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_unitree_ros2.hpp"

constexpr char HardwareInterfaceUnitreeRos2::LOW_CMD_TOPIC[];
constexpr char HardwareInterfaceUnitreeRos2::LOW_STATE_TOPIC[];

HardwareInterfaceUnitreeRos2::HardwareInterfaceUnitreeRos2(rclcpp::Node::SharedPtr node)
    : node_(node)
{
}

HardwareInterfaceUnitreeRos2::~HardwareInterfaceUnitreeRos2()
{
    Stop();
}

bool HardwareInterfaceUnitreeRos2::Initialize()
{
    try {
        // Create QoS profile for reliable communication
        auto qos = rclcpp::QoS(10).reliable();
        
        // Create publisher for low-level commands
        low_cmd_pub_ = node_->create_publisher<unitree_go::msg::LowCmd>(
            LOW_CMD_TOPIC, qos);
        
        // Create subscriber for low-level state
        low_state_sub_ = node_->create_subscription<unitree_go::msg::LowState>(
            LOW_STATE_TOPIC, qos,
            std::bind(&HardwareInterfaceUnitreeRos2::lowStateCallback, this, std::placeholders::_1));
        
        // Initialize command message - motor_cmd is a fixed-size array, no need to resize
        for (int i = 0; i < NUM_JOINTS && i < (int)low_cmd_.motor_cmd.size(); ++i) {
            low_cmd_.motor_cmd[i].mode = 0x01;  // Servo mode
            low_cmd_.motor_cmd[i].q = 0.0f;
            low_cmd_.motor_cmd[i].dq = 0.0f;
            low_cmd_.motor_cmd[i].tau = 0.0f;
            low_cmd_.motor_cmd[i].kp = 0.0f;
            low_cmd_.motor_cmd[i].kd = 0.0f;
        }

        ready_ = true;
        RCLCPP_INFO(node_->get_logger(), "[UnitreeROS2] Hardware interface initialized");
        return true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "[UnitreeROS2] Failed to initialize: %s", e.what());
        return false;
    }
}

void HardwareInterfaceUnitreeRos2::Start()
{
    if (active_) {
        RCLCPP_INFO(node_->get_logger(), "[UnitreeROS2] Already active");
        return;
    }
    
    active_ = true;
    RCLCPP_INFO(node_->get_logger(), "[UnitreeROS2] Interface activated");
}

void HardwareInterfaceUnitreeRos2::Stop()
{
    if (!active_) {
        return;
    }
    
    active_ = false;
    ready_ = false;
    
    RCLCPP_INFO(node_->get_logger(), "[UnitreeROS2] Interface deactivated");
}

bool HardwareInterfaceUnitreeRos2::GetState(
    std::vector<float>& joint_positions,
    std::vector<float>& joint_velocities,
    std::vector<float>& joint_efforts,
    std::vector<float>& imu_quaternion,
    std::vector<float>& imu_gyroscope,
    std::vector<float>& imu_accelerometer)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!latest_state_ || !ready_) {
        return false;
    }
    
    // Resize vectors if necessary
    joint_positions.resize(NUM_JOINTS);
    joint_velocities.resize(NUM_JOINTS);
    joint_efforts.resize(NUM_JOINTS);
    imu_quaternion.resize(4);
    imu_gyroscope.resize(3);
    imu_accelerometer.resize(3);
    
    // Copy joint states
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(latest_state_->motor_state.size()); ++i) {
        joint_positions[i] = latest_state_->motor_state[i].q;
        joint_velocities[i] = latest_state_->motor_state[i].dq;
        joint_efforts[i] = latest_state_->motor_state[i].tau_est;
    }
    
    // Copy IMU data
    imu_quaternion[0] = latest_state_->imu_state.quaternion[0];
    imu_quaternion[1] = latest_state_->imu_state.quaternion[1];
    imu_quaternion[2] = latest_state_->imu_state.quaternion[2];
    imu_quaternion[3] = latest_state_->imu_state.quaternion[3];
    
    imu_gyroscope[0] = latest_state_->imu_state.gyroscope[0];
    imu_gyroscope[1] = latest_state_->imu_state.gyroscope[1];
    imu_gyroscope[2] = latest_state_->imu_state.gyroscope[2];
    
    imu_accelerometer[0] = latest_state_->imu_state.accelerometer[0];
    imu_accelerometer[1] = latest_state_->imu_state.accelerometer[1];
    imu_accelerometer[2] = latest_state_->imu_state.accelerometer[2];
    
    return true;
}

void HardwareInterfaceUnitreeRos2::SetCommand(
    const std::vector<float>& joint_positions,
    const std::vector<float>& joint_velocities,
    const std::vector<float>& joint_torques,
    const std::vector<float>& joint_kp,
    const std::vector<float>& joint_kd)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_positions.size()); ++i) {
        low_cmd_.motor_cmd[i].mode = 0x01;  // Servo mode
        low_cmd_.motor_cmd[i].q = joint_positions[i];
        low_cmd_.motor_cmd[i].dq = i < static_cast<int>(joint_velocities.size()) ? joint_velocities[i] : 0.0f;
        low_cmd_.motor_cmd[i].tau = i < static_cast<int>(joint_torques.size()) ? joint_torques[i] : 0.0f;
        low_cmd_.motor_cmd[i].kp = i < static_cast<int>(joint_kp.size()) ? joint_kp[i] : 0.0f;
        low_cmd_.motor_cmd[i].kd = i < static_cast<int>(joint_kd.size()) ? joint_kd[i] : 0.0f;
    }
    
    // Publish command
    if (active_ && low_cmd_pub_) {
        low_cmd_pub_->publish(low_cmd_);
    }
}

bool HardwareInterfaceUnitreeRos2::IsReady() const
{
    return ready_;
}

void HardwareInterfaceUnitreeRos2::lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg)
{
    if (!active_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = msg;
    
    // if (!ready_) {
        // ready_ = true;
        // RCLCPP_INFO(node_->get_logger(), "[UnitreeROS2] First state received");
    // }
}
