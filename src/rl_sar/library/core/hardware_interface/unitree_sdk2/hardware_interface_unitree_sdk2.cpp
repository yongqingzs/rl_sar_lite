/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_unitree_sdk2.hpp"
#include <iostream>
#include <cstring>

constexpr char HardwareInterfaceUnitreeSdk2::TOPIC_LOWCMD[];
constexpr char HardwareInterfaceUnitreeSdk2::TOPIC_LOWSTATE[];
constexpr char HardwareInterfaceUnitreeSdk2::TOPIC_HIGHSTATE[];

HardwareInterfaceUnitreeSdk2::HardwareInterfaceUnitreeSdk2()
{
}

HardwareInterfaceUnitreeSdk2::~HardwareInterfaceUnitreeSdk2()
{
    Stop();
}

void HardwareInterfaceUnitreeSdk2::SetNetworkInterface(const std::string& interface)
{
    network_interface_ = interface;
}

void HardwareInterfaceUnitreeSdk2::SetDomain(int domain)
{
    domain_ = domain;
}

bool HardwareInterfaceUnitreeSdk2::Initialize()
{
    try {
        std::cout << "[SDK2] Initializing with network interface: " << network_interface_ 
                  << ", domain: " << domain_ << std::endl;
        
        // Initialize DDS channel factory
        unitree::robot::ChannelFactory::Instance()->Init(domain_, network_interface_);
        
        // Create publisher for low-level commands
        low_cmd_publisher_ = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
            TOPIC_LOWCMD);
        low_cmd_publisher_->InitChannel();
        
        // Create subscriber for low-level state
        low_state_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
            TOPIC_LOWSTATE);
        low_state_subscriber_->InitChannel(
            [this](const void* msg) { lowStateCallback(msg); }, 1);
        
        // Create subscriber for high-level state (sport mode)
        high_state_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(
            TOPIC_HIGHSTATE);
        high_state_subscriber_->InitChannel(
            [this](const void* msg) { highStateCallback(msg); }, 1);
        
        // Initialize command message
        initLowCmd();
        
        ready_ = true;
        std::cout << "[SDK2] Hardware interface initialized successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[SDK2] Failed to initialize: " << e.what() << std::endl;
        return false;
    }
}

void HardwareInterfaceUnitreeSdk2::initLowCmd()
{
    // Initialize all motor commands to safe defaults
    for (int i = 0; i < NUM_JOINTS && i < 20; ++i) {
        low_cmd_.motor_cmd()[i].mode() = 0x01;  // Servo mode
        low_cmd_.motor_cmd()[i].q() = 0.0f;
        low_cmd_.motor_cmd()[i].dq() = 0.0f;
        low_cmd_.motor_cmd()[i].tau() = 0.0f;
        low_cmd_.motor_cmd()[i].kp() = 0.0f;
        low_cmd_.motor_cmd()[i].kd() = 0.0f;
    }
    
    // CRC is calculated and set by the SDK before sending
}

void HardwareInterfaceUnitreeSdk2::Start()
{
    if (active_) {
        std::cout << "[SDK2] Already active" << std::endl;
        return;
    }
    
    active_ = true;
    std::cout << "[SDK2] Interface activated" << std::endl;
}

void HardwareInterfaceUnitreeSdk2::Stop()
{
    if (!active_) {
        return;
    }
    
    active_ = false;
    ready_ = false;
    
    // Send zero commands before stopping
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    initLowCmd();
    if (low_cmd_publisher_) {
        low_cmd_publisher_->Write(low_cmd_);
    }
    
    std::cout << "[SDK2] Interface deactivated" << std::endl;
}

void HardwareInterfaceUnitreeSdk2::lowStateCallback(const void* message)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    const auto* state_msg = static_cast<const unitree_go::msg::dds_::LowState_*>(message);
    low_state_ = *state_msg;
    state_received_ = true;
}

void HardwareInterfaceUnitreeSdk2::highStateCallback(const void* message)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    const auto* high_msg = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    high_state_ = *high_msg;
}

bool HardwareInterfaceUnitreeSdk2::GetState(
    std::vector<float>& joint_positions,
    std::vector<float>& joint_velocities,
    std::vector<float>& joint_efforts,
    std::vector<float>& imu_quaternion,
    std::vector<float>& imu_gyroscope,
    std::vector<float>& imu_accelerometer)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!state_received_ || !ready_) {
        return false;
    }
    
    // Resize vectors
    joint_positions.resize(NUM_JOINTS);
    joint_velocities.resize(NUM_JOINTS);
    joint_efforts.resize(NUM_JOINTS);
    imu_quaternion.resize(4);
    imu_gyroscope.resize(3);
    imu_accelerometer.resize(3);
    
    // Copy joint states (motor_state is an array in SDK2)
    for (int i = 0; i < NUM_JOINTS && i < 20; ++i) {
        joint_positions[i] = low_state_.motor_state()[i].q();
        joint_velocities[i] = low_state_.motor_state()[i].dq();
        joint_efforts[i] = low_state_.motor_state()[i].tau_est();
    }
    
    // Copy IMU data
    imu_quaternion[0] = low_state_.imu_state().quaternion()[0];
    imu_quaternion[1] = low_state_.imu_state().quaternion()[1];
    imu_quaternion[2] = low_state_.imu_state().quaternion()[2];
    imu_quaternion[3] = low_state_.imu_state().quaternion()[3];
    
    imu_gyroscope[0] = low_state_.imu_state().gyroscope()[0];
    imu_gyroscope[1] = low_state_.imu_state().gyroscope()[1];
    imu_gyroscope[2] = low_state_.imu_state().gyroscope()[2];
    
    imu_accelerometer[0] = low_state_.imu_state().accelerometer()[0];
    imu_accelerometer[1] = low_state_.imu_state().accelerometer()[1];
    imu_accelerometer[2] = low_state_.imu_state().accelerometer()[2];
    
    return true;
}

void HardwareInterfaceUnitreeSdk2::SetCommand(
    const std::vector<float>& joint_positions,
    const std::vector<float>& joint_velocities,
    const std::vector<float>& joint_torques,
    const std::vector<float>& joint_kp,
    const std::vector<float>& joint_kd)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    // Update motor commands
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_positions.size()) && i < 20; ++i) {
        low_cmd_.motor_cmd()[i].mode() = 0x01;  // Servo mode
        low_cmd_.motor_cmd()[i].q() = joint_positions[i];
        low_cmd_.motor_cmd()[i].dq() = i < static_cast<int>(joint_velocities.size()) ? joint_velocities[i] : 0.0f;
        low_cmd_.motor_cmd()[i].tau() = i < static_cast<int>(joint_torques.size()) ? joint_torques[i] : 0.0f;
        low_cmd_.motor_cmd()[i].kp() = i < static_cast<int>(joint_kp.size()) ? joint_kp[i] : 0.0f;
        low_cmd_.motor_cmd()[i].kd() = i < static_cast<int>(joint_kd.size()) ? joint_kd[i] : 0.0f;
    }
    
    // Publish command via DDS
    if (active_ && low_cmd_publisher_) {
        low_cmd_publisher_->Write(low_cmd_);
    }
}

bool HardwareInterfaceUnitreeSdk2::IsReady() const
{
    return ready_;
}
