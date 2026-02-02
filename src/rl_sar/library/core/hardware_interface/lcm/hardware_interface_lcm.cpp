/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_lcm.hpp"
#include <iostream>
#include <chrono>

using namespace std::chrono;

HardwareInterfaceLCM::HardwareInterfaceLCM()
    : running_(false)
    , ready_(false)
    , state_topic_("/unitree_go/low_state")
    , cmd_topic_("/unitree_go/low_cmd")
{
    joint_positions_.resize(NUM_MOTORS, 0.0f);
    joint_velocities_.resize(NUM_MOTORS, 0.0f);
    joint_torques_.resize(NUM_MOTORS, 0.0f);
    imu_quaternion_.resize(4, 0.0f);
    imu_gyroscope_.resize(3, 0.0f);
    imu_accelerometer_.resize(3, 0.0f);
    imu_quaternion_[0] = 1.0f;  // Initialize to identity quaternion
}

HardwareInterfaceLCM::~HardwareInterfaceLCM() {
    Stop();
}

bool HardwareInterfaceLCM::Initialize() {
    lcm_ = std::make_unique<lcm::LCM>();
    if (!lcm_->good()) {
        std::cerr << "[HardwareInterfaceLCM] Failed to initialize LCM" << std::endl;
        return false;
    }
    
    // Subscribe to state topic
    lcm_->subscribe(state_topic_, &HardwareInterfaceLCM::lcmStateHandler, this);
    
    std::cout << "[HardwareInterfaceLCM] Initialized successfully" << std::endl;
    std::cout << "[HardwareInterfaceLCM] State topic: " << state_topic_ << std::endl;
    std::cout << "[HardwareInterfaceLCM] Command topic: " << cmd_topic_ << std::endl;
    
    return true;
}

void HardwareInterfaceLCM::Start() {
    if (running_) {
        std::cerr << "[HardwareInterfaceLCM] Already running" << std::endl;
        return;
    }
    
    running_ = true;
    lcm_thread_ = std::thread(&HardwareInterfaceLCM::lcmHandleLoop, this);
    
    // Wait for first state message
    auto start = std::chrono::steady_clock::now();
    while (!ready_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (!ready_) {
        std::cerr << "[HardwareInterfaceLCM] Timeout waiting for state messages" << std::endl;
        Stop();
        return;
    }
    
    std::cout << "[HardwareInterfaceLCM] Started successfully" << std::endl;
}

void HardwareInterfaceLCM::Stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (lcm_thread_.joinable()) {
        lcm_thread_.join();
    }
    
    ready_ = false;
    std::cout << "[HardwareInterfaceLCM] Stopped" << std::endl;
}

bool HardwareInterfaceLCM::GetState(std::vector<float>& joint_positions,
                                    std::vector<float>& joint_velocities,
                                    std::vector<float>& joint_efforts,
                                    std::vector<float>& imu_quaternion,
                                    std::vector<float>& imu_gyroscope,
                                    std::vector<float>& imu_accelerometer) {
    if (!ready_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    joint_positions = joint_positions_;
    joint_velocities = joint_velocities_;
    joint_efforts = joint_torques_;
    imu_quaternion = imu_quaternion_;
    imu_gyroscope = imu_gyroscope_;
    imu_accelerometer = imu_accelerometer_;
    
    return true;
}

void HardwareInterfaceLCM::SetCommand(const std::vector<float>& joint_positions,
                                      const std::vector<float>& joint_velocities,
                                      const std::vector<float>& joint_torques,
                                      const std::vector<float>& joint_kp,
                                      const std::vector<float>& joint_kd) {
    if (!ready_ || !lcm_->good()) {
        return;
    }
    
    // For now, send a single unified command
    // In a real implementation, you might send per-motor commands
    exlcm::low_motor_ctrl cmd;
    cmd.timestamp = getTimestampMs();
    
    // Use first motor values as example (modify as needed)
    if (!joint_torques.empty()) {
        cmd.tor_des = joint_torques[0];
    } else {
        cmd.tor_des = 0.0f;
    }
    
    if (!joint_velocities.empty()) {
        cmd.spd_des = joint_velocities[0];
    } else {
        cmd.spd_des = 0.0f;
    }
    
    if (!joint_positions.empty()) {
        cmd.pos_des = joint_positions[0];
    } else {
        cmd.pos_des = 0.0f;
    }
    
    if (!joint_kp.empty()) {
        cmd.k_pos = joint_kp[0];
    } else {
        cmd.k_pos = 1.0f;
    }
    
    if (!joint_kd.empty()) {
        cmd.k_spd = joint_kd[0];
    } else {
        cmd.k_spd = 0.1f;
    }
    
    cmd.mode_set = 0;  // Default mode
    
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    lcm_->publish(cmd_topic_, &cmd);
}

bool HardwareInterfaceLCM::IsReady() const {
    return ready_;
}

void HardwareInterfaceLCM::lcmHandleLoop() {
    while (running_) {
        lcm_->handleTimeout(1);  // 1ms timeout
    }
}

void HardwareInterfaceLCM::lcmStateHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                                           const std::string& /*chan*/,
                                           const exlcm::low_motor_state* msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Parse motor states (assuming 12 motors from array)
    // motors is a fixed-size array of motor_data
    for (int i = 0; i < NUM_MOTORS; ++i) {
        joint_positions_[i] = msg->motors[i].pos;
        joint_velocities_[i] = msg->motors[i].spd;
        joint_torques_[i] = msg->motors[i].tor;
    }
    
    // Parse IMU data - quat is [w, x, y, z]
    imu_quaternion_[0] = msg->imu.quat[0];  // w
    imu_quaternion_[1] = msg->imu.quat[1];  // x
    imu_quaternion_[2] = msg->imu.quat[2];  // y
    imu_quaternion_[3] = msg->imu.quat[3];  // z
    
    imu_gyroscope_[0] = msg->imu.gyro[0];
    imu_gyroscope_[1] = msg->imu.gyro[1];
    imu_gyroscope_[2] = msg->imu.gyro[2];
    
    imu_accelerometer_[0] = msg->imu.acc[0];
    imu_accelerometer_[1] = msg->imu.acc[1];
    imu_accelerometer_[2] = msg->imu.acc[2];
    
    ready_ = true;
}

int64_t HardwareInterfaceLCM::getTimestampMs() const {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
