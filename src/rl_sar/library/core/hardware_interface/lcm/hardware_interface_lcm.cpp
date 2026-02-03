/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_lcm.hpp"
#include <iostream>
#include <chrono>

HardwareInterfaceLCM::HardwareInterfaceLCM()
    : running_(false)
    , ready_(false)
    , state_topic_("/low_state")
    , cmd_topic_("/low_cmd")
{
    joint_positions_.resize(NUM_MOTORS, 0.0f);
    joint_velocities_.resize(NUM_MOTORS, 0.0f);
    joint_torques_.resize(NUM_MOTORS, 0.0f);
    imu_quaternion_.resize(4, 0.0f);
    imu_gyroscope_.resize(3, 0.0f);
    imu_accelerometer_.resize(3, 0.0f);
    imu_quaternion_[0] = 1.0f;  // Initialize to identity quaternion
    
    // Initialize command message
    current_cmd_.timestamp = 0;
    for (int i = 0; i < NUM_MOTORS; ++i) {
        current_cmd_.motors[i].timestamp = 0;
        current_cmd_.motors[i].tor_des = 0.0f;
        current_cmd_.motors[i].spd_des = 0.0f;
        current_cmd_.motors[i].pos_des = 0.0f;
        current_cmd_.motors[i].k_pos = 0.0f;
        current_cmd_.motors[i].k_spd = 0.0f;
        current_cmd_.motors[i].mode_set = 10;
    }
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
    
    // Create send loop (500Hz, same as free_dog_sdk)
    send_loop_ = std::make_shared<LoopFunc>("LCM_Send", 0.002, 
        std::bind(&HardwareInterfaceLCM::lcmSendLoop, this), 3);
    
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
    
    // Start send loop
    if (send_loop_) {
        send_loop_->start();
    }
    
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
    
    // Stop send loop
    if (send_loop_) {
        send_loop_->shutdown();
    }
    
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
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    int64_t timestamp = getTimestampMs();
    current_cmd_.timestamp = timestamp;
    
    // Update all motor commands
    for (int i = 0; i < NUM_MOTORS; ++i) {
        current_cmd_.motors[i].timestamp = timestamp;
        
        if (i < static_cast<int>(joint_positions.size())) {
            current_cmd_.motors[i].pos_des = joint_positions[i];
        }
        
        if (i < static_cast<int>(joint_velocities.size())) {
            current_cmd_.motors[i].spd_des = joint_velocities[i];
        }
        
        if (i < static_cast<int>(joint_torques.size())) {
            current_cmd_.motors[i].tor_des = joint_torques[i];
        }
        
        if (i < static_cast<int>(joint_kp.size())) {
            current_cmd_.motors[i].k_pos = joint_kp[i];
        }
        
        if (i < static_cast<int>(joint_kd.size())) {
            current_cmd_.motors[i].k_spd = joint_kd[i];
        }
        
        current_cmd_.motors[i].mode_set = 10;  // Servo mode
    }
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
                                           const exlcm::low_state* msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Parse motor states (assuming 12 motors from array)
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

void HardwareInterfaceLCM::lcmSendLoop() {
    if (!lcm_->good() || !running_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    // Publish the complete low_cmd message
    lcm_->publish(cmd_topic_, &current_cmd_);
}

int64_t HardwareInterfaceLCM::getTimestampMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
