/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_free_dog_sdk.hpp"
#include "loop.hpp"
#include <iostream>
#include <cstring>

HardwareInterfaceFreeDogSdk::HardwareInterfaceFreeDogSdk(const std::string& connection_settings)
    : connection_settings_(connection_settings)
{
}

HardwareInterfaceFreeDogSdk::~HardwareInterfaceFreeDogSdk()
{
    Stop();
}

bool HardwareInterfaceFreeDogSdk::Initialize()
{
    try {
        fdsc_conn_ = std::make_shared<FDSC::UnitreeConnection>(connection_settings_);
        fdsc_conn_->startRecv();
        
        // Send initial command to establish connection
        std::vector<uint8_t> init_cmd = low_cmd_.buildCmd(false);
        fdsc_conn_->send(init_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Create LoopFunc instances for communication loops
        recv_loop_ = std::make_shared<LoopFunc>("FreeDogSDK_Recv", 0.002, std::bind(&HardwareInterfaceFreeDogSdk::RecvLoop, this), 3);
        send_loop_ = std::make_shared<LoopFunc>("FreeDogSDK_Send", 0.002, std::bind(&HardwareInterfaceFreeDogSdk::SendLoop, this), 3);
        
        ready_ = true;
        std::cout << "[FreeDogSDK] Hardware interface initialized with settings: " 
                  << connection_settings_ << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[FreeDogSDK] Failed to initialize: " << e.what() << std::endl;
        return false;
    }
}

void HardwareInterfaceFreeDogSdk::Start()
{
    if (running_) {
        std::cout << "[FreeDogSDK] Already running" << std::endl;
        return;
    }
    
    running_ = true;
    
    if (recv_loop_) {
        recv_loop_->start();
    }
    if (send_loop_) {
        send_loop_->start();
    }
    
    std::cout << "[FreeDogSDK] Communication loops started" << std::endl;
}

void HardwareInterfaceFreeDogSdk::Stop()
{
    if (!running_) {
        return;
    }
    
    running_ = false;
    ready_ = false;
    
    if (recv_loop_) {
        recv_loop_->shutdown();
    }
    if (send_loop_) {
        send_loop_->shutdown();
    }
    
    if (fdsc_conn_) {
        fdsc_conn_->stopRecv();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        fdsc_conn_.reset();
    }
    
    std::cout << "[FreeDogSDK] Communication loops stopped" << std::endl;
}

bool HardwareInterfaceFreeDogSdk::GetState(
    std::vector<float>& joint_positions,
    std::vector<float>& joint_velocities,
    std::vector<float>& joint_efforts,
    std::vector<float>& imu_quaternion,
    std::vector<float>& imu_gyroscope,
    std::vector<float>& imu_accelerometer)
{
    // Get latest data from buffer
    std::vector<uint8_t> latest_data;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (!data_buffer_.empty()) {
            latest_data = data_buffer_.back();
            data_buffer_.clear();
        }
    }
    
    if (latest_data.empty()) {
        return false;
    }
    
    // Parse data
    try {
        low_state_.parseData(latest_data);
    }
    catch (const std::exception& e) {
        std::cerr << "[FreeDogSDK] Failed to parse low state data: " << e.what() << std::endl;
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
    for (int i = 0; i < NUM_JOINTS; ++i) {
        joint_positions[i] = low_state_.motorState[i].q;
        joint_velocities[i] = low_state_.motorState[i].dq;
        joint_efforts[i] = low_state_.motorState[i].tauEst;
    }
    
    // Copy IMU data - free_dog_sdk uses [w, x, y, z] format
    imu_quaternion[0] = low_state_.imu_quaternion[0];  // w
    imu_quaternion[1] = low_state_.imu_quaternion[1];  // x
    imu_quaternion[2] = low_state_.imu_quaternion[2];  // y
    imu_quaternion[3] = low_state_.imu_quaternion[3];  // z
    
    imu_gyroscope[0] = low_state_.imu_gyroscope[0];
    imu_gyroscope[1] = low_state_.imu_gyroscope[1];
    imu_gyroscope[2] = low_state_.imu_gyroscope[2];
    
    imu_accelerometer[0] = low_state_.imu_accelerometer[0];
    imu_accelerometer[1] = low_state_.imu_accelerometer[1];
    imu_accelerometer[2] = low_state_.imu_accelerometer[2];
    
    // if (!ready_) {
        // ready_ = true;
        // std::cout << "[FreeDogSDK] First state received" << std::endl;
    // }
    
    return true;
}

void HardwareInterfaceFreeDogSdk::SetCommand(
    const std::vector<float>& joint_positions,
    const std::vector<float>& joint_velocities,
    const std::vector<float>& joint_torques,
    const std::vector<float>& joint_kp,
    const std::vector<float>& joint_kd)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_positions.size()); ++i) {
        low_cmd_.motorCmd.motors[i].mode = FDSC::MotorModeLow::Servo;
        low_cmd_.motorCmd.motors[i].q = joint_positions[i];
        low_cmd_.motorCmd.motors[i].dq = i < static_cast<int>(joint_velocities.size()) ? joint_velocities[i] : 0.0f;
        low_cmd_.motorCmd.motors[i].tau = i < static_cast<int>(joint_torques.size()) ? joint_torques[i] : 0.0f;
        low_cmd_.motorCmd.motors[i].Kp = i < static_cast<int>(joint_kp.size()) ? joint_kp[i] : 0.0f;
        low_cmd_.motorCmd.motors[i].Kd = i < static_cast<int>(joint_kd.size()) ? joint_kd[i] : 0.0f;
    }
}

bool HardwareInterfaceFreeDogSdk::IsReady() const
{
    return ready_;
}

void HardwareInterfaceFreeDogSdk::RecvLoop()
{
    if (fdsc_conn_) {
        std::vector<std::vector<uint8_t>> dataall;
        fdsc_conn_->getData(dataall);
        
        if (!dataall.empty()) {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            data_buffer_ = std::move(dataall);
        }
    }
}

void HardwareInterfaceFreeDogSdk::SendLoop()
{
    if (fdsc_conn_) {
        std::vector<uint8_t> cmd_bytes;
        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            cmd_bytes = low_cmd_.buildCmd(false);
        }
        fdsc_conn_->send(cmd_bytes);
    }
}
