/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_free_dog_sdk.hpp"
#include "loop.hpp"
#include <iostream>
#include <cstring>

#ifdef USE_ROS2
#include <unitree_go/msg/motor_state.hpp>
#include <unitree_go/msg/motor_cmd.hpp>
#include <unitree_go/msg/imu_state.hpp>
#include <unitree_go/msg/bms_state.hpp>
#endif

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
    
#ifdef USE_ROS2
    if (ros2_publish_loop_) {
        ros2_publish_loop_->shutdown();
    }
    ros2_bridge_enabled_ = false;
#endif
    
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

#ifdef USE_ROS2
void HardwareInterfaceFreeDogSdk::EnableRos2Bridge(bool enable, std::shared_ptr<rclcpp::Node> node)
{
    if (enable && node) {
        ros2_node_ = node;
        ros2_bridge_enabled_ = true;
        
        // Create publishers
        state_pub_ = ros2_node_->create_publisher<unitree_go::msg::LowState>("dog_state", 10);
        cmd_pub_ = ros2_node_->create_publisher<unitree_go::msg::LowCmd>("dog_cmd", 10);
        
        // Create ROS2 publish loop (200Hz)
        ros2_publish_loop_ = std::make_shared<LoopFunc>("FreeDogSDK_ROS2", 0.005, 
            std::bind(&HardwareInterfaceFreeDogSdk::Ros2PublishLoop, this), 3);
        ros2_publish_loop_->start();
        
        std::cout << "[FreeDogSDK] ROS2 bridge enabled - publishing to /dog_state and /dog_cmd" << std::endl;
    } else {
        ros2_bridge_enabled_ = false;
        if (ros2_publish_loop_) {
            ros2_publish_loop_->shutdown();
        }
        state_pub_.reset();
        cmd_pub_.reset();
        ros2_node_.reset();
        
        std::cout << "[FreeDogSDK] ROS2 bridge disabled" << std::endl;
    }
}

void HardwareInterfaceFreeDogSdk::Ros2PublishLoop()
{
    if (!ros2_bridge_enabled_ || !ros2_node_) {
        return;
    }
    
    PublishLowState();
    PublishLowCmd();
}

void HardwareInterfaceFreeDogSdk::PublishLowState()
{
    if (!state_pub_) return;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto msg = unitree_go::msg::LowState();
    
    msg.head[0] = low_state_.head[0];
    msg.head[1] = low_state_.head[1];
    msg.level_flag = low_state_.levelFlag;
    msg.frame_reserve = low_state_.frameReserve;
    
    for (int i = 0; i < 2; ++i) {
        uint32_t sn_val = 0, ver_val = 0;
        for (int j = 0; j < 4; ++j) {
            sn_val |= (static_cast<uint32_t>(low_state_.SN[i*4 + j]) << (j * 8));
            ver_val |= (static_cast<uint32_t>(low_state_.version[i*4 + j]) << (j * 8));
        }
        msg.sn[i] = sn_val;
        msg.version[i] = ver_val;
    }
    
    msg.bandwidth = static_cast<uint16_t>(low_state_.bandWidth[0]) | 
                   (static_cast<uint16_t>(low_state_.bandWidth[1]) << 8);
    
    msg.imu_state.quaternion[0] = low_state_.imu_quaternion[0];
    msg.imu_state.quaternion[1] = low_state_.imu_quaternion[1];
    msg.imu_state.quaternion[2] = low_state_.imu_quaternion[2];
    msg.imu_state.quaternion[3] = low_state_.imu_quaternion[3];
    
    msg.imu_state.gyroscope[0] = low_state_.imu_gyroscope[0];
    msg.imu_state.gyroscope[1] = low_state_.imu_gyroscope[1];
    msg.imu_state.gyroscope[2] = low_state_.imu_gyroscope[2];
    
    msg.imu_state.accelerometer[0] = low_state_.imu_accelerometer[0];
    msg.imu_state.accelerometer[1] = low_state_.imu_accelerometer[1];
    msg.imu_state.accelerometer[2] = low_state_.imu_accelerometer[2];
    
    msg.imu_state.rpy[0] = low_state_.imu_rpy[0];
    msg.imu_state.rpy[1] = low_state_.imu_rpy[1];
    msg.imu_state.rpy[2] = low_state_.imu_rpy[2];
    
    msg.imu_state.temperature = low_state_.temperature_imu;
    
    for (int i = 0; i < 20; ++i) {
        msg.motor_state[i].mode = low_state_.motorState[i].mode;
        msg.motor_state[i].q = low_state_.motorState[i].q;
        msg.motor_state[i].dq = low_state_.motorState[i].dq;
        msg.motor_state[i].ddq = low_state_.motorState[i].ddq;
        msg.motor_state[i].tau_est = low_state_.motorState[i].tauEst;
        msg.motor_state[i].q_raw = low_state_.motorState[i].q_raw;
        msg.motor_state[i].dq_raw = low_state_.motorState[i].dq_raw;
        msg.motor_state[i].ddq_raw = low_state_.motorState[i].ddq_raw;
        msg.motor_state[i].temperature = low_state_.motorState[i].temperature;
        msg.motor_state[i].lost = 0;
        msg.motor_state[i].reserve[0] = 0;
        msg.motor_state[i].reserve[1] = 0;
    }
    
    msg.bms_state.version_high = low_state_.bm_s.version_h;
    msg.bms_state.version_low = low_state_.bm_s.version_l;
    msg.bms_state.status = low_state_.bm_s.bms_status;
    msg.bms_state.soc = low_state_.bm_s.SOC;
    msg.bms_state.current = low_state_.bm_s.current;
    msg.bms_state.cycle = low_state_.bm_s.cycle;
    
    for (int i = 0; i < 2; ++i) {
        msg.bms_state.bq_ntc[i] = low_state_.bm_s.BQ_NTC[i];
        msg.bms_state.mcu_ntc[i] = low_state_.bm_s.MCU_NTC[i];
    }
    
    for (int i = 0; i < 10; ++i) {
        msg.bms_state.cell_vol[i] = low_state_.bm_s.cell_vol[i];
    }
    
    for (int i = 0; i < 4; ++i) {
        msg.foot_force[i] = low_state_.footForce[i];
        msg.foot_force_est[i] = low_state_.footForceEst[i];
    }
    
    for (int i = 0; i < 40; ++i) {
        msg.wireless_remote[i] = low_state_.wirelessRemote[i];
    }
    
    msg.tick = 0;
    msg.reserve = 0;
    msg.crc = 0;
    
    state_pub_->publish(msg);
}

void HardwareInterfaceFreeDogSdk::PublishLowCmd()
{
    if (!cmd_pub_) return;
    
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    auto msg = unitree_go::msg::LowCmd();
    
    msg.head[0] = low_cmd_.head[0];
    msg.head[1] = low_cmd_.head[1];
    msg.level_flag = low_cmd_.levelFlag;
    msg.frame_reserve = low_cmd_.frameReserve;
    
    for (int i = 0; i < 2; ++i) {
        uint32_t sn_val = 0, ver_val = 0;
        for (int j = 0; j < 4; ++j) {
            sn_val |= (static_cast<uint32_t>(low_cmd_.SN[i*4 + j]) << (j * 8));
            ver_val |= (static_cast<uint32_t>(low_cmd_.version[i*4 + j]) << (j * 8));
        }
        msg.sn[i] = sn_val;
        msg.version[i] = ver_val;
    }
    
    msg.bandwidth = static_cast<uint16_t>(low_cmd_.bandWidth[0]) | 
                   (static_cast<uint16_t>(low_cmd_.bandWidth[1]) << 8);
    
    for (int i = 0; i < 20; ++i) {
        msg.motor_cmd[i].mode = static_cast<uint8_t>(low_cmd_.motorCmd.motors[i].mode);
        msg.motor_cmd[i].q = low_cmd_.motorCmd.motors[i].q;
        msg.motor_cmd[i].dq = low_cmd_.motorCmd.motors[i].dq;
        msg.motor_cmd[i].tau = low_cmd_.motorCmd.motors[i].tau;
        msg.motor_cmd[i].kp = low_cmd_.motorCmd.motors[i].Kp;
        msg.motor_cmd[i].kd = low_cmd_.motorCmd.motors[i].Kd;
        msg.motor_cmd[i].reserve[0] = 0;
        msg.motor_cmd[i].reserve[1] = 0;
        msg.motor_cmd[i].reserve[2] = 0;
    }
    
    for (int i = 0; i < 12; ++i) {
        msg.led[i] = 0;
    }
    
    for (int i = 0; i < 40; ++i) {
        msg.wireless_remote[i] = low_cmd_.wirelessRemote[i];
    }
    
    msg.reserve = 0;
    msg.crc = 0;
    
    cmd_pub_->publish(msg);
}
#endif
