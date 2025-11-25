/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_FREE_DOG_SDK_HPP
#define HARDWARE_INTERFACE_FREE_DOG_SDK_HPP

#include "hardware_interface_base.hpp"
#include "fdsc_utils/free_dog_sdk_h.hpp"
#include <thread>
#include <atomic>
#include <mutex>

/**
 * Hardware interface implementation using free_dog_sdk (FDSC)
 * This adapter wraps the original free_dog_sdk communication
 */
class HardwareInterfaceFreeDogSdk : public HardwareInterfaceBase
{
public:
    HardwareInterfaceFreeDogSdk(const std::string& connection_settings = "LOW_WIRED_DEFAULTS");
    ~HardwareInterfaceFreeDogSdk() override;

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
    void RecvLoop();
    void SendLoop();

    std::string connection_settings_;
    
    std::shared_ptr<FDSC::UnitreeConnection> fdsc_conn_;
    
    FDSC::lowState low_state_;
    FDSC::lowCmd low_cmd_;
    
    std::vector<std::vector<uint8_t>> data_buffer_;
    
    std::thread recv_thread_;
    std::thread send_thread_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    
    mutable std::mutex state_mutex_;
    mutable std::mutex cmd_mutex_;
    mutable std::mutex buffer_mutex_;
    
    static constexpr int NUM_JOINTS = 12;
};

#endif // HARDWARE_INTERFACE_FREE_DOG_SDK_HPP
