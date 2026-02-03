/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_LCM_HPP
#define HARDWARE_INTERFACE_LCM_HPP

#include "../hardware_interface_base.hpp"
#include "loop.hpp"
#include <lcm/lcm-cpp.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include "types/exlcm/low_state.hpp"
#include "types/exlcm/low_cmd.hpp"

class HardwareInterfaceLCM : public HardwareInterfaceBase {
public:
    HardwareInterfaceLCM();
    ~HardwareInterfaceLCM() override;

    bool Initialize() override;
    void Start() override;
    void Stop() override;
    
    bool GetState(std::vector<float>& joint_positions,
                  std::vector<float>& joint_velocities,
                  std::vector<float>& joint_efforts,
                  std::vector<float>& imu_quaternion,
                  std::vector<float>& imu_gyroscope,
                  std::vector<float>& imu_accelerometer) override;
                  
    void SetCommand(const std::vector<float>& joint_positions,
                    const std::vector<float>& joint_velocities,
                    const std::vector<float>& joint_torques,
                    const std::vector<float>& joint_kp,
                    const std::vector<float>& joint_kd) override;
                    
    bool IsReady() const override;

private:
    void lcmHandleLoop();
    void lcmSendLoop();
    void lcmStateHandler(const lcm::ReceiveBuffer* rbuf, 
                        const std::string& chan, 
                        const exlcm::low_state* msg);

    std::unique_ptr<lcm::LCM> lcm_;
    std::thread lcm_thread_;
    std::shared_ptr<LoopFunc> send_loop_;
    std::atomic<bool> running_;
    std::atomic<bool> ready_;
    
    // State data
    std::mutex state_mutex_;
    std::vector<float> joint_positions_;
    std::vector<float> joint_velocities_;
    std::vector<float> joint_torques_;
    std::vector<float> imu_quaternion_;
    std::vector<float> imu_gyroscope_;
    std::vector<float> imu_accelerometer_;
    
    // Command data
    std::mutex cmd_mutex_;
    exlcm::low_cmd current_cmd_;
    
    // Topics
    std::string state_topic_;
    std::string cmd_topic_;
    
    // Motor count
    static constexpr int NUM_MOTORS = 12;
    
    // Timestamp
    int64_t getTimestampMs() const;
};

#endif // HARDWARE_INTERFACE_LCM_HPP
