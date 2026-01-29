/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_FREE_DOG_SDK_HPP
#define HARDWARE_INTERFACE_FREE_DOG_SDK_HPP

#include "hardware_interface_base.hpp"
#include "fdsc_utils/free_dog_sdk_h.hpp"
#include "loop.hpp"
#include <memory>
#include <atomic>
#include <mutex>

#ifdef USE_ROS2
#include <rclcpp/rclcpp.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#endif

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

#ifdef USE_ROS2
    void EnableRos2Bridge(bool enable, std::shared_ptr<rclcpp::Node> node = nullptr);
#endif

private:
    void RecvLoop();
    void SendLoop();
    
#ifdef USE_ROS2
    void Ros2PublishLoop();
    void PublishLowState();
    void PublishLowCmd();
#endif

    std::string connection_settings_;
    
    std::shared_ptr<FDSC::UnitreeConnection> fdsc_conn_;
    
    FDSC::lowState low_state_;
    FDSC::lowCmd low_cmd_;
    
    std::vector<std::vector<uint8_t>> data_buffer_;
    
    std::shared_ptr<LoopFunc> recv_loop_;
    std::shared_ptr<LoopFunc> send_loop_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    
    mutable std::mutex state_mutex_;
    mutable std::mutex cmd_mutex_;
    mutable std::mutex buffer_mutex_;
    
    static constexpr int NUM_JOINTS = 12;
    
#ifdef USE_ROS2
    std::atomic<bool> ros2_bridge_enabled_{false};
    std::shared_ptr<rclcpp::Node> ros2_node_;
    rclcpp::Publisher<unitree_go::msg::LowState>::SharedPtr state_pub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr cmd_pub_;
    std::shared_ptr<LoopFunc> ros2_publish_loop_;
#endif
};

#endif // HARDWARE_INTERFACE_FREE_DOG_SDK_HPP
