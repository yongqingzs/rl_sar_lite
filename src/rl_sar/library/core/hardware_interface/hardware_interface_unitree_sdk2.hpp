/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_UNITREE_SDK2_HPP
#define HARDWARE_INTERFACE_UNITREE_SDK2_HPP

#include "hardware_interface_base.hpp"
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <mutex>
#include <atomic>
#include <memory>

/**
 * Hardware interface implementation using unitree_sdk2 DDS communication
 * This adapter communicates directly via DDS channels (no ROS2 bridge needed)
 */
class HardwareInterfaceUnitreeSdk2 : public HardwareInterfaceBase
{
public:
    HardwareInterfaceUnitreeSdk2();
    ~HardwareInterfaceUnitreeSdk2() override;

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

    // Configuration options
    void SetNetworkInterface(const std::string& interface);
    void SetDomain(int domain);

private:
    void lowStateCallback(const void* message);
    void highStateCallback(const void* message);
    void initLowCmd();

    // DDS communication channels
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> low_cmd_publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_state_subscriber_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> high_state_subscriber_;
    
    // Message buffers
    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    unitree_go::msg::dds_::LowState_ low_state_{};
    unitree_go::msg::dds_::SportModeState_ high_state_{};
    
    mutable std::mutex state_mutex_;
    mutable std::mutex cmd_mutex_;
    
    std::atomic<bool> active_{false};
    std::atomic<bool> ready_{false};
    std::atomic<bool> state_received_{false};
    
    // Configuration
    std::string network_interface_{"lo"};  // Default to loopback for simulation
    int domain_{0};  // Default domain
    
    static constexpr int NUM_JOINTS = 12;
    static constexpr char TOPIC_LOWCMD[] = "rt/lowcmd";
    static constexpr char TOPIC_LOWSTATE[] = "rt/lowstate";
    static constexpr char TOPIC_HIGHSTATE[] = "rt/sportmodestate";
};

#endif // HARDWARE_INTERFACE_UNITREE_SDK2_HPP
