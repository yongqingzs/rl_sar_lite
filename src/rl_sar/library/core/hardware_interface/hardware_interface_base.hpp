/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_BASE_HPP
#define HARDWARE_INTERFACE_BASE_HPP

#include <vector>
#include <memory>
#include <functional>

// Abstract base class for hardware communication interfaces
class HardwareInterfaceBase
{
public:
    virtual ~HardwareInterfaceBase() = default;

    // Initialize the hardware interface
    virtual bool Initialize() = 0;

    // Start communication loops
    virtual void Start() = 0;

    // Stop communication loops
    virtual void Stop() = 0;

    // Get robot state data
    virtual bool GetState(
        std::vector<float>& joint_positions,
        std::vector<float>& joint_velocities,
        std::vector<float>& joint_efforts,
        std::vector<float>& imu_quaternion,
        std::vector<float>& imu_gyroscope,
        std::vector<float>& imu_accelerometer) = 0;

    // Set robot command
    virtual void SetCommand(
        const std::vector<float>& joint_positions,
        const std::vector<float>& joint_velocities,
        const std::vector<float>& joint_torques,
        const std::vector<float>& joint_kp,
        const std::vector<float>& joint_kd) = 0;

    // Check if interface is ready
    virtual bool IsReady() const = 0;
};

#endif // HARDWARE_INTERFACE_BASE_HPP
