/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HARDWARE_INTERFACE_SHARED_MEMORY_HPP
#define HARDWARE_INTERFACE_SHARED_MEMORY_HPP

#include "hardware_interface_base.hpp"
#include <memory>
#include <atomic>
#include <mutex>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/ipc.h>

// Forward declarations for shared memory structures
struct SharedMotorData;
struct SharedMotorCmd;
struct SharedIMUData;

/**
 * Hardware interface implementation using shared memory IPC
 * This adapter wraps shared memory communication for motor control
 */
class HardwareInterfaceSharedMemory : public HardwareInterfaceBase
{
public:
    HardwareInterfaceSharedMemory();
    ~HardwareInterfaceSharedMemory() override;

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
    // Shared memory IDs
    int shm_motor_state_id_;
    int shm_motor_cmd_id_;
    int shm_imu_id_;
    
    // Semaphore IDs
    int sem_motor_state_id_;
    int sem_motor_cmd_id_;
    int sem_imu_id_;
    
    // Shared memory pointers
    SharedMotorData* motor_state_data_;
    SharedMotorCmd* motor_cmd_data_;
    SharedIMUData* imu_data_;
    
    std::atomic<bool> initialized_{false};
    std::atomic<bool> ready_{false};
    
    mutable std::mutex state_mutex_;
    mutable std::mutex cmd_mutex_;
    
    // Cached state data
    std::vector<float> cached_joint_positions_;
    std::vector<float> cached_joint_velocities_;
    std::vector<float> cached_joint_efforts_;
    std::vector<float> cached_imu_quaternion_;
    std::vector<float> cached_imu_gyroscope_;
    std::vector<float> cached_imu_accelerometer_;
    
    static constexpr int NUM_JOINTS = 12;
    static constexpr int TIMEOUT_MS = 1;
};

#endif // HARDWARE_INTERFACE_SHARED_MEMORY_HPP
