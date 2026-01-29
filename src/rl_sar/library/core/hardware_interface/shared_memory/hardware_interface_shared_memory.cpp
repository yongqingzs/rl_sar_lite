/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hardware_interface_shared_memory.hpp"
#include "logger.hpp"
#include <iostream>
#include <cstring>
#include <errno.h>
#include <unistd.h>

// Include shared memory definitions
// We need to define the structures here or include the header
struct S_Motor_cmd {
    float tor_des;
    float spd_des;
    float pos_des;
    float k_pos;
    float k_spd;
    uint16_t mode_set;
};

struct S_Motor_state {
    float tor;
    float spd;
    float pos;
    float temp;
    float spd_raw;
    float ddq;
    uint16_t err;
    uint16_t mode;
};

struct S_IMU_state {
    float acc[3];
    float gyro[3];
    float mag[3];
    float roll;
    float pitch;
    float yaw;
    float quat[4];
    float temp;
};

struct SharedMotorData {
    S_Motor_state motor_states[12];
    uint64_t update_counter;
    bool valid;
    double timestamp;
    int num_motors;
};

struct SharedMotorCmd {
    S_Motor_cmd motor_commands[12];
    uint64_t update_counter;
    bool valid;
    double timestamp;
    int num_motors;
};

struct SharedIMUData {
    S_IMU_state imu_states;
    uint64_t update_counter;
    bool valid;
    double timestamp;
};

// Shared memory keys
#define SHM_KEY     0x1234
#define SHMCMD_KEY  0x5678
#define SHMIMU_KEY  0x9ABC

#define SEM_KEY     0x2345
#define SEMCMD_KEY  0x6789
#define SEMIMU_KEY  0xABCD

HardwareInterfaceSharedMemory::HardwareInterfaceSharedMemory()
    : shm_motor_state_id_(-1),
      shm_motor_cmd_id_(-1),
      shm_imu_id_(-1),
      sem_motor_state_id_(-1),
      sem_motor_cmd_id_(-1),
      sem_imu_id_(-1),
      motor_state_data_(nullptr),
      motor_cmd_data_(nullptr),
      imu_data_(nullptr)
{
    cached_joint_positions_.resize(NUM_JOINTS, 0.0f);
    cached_joint_velocities_.resize(NUM_JOINTS, 0.0f);
    cached_joint_efforts_.resize(NUM_JOINTS, 0.0f);
    cached_imu_quaternion_.resize(4, 0.0f);
    cached_imu_quaternion_[0] = 1.0f; // w component
    cached_imu_gyroscope_.resize(3, 0.0f);
    cached_imu_accelerometer_.resize(3, 0.0f);
    cached_imu_accelerometer_[2] = 9.81f; // gravity
}

HardwareInterfaceSharedMemory::~HardwareInterfaceSharedMemory()
{
    Stop();
    
    // Detach shared memory
    if (motor_state_data_ != nullptr && motor_state_data_ != (void*)-1) {
        shmdt(motor_state_data_);
    }
    if (motor_cmd_data_ != nullptr && motor_cmd_data_ != (void*)-1) {
        shmdt(motor_cmd_data_);
    }
    if (imu_data_ != nullptr && imu_data_ != (void*)-1) {
        shmdt(imu_data_);
    }
}

bool HardwareInterfaceSharedMemory::Initialize()
{
    std::cout << LOGGER::INFO << "Initializing Shared Memory hardware interface" << std::endl;
    
    // Connect to motor state shared memory
    shm_motor_state_id_ = shmget(SHM_KEY, sizeof(SharedMotorData), 0666);
    if (shm_motor_state_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to motor state shared memory: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    // Connect to motor command shared memory
    shm_motor_cmd_id_ = shmget(SHMCMD_KEY, sizeof(SharedMotorCmd), 0666);
    if (shm_motor_cmd_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to motor command shared memory: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    // Connect to IMU shared memory
    shm_imu_id_ = shmget(SHMIMU_KEY, sizeof(SharedIMUData), 0666);
    if (shm_imu_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to IMU shared memory: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    // Get semaphore IDs
    sem_motor_state_id_ = semget(SEM_KEY, 1, 0666);
    if (sem_motor_state_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to motor state semaphore: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    sem_motor_cmd_id_ = semget(SEMCMD_KEY, 1, 0666);
    if (sem_motor_cmd_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to motor command semaphore: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    sem_imu_id_ = semget(SEMIMU_KEY, 1, 0666);
    if (sem_imu_id_ == -1) {
        std::cerr << LOGGER::ERROR << "Failed to connect to IMU semaphore: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    // Map shared memory
    motor_state_data_ = static_cast<SharedMotorData*>(shmat(shm_motor_state_id_, nullptr, 0));
    if (motor_state_data_ == (void*)-1) {
        std::cerr << LOGGER::ERROR << "Failed to map motor state shared memory: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    motor_cmd_data_ = static_cast<SharedMotorCmd*>(shmat(shm_motor_cmd_id_, nullptr, 0));
    if (motor_cmd_data_ == (void*)-1) {
        std::cerr << LOGGER::ERROR << "Failed to map motor command shared memory: " 
                  << strerror(errno) << std::endl;
        shmdt(motor_state_data_);
        return false;
    }
    
    imu_data_ = static_cast<SharedIMUData*>(shmat(shm_imu_id_, nullptr, 0));
    if (imu_data_ == (void*)-1) {
        std::cerr << LOGGER::ERROR << "Failed to map IMU shared memory: " 
                  << strerror(errno) << std::endl;
        shmdt(motor_state_data_);
        shmdt(motor_cmd_data_);
        return false;
    }
    
    initialized_ = true;
    std::cout << LOGGER::INFO << "Shared Memory hardware interface initialized successfully" << std::endl;
    return true;
}

void HardwareInterfaceSharedMemory::Start()
{
    if (!initialized_) {
        std::cerr << LOGGER::ERROR << "Cannot start: hardware interface not initialized" << std::endl;
        return;
    }
    
    ready_ = true;
    std::cout << LOGGER::INFO << "Shared Memory hardware interface started" << std::endl;
}

void HardwareInterfaceSharedMemory::Stop()
{
    ready_ = false;
    std::cout << LOGGER::INFO << "Shared Memory hardware interface stopped" << std::endl;
}

bool HardwareInterfaceSharedMemory::GetState(
    std::vector<float>& joint_positions,
    std::vector<float>& joint_velocities,
    std::vector<float>& joint_efforts,
    std::vector<float>& imu_quaternion,
    std::vector<float>& imu_gyroscope,
    std::vector<float>& imu_accelerometer)
{
    if (!ready_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    struct timespec timeout_ts = {0, TIMEOUT_MS * 1000000};
    struct sembuf sb_lock = {0, -1, SEM_UNDO};
    struct sembuf sb_unlock = {0, 1, SEM_UNDO};
    
    // Read motor state data
    if (semtimedop(sem_motor_state_id_, &sb_lock, 1, &timeout_ts) == 0) {
        if (motor_state_data_->valid && motor_state_data_->num_motors >= NUM_JOINTS) {
            for (int i = 0; i < NUM_JOINTS; ++i) {
                cached_joint_positions_[i] = motor_state_data_->motor_states[i].pos;
                cached_joint_velocities_[i] = motor_state_data_->motor_states[i].spd;
                cached_joint_efforts_[i] = motor_state_data_->motor_states[i].tor;
            }
        }
        semtimedop(sem_motor_state_id_, &sb_unlock, 1, &timeout_ts);
    }
    
    // Read IMU data
    if (semtimedop(sem_imu_id_, &sb_lock, 1, &timeout_ts) == 0) {
        if (imu_data_->valid) {
            for (int i = 0; i < 4; ++i) {
                cached_imu_quaternion_[i] = imu_data_->imu_states.quat[i];
            }
            for (int i = 0; i < 3; ++i) {
                cached_imu_gyroscope_[i] = imu_data_->imu_states.gyro[i];
                cached_imu_accelerometer_[i] = imu_data_->imu_states.acc[i];
            }
        }
        semtimedop(sem_imu_id_, &sb_unlock, 1, &timeout_ts);
    }
    
    // Copy to output
    joint_positions = cached_joint_positions_;
    joint_velocities = cached_joint_velocities_;
    joint_efforts = cached_joint_efforts_;
    imu_quaternion = cached_imu_quaternion_;
    imu_gyroscope = cached_imu_gyroscope_;
    imu_accelerometer = cached_imu_accelerometer_;
    
    return true;
}

void HardwareInterfaceSharedMemory::SetCommand(
    const std::vector<float>& joint_positions,
    const std::vector<float>& joint_velocities,
    const std::vector<float>& joint_torques,
    const std::vector<float>& joint_kp,
    const std::vector<float>& joint_kd)
{
    if (!ready_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    
    struct sembuf sb_lock = {0, -1, SEM_UNDO};
    struct sembuf sb_unlock = {0, 1, SEM_UNDO};
    
    if (semop(sem_motor_cmd_id_, &sb_lock, 1) == 0) {
        motor_cmd_data_->valid = true;
        motor_cmd_data_->num_motors = NUM_JOINTS;
        motor_cmd_data_->update_counter++;
        
        for (int i = 0; i < NUM_JOINTS && i < (int)joint_positions.size(); ++i) {
            motor_cmd_data_->motor_commands[i].pos_des = joint_positions[i];
            motor_cmd_data_->motor_commands[i].spd_des = 
                i < (int)joint_velocities.size() ? joint_velocities[i] : 0.0f;
            motor_cmd_data_->motor_commands[i].tor_des = 
                i < (int)joint_torques.size() ? joint_torques[i] : 0.0f;
            motor_cmd_data_->motor_commands[i].k_pos = 
                i < (int)joint_kp.size() ? joint_kp[i] : 0.0f;
            motor_cmd_data_->motor_commands[i].k_spd = 
                i < (int)joint_kd.size() ? joint_kd[i] : 0.0f;
            motor_cmd_data_->motor_commands[i].mode_set = 10; // servo mode
        }
        
        semop(sem_motor_cmd_id_, &sb_unlock, 1);
    }
}

bool HardwareInterfaceSharedMemory::IsReady() const
{
    return ready_;
}
