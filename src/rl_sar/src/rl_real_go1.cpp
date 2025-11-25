/*
* Copyright (c) 2024-2025 Ziqi Fan
* SPDX-License-Identifier: Apache-2.0
*/

#include "rl_real_go1.hpp"

RL_Real::RL_Real(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_node");
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

#if defined(CONTROL_EXTRA) && defined(USE_ROS)
    this->control_input_subscriber = ros2_node->create_subscription<control_input_msgs::msg::Inputs>(
        "/control_input", 10,
        [this] (const control_input_msgs::msg::Inputs::SharedPtr msg) {this->ControlInputsCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = "go1";
    this->ReadYaml(this->robot_name, "base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // Initialize hardware interface based on configuration
    InitializeHardwareInterface();
    
    // Initialize print timing
    this->last_print_time = std::chrono::steady_clock::now();

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    // Shutdown loops first
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    
    // Stop hardware interface
    if (hardware_interface_) {
        hardware_interface_->Stop();
    }
    
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::InitializeHardwareInterface()
{
    // Read hardware protocol from configuration
    std::string protocol_str = this->params.Get<std::string>("hardware_protocol", "free_dog_sdk");
    
    std::cout << LOGGER::INFO << "Initializing hardware interface: " << protocol_str << std::endl;
    
    if (protocol_str == "unitree_ros2") {
#ifdef UNITREE_ROS2_AVAILABLE
        hardware_protocol_ = HardwareProtocol::UNITREE_ROS2;
        hardware_interface_ = std::make_unique<HardwareInterfaceUnitreeRos2>(ros2_node);
        std::cout << LOGGER::INFO << "Using unitree_ros2 protocol" << std::endl;
#else
        std::cout << LOGGER::WARNING << "unitree_ros2 not available, falling back to free_dog_sdk" << std::endl;
        hardware_protocol_ = HardwareProtocol::FREE_DOG_SDK;
        hardware_interface_ = std::make_unique<HardwareInterfaceFreeDogSdk>();
#endif
    } else {
        // Default to free_dog_sdk
        hardware_protocol_ = HardwareProtocol::FREE_DOG_SDK;
        hardware_interface_ = std::make_unique<HardwareInterfaceFreeDogSdk>();
        std::cout << LOGGER::INFO << "Using free_dog_sdk protocol" << std::endl;
    }
    
    // Initialize and start the hardware interface
    if (!hardware_interface_->Initialize()) {
        std::cerr << LOGGER::ERROR << "Failed to initialize hardware interface!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    hardware_interface_->Start();
    
    // Wait for hardware interface to be ready
    std::cout << LOGGER::INFO << "Waiting for hardware interface to be ready..." << std::endl;
    int retry_count = 0;
    while (!hardware_interface_->IsReady() && retry_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        retry_count++;
    }
    
    if (hardware_interface_->IsReady()) {
        std::cout << LOGGER::INFO << "Hardware interface ready!" << std::endl;
    } else {
        std::cerr << LOGGER::ERROR << "Hardware interface failed to become ready!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void RL_Real::GetState(RobotState<float> *state)
{
#if defined(CONTROL_EXTRA) && defined(USE_ROS)
    // Handle control input messages
    this->control.x = this->control_input.ly;
    this->control.y = -this->control_input.lx;
    this->control.yaw = -this->control_input.rx;
    static int last_command_ = 0;
    if (last_command_ != this->control_input.command)
    {
        last_command_ = this->control_input.command;
        std::cout << LOGGER::INFO << "Control command changed to: " << last_command_ << std::endl;

        int cmd = last_command_;
        if (cmd == 1)
            this->control.current_keyboard = Input::Keyboard::Num9;
        else if (cmd == 2)
            this->control.current_keyboard = Input::Keyboard::Num0;
        else if (cmd == 3)
            this->control.current_keyboard = Input::Keyboard::Num1;
        else if (cmd == 4)
            this->control.current_keyboard = Input::Keyboard::Num2;

        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
    }
#endif

    // Get state from hardware interface
    std::vector<float> joint_positions, joint_velocities, joint_efforts;
    std::vector<float> imu_quaternion, imu_gyroscope, imu_accelerometer;
    
    if (!hardware_interface_->GetState(
        joint_positions, joint_velocities, joint_efforts,
        imu_quaternion, imu_gyroscope, imu_accelerometer))
    {
        // No data available or interface not ready
        return;
    }
    
    // Copy IMU data
    for (int i = 0; i < 4 && i < (int)imu_quaternion.size(); ++i) {
        state->imu.quaternion[i] = imu_quaternion[i];
    }
    for (int i = 0; i < 3 && i < (int)imu_gyroscope.size(); ++i) {
        state->imu.gyroscope[i] = imu_gyroscope[i];
    }
    
    // Copy motor state with joint mapping
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        if (motor_idx < (int)joint_positions.size()) {
            state->motor_state.q[i] = joint_positions[motor_idx];
            state->motor_state.dq[i] = joint_velocities[motor_idx];
            state->motor_state.tau_est[i] = joint_efforts[motor_idx];
        }
    }
}

void RL_Real::PrintDebugInfo(const RobotCommand<float> *command)
{
    // Print IMU state
    std::cout << "IMU State - Quaternion: [" 
              << this->robot_state.imu.quaternion[0] << ", " 
              << this->robot_state.imu.quaternion[1] << ", " 
              << this->robot_state.imu.quaternion[2] << ", " 
              << this->robot_state.imu.quaternion[3] << "] Gyroscope: [" 
              << this->robot_state.imu.gyroscope[0] << ", " 
              << this->robot_state.imu.gyroscope[1] << ", " 
              << this->robot_state.imu.gyroscope[2] << "]" << std::endl;
    
    // Print motor states
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        std::cout << "Motor " << i << " (idx " << motor_idx << ") - q: " 
                  << this->robot_state.motor_state.q[i] << " dq: " 
                  << this->robot_state.motor_state.dq[i] << " tau_est: " 
                  << this->robot_state.motor_state.tau_est[i] << std::endl;
    }
    
    // Print motor commands
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        std::cout << "Motor Command " << i << " (idx " << motor_idx << ") - q: " 
                  << command->motor_command.q[i] << " dq: " 
                  << command->motor_command.dq[i] << " tau: " 
                  << command->motor_command.tau[i] << " Kp: " 
                  << command->motor_command.kp[i] << " Kd: " 
                  << command->motor_command.kd[i] << std::endl;
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    // Print all states and commands every 1 second
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->last_print_time);
    if (elapsed.count() >= 1000)
    {
        this->PrintDebugInfo(command);
        this->last_print_time = now;
    }

    // Prepare command vectors with joint mapping
    std::vector<float> joint_positions(12, 0.0f);
    std::vector<float> joint_velocities(12, 0.0f);
    std::vector<float> joint_torques(12, 0.0f);
    std::vector<float> joint_kp(12, 0.0f);
    std::vector<float> joint_kd(12, 0.0f);
    
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        joint_positions[motor_idx] = command->motor_command.q[i];
        joint_velocities[motor_idx] = command->motor_command.dq[i];
        joint_torques[motor_idx] = command->motor_command.tau[i];
        joint_kp[motor_idx] = command->motor_command.kp[i];
        joint_kd[motor_idx] = command->motor_command.kd[i];
    }
    
    // Send command via hardware interface
    hardware_interface_->SetCommand(
        joint_positions, joint_velocities, joint_torques,
        joint_kp, joint_kd);
}

void RL_Real::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
#if defined(USE_ROS2) && defined(USE_ROS)
        if (buffer_.readFromRT() == nullptr || twist_count <= 0)
        {
            this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        }
        else
        {
            const geometry_msgs::msg::Twist twist = *buffer_.readFromRT();
            std::cout << std::endl << "twist: " << twist.linear.x << ", " << twist.linear.y << ", " << twist.angular.z << std::endl;
            this->obs.commands = {(float)twist.linear.x, (float)twist.linear.y, (float)twist.angular.z};
            twist_count--;
            if (twist_count <= 0)
            {
                buffer_.reset();
            }
        }
#else
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
#endif
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

#ifdef PLOT
void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        
        // Use robot_state and robot_command instead of fdsc members
        this->plot_real_joint_pos[i].push_back(this->robot_state.motor_state.q[i]);
        this->plot_target_joint_pos[i].push_back(this->robot_command.motor_command.q[i]);
        
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}
#endif

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
#if defined(USE_ROS2) && defined(USE_ROS)
    buffer_.writeFromNonRT(*msg);
    twist_count = 200 / 5.0;
#endif
    this->cmd_vel = *msg;
}
#endif

#if defined(CONTROL_EXTRA) && defined(USE_ROS)
void RL_Real::ControlInputsCallback(const control_input_msgs::msg::Inputs::SharedPtr msg)
{
    this->control_input = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

// Global flag for graceful shutdown
std::atomic<bool> shutdown_requested(false);

void signalHandler(int signum)
{
    shutdown_requested = true;
}

int main(int argc, char **argv)
{
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

#if defined(USE_ROS1) && defined(USE_ROS)
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar(argc, argv);
    while (!shutdown_requested && ros::ok())
    {
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ros::shutdown();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    while (!shutdown_requested && rclcpp::ok())
    {
        rclcpp::spin_some(rl_sar->ros2_node);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    RL_Real rl_sar(argc, argv);
    while (!shutdown_requested)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
    return 0;
}