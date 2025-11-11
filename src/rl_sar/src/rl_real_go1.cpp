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
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
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

    // init free_dog_sdk connection
    std::string connection_settings = "LOW_WIRED_DEFAULTS";  // Default to wired connection
    this->fdsc_conn = std::make_shared<FDSC::UnitreeConnection>(connection_settings);
    this->fdsc_conn->startRecv();
    
    // Initialize print timing
    this->last_print_time = std::chrono::steady_clock::now();
    
    // Send initial command to establish connection
    std::vector<uint8_t> init_cmd = this->fdsc_low_command.buildCmd(false);
    this->fdsc_conn->send(init_cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_udpSend = std::make_shared<LoopFunc>("loop_udpSend", 0.002, std::bind(&RL_Real::UDPSend, this), 3);
    this->loop_udpRecv = std::make_shared<LoopFunc>("loop_udpRecv", 0.002, std::bind(&RL_Real::UDPRecv, this), 3);
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_udpSend->start();
    this->loop_udpRecv->start();
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
    this->loop_udpSend->shutdown();
    this->loop_udpRecv->shutdown();
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    
    // Stop free_dog_sdk connection and wait for threads to finish
    if (this->fdsc_conn)
    {
        this->fdsc_conn->stopRecv();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        this->fdsc_conn.reset();
    }
    
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::UDPSend()
{
    std::vector<uint8_t> cmd_bytes = this->fdsc_low_command.buildCmd(false);
    this->fdsc_conn->send(cmd_bytes);
}

void RL_Real::UDPRecv()
{
    std::vector<std::vector<uint8_t>> dataall;
    this->fdsc_conn->getData(dataall);
    
    if (!dataall.empty())
    {
        std::lock_guard<std::mutex> lock(this->fdsc_data_mutex_);
        // Replace buffer with latest data to avoid accumulation
        this->fdsc_data_buffer = std::move(dataall);
    }
}

void RL_Real::GetState(RobotState<float> *state)
{
    // Get latest data from buffer with mutex protection
    std::vector<uint8_t> latest_data;
    {
        std::lock_guard<std::mutex> lock(this->fdsc_data_mutex_);
        if (!this->fdsc_data_buffer.empty())
        {
            latest_data = this->fdsc_data_buffer.back();
            this->fdsc_data_buffer.clear();
        }
    }
    
#if defined(CONTROL_EXTRA) && defined(USE_ROS)
    this->control.x = this->control_input.ly;
    this->control.y = -this->control_input.lx;
    this->control.yaw = -this->control_input.rx;
    if (last_command_ != this->control_input.command)
    {
        last_command_ = this->control_input.command;
        std::cout << LOGGER::INFO << "Control command changed to: " << last_command_ << std::endl;

        // int cmd = std::max(0, last_command_ - 2);
        int cmd = last_command_;
        if (cmd == 1)
            this->control.current_keyboard = Input::Keyboard::Num9;
        else if (cmd == 2)
            this->control.current_keyboard = Input::Keyboard::Num0;
        else if (cmd == 3)
            this->control.current_keyboard = Input::Keyboard::Num1;

        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
    }
#endif

    // Parse data outside of mutex lock
    if (!latest_data.empty())
    {
        // Check data length before parsing (minimum expected size for LowState)
        const size_t min_expected_size = 100;  // Adjust based on FDSC::lowState structure
        if (latest_data.size() >= min_expected_size)
        {
            try
            {
                this->fdsc_low_state.parseData(latest_data);
            }
            catch (const std::exception& e)
            {
                std::cout << LOGGER::ERROR << "Failed to parse low state data: " << e.what() << std::endl;
                return;  // Early return on parse error
            }
        }
        else
        {
            std::cout << LOGGER::WARNING << "Received incomplete low state data packet, size: " 
                      << latest_data.size() << " (expected >= " << min_expected_size << ")" << std::endl;
            return;  // Early return on incomplete data
        }
    }
    else
    {
        // No data available, skip state update
        return;
    }

    // Parse joystick input
    if (this->fdsc_low_state.wirelessdata.btn.components.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->fdsc_low_state.wirelessdata.btn.components.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->fdsc_low_state.wirelessdata.btn.components.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->fdsc_low_state.wirelessdata.btn.components.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->fdsc_low_state.wirelessdata.btn.components.F1) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.F2) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->fdsc_low_state.wirelessdata.btn.components.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->fdsc_low_state.wirelessdata.btn.components.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->fdsc_low_state.wirelessdata.btn.components.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.A) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.B) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.X) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.Y) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.F1) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.F2) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.up) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.down) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.left) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.right) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.A) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.B) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.X) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.Y) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.F1) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.F2) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.up) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.down) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.left) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->fdsc_low_state.wirelessdata.btn.components.R1 && this->fdsc_low_state.wirelessdata.btn.components.right) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->fdsc_low_state.wirelessdata.btn.components.L1 && this->fdsc_low_state.wirelessdata.btn.components.R1) this->control.SetGamepad(Input::Gamepad::LB_RB);

    // IMU data - free_dog_sdk uses [w, x, y, z] format
    state->imu.quaternion[0] = this->fdsc_low_state.imu_quaternion[0]; // w
    state->imu.quaternion[1] = this->fdsc_low_state.imu_quaternion[1]; // x
    state->imu.quaternion[2] = this->fdsc_low_state.imu_quaternion[2]; // y
    state->imu.quaternion[3] = this->fdsc_low_state.imu_quaternion[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->fdsc_low_state.imu_gyroscope[i];
    }
    
    // Motor state
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        state->motor_state.q[i] = this->fdsc_low_state.motorState[motor_idx].q;
        state->motor_state.dq[i] = this->fdsc_low_state.motorState[motor_idx].dq;
        state->motor_state.tau_est[i] = this->fdsc_low_state.motorState[motor_idx].tauEst;
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

    // Directly set motor commands like HardwareFreeDogSdk does
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        
        // Direct assignment to motor array (safer than setMotorCmd)
        this->fdsc_low_command.motorCmd.motors[motor_idx].mode = FDSC::MotorModeLow::Servo;
        this->fdsc_low_command.motorCmd.motors[motor_idx].q = static_cast<float>(command->motor_command.q[i]);
        this->fdsc_low_command.motorCmd.motors[motor_idx].dq = static_cast<float>(command->motor_command.dq[i]);
        this->fdsc_low_command.motorCmd.motors[motor_idx].tau = static_cast<float>(command->motor_command.tau[i]);
        this->fdsc_low_command.motorCmd.motors[motor_idx].Kp = static_cast<float>(command->motor_command.kp[i]);
        this->fdsc_low_command.motorCmd.motors[motor_idx].Kd = static_cast<float>(command->motor_command.kd[i]);
    }
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
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
#if !defined(USE_CMAKE) && defined(USE_ROS)
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};

        }
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
        
        int motor_idx = this->params.Get<std::vector<int>>("joint_mapping")[i];
        this->plot_real_joint_pos[i].push_back(this->fdsc_low_state.motorState[motor_idx].q);
        this->plot_target_joint_pos[i].push_back(this->fdsc_low_command.motorCmd.motors[motor_idx].q);
        
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
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