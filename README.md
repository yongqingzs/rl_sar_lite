# rl_sar_lite

[![ROS2 Jazzy](https://img.shields.io/badge/ros2-jazzy-brightgreen.svg?logo=ros)](https://docs.ros.org/en/jazzy/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.2.7-orange.svg?logo=mujoco)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

A lightweight reinforcement learning deployment framework for quadruped robots. This is a streamlined version of [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar) with **decoupled communication interfaces**, **enhanced ARM64 support** (RK3588, etc.), and **ROS2 Jazzy compatibility**.

> [!CAUTION]
> **Disclaimer: Users acknowledge that all risks and consequences arising from using this code shall be solely borne by the user. The author assumes no liability for any direct or indirect damages. Proper safety measures must be implemented prior to operation.**

## Features

- 🚀 **Decoupled Hardware Interfaces**: Modular communication protocol support
- 🦾 **ARM64 Optimized**: Native support for RK3588 and other ARM platforms
  - LibTorch ARM64 backend
  - RKNPU2 inference format support
- 🤖 **Multi-Robot Simulation**: MuJoCo-based simulation for 8+ quadruped robots
- 🔌 **Flexible Protocol Support**: `free_dog_sdk`, `unitree_sdk2`, `unitree_ros2`
- 🎮 **State Machine Control**: Multi-policy switching via keyboard/gamepad
- 🔧 **Easy Extension**: Simple interface for adding new robots/protocols

## Supported Robots

### Simulation (MuJoCo)
Powered by [unitree_mujoco](https://github.com/yongqingzs/unitree_mujoco):

- Unitree A1
- Unitree Aliengo
- Unitree Go1
- Xiaomi Cyberdog
- Deep Robotics Lite3
- Deep Robotics X30
- Anybotics Anymal B
- Anybotics Anymal C

### Sim-to-Real
Currently supports **Unitree quadrupeds** (Go1, Go2) with three communication protocols:

| Protocol | Description | Use Case |
|----------|-------------|----------|
| `free_dog_sdk` | Open-source lightweight SDK | Community development |
| `unitree_sdk2` | Official Unitree SDK v2 | Latest robots (Go2, etc.) |
| `unitree_ros2` | ROS2 native interface | ROS2 integration |

> **Note**: `unitree_mujoco` uses `unitree_sdk2` message types for simulation communication.

## ROS Compatibility

| ROS Version | Status | Notes |
|-------------|--------|-------|
| ROS1 Melodic | ✅ Supported | |
| ROS1 Noetic | ✅ Supported | |
| ROS2 Foxy | ✅ Supported | |
| ROS2 Galactic | ✅ Supported | |
| ROS2 Humble | ✅ Supported | |
| **ROS2 Jazzy** | ✅ **Recommended** | Primary development target |

> 💡 **Recommendation**: Use **ROS2 Jazzy** on Ubuntu 24.04 for the best experience.

## Installation

### Prerequisites

#### System Dependencies
```bash
sudo apt install cmake g++ build-essential libyaml-cpp-dev libeigen3-dev \
                 libboost-all-dev libspdlog-dev libfmt-dev libtbb-dev liblcm-dev
```

#### ROS2 Jazzy Dependencies
```bash
sudo apt install \
  ros-jazzy-teleop-twist-keyboard \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-control-toolbox \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-xacro \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-rosidl-generator-dds-idl

sudo apt install libopenblas-dev libopenblas0 liblapack-dev liblapack3
```

### Build from Source

#### 1. Install MuJoCo Simulator
```bash
mkdir -p ~/jazzy_ws/src && cd ~/jazzy_ws/src
git clone https://github.com/yongqingzs/unitree_mujoco.git
# Follow unitree_mujoco installation guide
```

#### 2. Install Keyboard Control (Optional: Gamepad Control)
```bash
cd ~/jazzy_ws/src
git clone https://github.com/yongqingzs/quadruped_ros2_control.git
MAKEFLAGS="-j4" colcon build --packages-up-to keyboard_input
```

#### 3. Build rl_sar_lite
```bash
cd ~
git clone https://github.com/yongqingzs/rl_sar_lite.git
git submodule update --init --recursive --recommend-shallow --progress

cd rl_sar_lite && ./build.sh -c
./build.sh
```

## Usage

### Simulation

**Terminal 1** - Launch MuJoCo simulator:
```bash
# Robot type configured in simulate/config.yaml
unitree_mujoco
```

**Terminal 2** - Keyboard control:
```bash
# Key mappings:
# 1: Lie down  2: Stand up  3: Policy 1  4: Policy 2  5: Policy 3
# Note: Must stand up (2) before switching policies
ros2 run keyboard_input keyboard_input
```

**Terminal 3** - RL controller:
```bash
ros2 run rl_sar rl_real_bridge
```

### Sim-to-Real

**Terminal 1** - Keyboard control:
```bash
ros2 run keyboard_input keyboard_input
```

**Terminal 2** - RL controller:
```bash
# Switch communication protocol in policy/bridge/base.yaml
# hardware_protocol: free_dog_sdk | unitree_sdk2 | unitree_ros2
ros2 run rl_sar rl_real_bridge
```

> **Note**: `rl_real_go1` is kept for version compatibility. Use `rl_real_bridge` for all sim/real testing.

## Architecture

### Hardware Interface Design

The project uses a **plugin-based architecture** for hardware communication:

```
library/core/hardware_interface/
├── hardware_interface_base.hpp          # Abstract base class
├── free_dog_sdk/                        # Protocol 1
│   ├── hardware_interface_free_dog_sdk.hpp
│   └── hardware_interface_free_dog_sdk.cpp
├── unitree_sdk2/                        # Protocol 2
│   ├── hardware_interface_unitree_sdk2.hpp
│   └── hardware_interface_unitree_sdk2.cpp
└── unitree_ros2/                        # Protocol 3
    ├── hardware_interface_unitree_ros2.hpp
    └── hardware_interface_unitree_ros2.cpp
```

**Key Components**:
- `hardware_interface_base.hpp`: Defines the interface contract (`init()`, `send()`, `receive()`)
- Protocol subdirectories: Isolated implementations for each communication method
- `rl_real_bridge`: Unified entry point that dynamically selects protocol at runtime

### Adding New Robots/Protocols

To add support for a new robot or protocol:

1. **Create protocol directory**:
   ```bash
   mkdir library/core/hardware_interface/new_protocol
   ```

2. **Implement interface** in `new_protocol/hardware_interface_new_protocol.cpp`:
   ```cpp
   class HardwareInterfaceNewProtocol : public HardwareInterfaceBase {
   public:
       void init() override;
       void send(const RobotCmd& cmd) override;
       RobotState receive() override;
   };
   ```

3. **Register in bridge** (`src/rl_real_bridge.cpp`):
   ```cpp
   #ifdef USE_NEW_PROTOCOL
   hardware_interface_ = std::make_shared<HardwareInterfaceNewProtocol>();
   #endif
   ```

4. **Update CMakeLists.txt**:
   ```cmake
   option(USE_NEW_PROTOCOL "Use new protocol" ON)
   if(USE_NEW_PROTOCOL)
       add_subdirectory(library/core/hardware_interface/new_protocol)
   endif()
   ```

The modular design ensures **zero coupling** between different protocols - adding or removing one doesn't affect others.

## Configuration

Before running, prepare:

1. **Copy trained policy** to `policy/<robot>/<config>/`:
   ```
   policy/
   └── bridge/
       ├── base.yaml              # Hardware/runtime config
       └── <your_policy>/
           ├── config.yaml        # Policy parameters
           └── model.pt           # PyTorch model
   ```

2. **Configure parameters**:
   - `base.yaml`: Communication protocol, motor limits, control frequency
   - `config.yaml`: Observation/action dimensions, normalization params

## Command Reference

### Build Options
```bash
./build.sh              # Build all ROS2 packages
./build.sh -c           # Clean workspace
./build.sh -m           # CMake-only build (no ROS)
./build.sh -mj          # Build with MuJoCo support(Not recommended, use unitree_mujoco instead)
```

### Control Keys

| Key | Function |
|-----|----------|
| `1` | Lie down (damping mode) |
| `2` | Stand up to default pose |
| `3-5` | Switch to Policy 1/2/3 |
| `W/S` | Forward/Backward |
| `A/D` | Strafe Left/Right |
| `Q/E` | Rotate CCW/CW |
| `Space` | Stop (reset velocities) |

## Troubleshooting

**Issue**: `unitree_mujoco` cannot find robot model  
**Solution**: Check `simulate/config.yaml` for correct robot name

**Issue**: Real robot not responding  
**Solution**: Verify `hardware_protocol` in `policy/bridge/base.yaml` matches your setup

**Issue**: Build fails with protocol errors  
**Solution**: Ensure submodules are updated: `git submodule update --init --recursive`

## Contributing

Contributions are welcome! Please ensure:
- New protocols follow the `HardwareInterfaceBase` interface
- Code is tested in both simulation and real hardware (if applicable)
- Documentation is updated accordingly

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details

## Acknowledgments

- [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar) - Original framework
- [legubiao/quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control.git) - Keyboard/gamepad control
- [unitree_mujoco](https://github.com/yongqingzs/unitree_mujoco) - MuJoCo simulation backend
- Unitree Robotics - Hardware support and SDKs
