/**                                                                                       
 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ▓██████████████        
 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ████████▒▒▓█  ▓███▓▓   
 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ▒█     ██████████████▒  
           ░          ░                          ░░░░░          ░░░ ██  ░   ████     ██     
   ██████     ██████     ██████   █████████ ███  ░░░   ░██████  ░░░  █     ██  ██   ██   ░░ 
  ███   ███  ██    ███  ███  ▓██  ███       ███  ░░░  ███   ███  ░  ██▒    ██   ███  █  ░░░ 
 ░██        ▒██    ██▒ ███        ███       ███  ░░░  ██    ▒██  ░  ████    █  ████  █    ░ 
 ░██        ░██  ░ ██▒ ▓██        █████████ ███  ░░░  ██░ ░ ▓██  ░  ███     █   ███  ███    
  ██    ███ ▓██    ██▓ ▒██    ███ ███       ███       ██     ██  ░░     ░ ███        ████   
  ████████   ████████   ████████  █████████ █████████ █████████  ░░░░░░░░ █████  ░░ ▓████   
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ░███   ░░         
███████████████████████████████████████████████████████████████████████████████████████████
*/


#pragma once  // robot.hpp

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <fx_cli/fx_client.h> 
#include "mcu.hpp"   
#include "safety.hpp"

namespace robot {

// ════════════════════════════════════════════════════════════════════════════
// Motor Configuration 
// ════════════════════════════════════════════════════════════════════════════

inline const std::vector<uint8_t>    MOTOR_IDS           = {1u, 2u,  3u,  4u,  5u,  6u,  7u,  8u,     ///< All registerd motor IDs 
                                                            9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};
constexpr std::size_t                NUM_MOTORS          = 16;                                        ///< Total number of motors
constexpr std::size_t                NUM_LIMB_MOTORS     = 12;                                        ///< Number of limb  motors (excluding wheels)
constexpr std::array<std::size_t, 4> WHEEL_MIDS          = {7, 8, 15, 16};                            ///< Motor IDs for wheel actuators

// ════════════════════════════════════════════════════════════════════════════
// MCU Configuration
// ════════════════════════════════════════════════════════════════════════════

inline const std::vector<uint8_t> FRONT_MCU_MOTOR_IDS  = {1u, 2u,  3u,  4u,  5u,  6u,  7u,  8u};  
inline const std::string          FRONT_MCU_IP         = "192.168.10.10";
constexpr int                     FRONT_MCU_PORT       = 5101;
constexpr bool                    FRONT_MCU_HAS_IMU    = false; 
constexpr bool                    FRONT_MCU_HAS_BAT    = false; 
constexpr bool                    FRONT_MCU_HAS_ESTOP  = false; 

inline const std::vector<uint8_t> REAR_MCU_MOTOR_IDS   = {9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};    
inline const std::string          REAR_MCU_IP          = "192.168.11.10";
constexpr int                     REAR_MCU_PORT        = 5101;
constexpr bool                    REAR_MCU_HAS_IMU     = true; 
constexpr bool                    REAR_MCU_HAS_BAT     = true; 
constexpr bool                    REAR_MCU_HAS_ESTOP   = true; 

// ──────────────────────────────────────────────────────────────────────────
// Joint Index Mapping for Observation and Action Vectors
// ──────────────────────────────────────────────────────────────────────────

constexpr std::size_t FRONT_LEFT_HIP_IDX       = 0;   ///< Front Left hip joint index
constexpr std::size_t FRONT_RIGHT_HIP_IDX      = 1;   ///< Front Right hip joint index
constexpr std::size_t REAR_LEFT_HIP_IDX        = 2;   ///< Rear Left hip joint index
constexpr std::size_t REAR_RIGHT_HIP_IDX       = 3;   ///< Rear Right hip joint index
constexpr std::size_t FRONT_LEFT_SHOULDER_IDX  = 4;   ///< Front Left shoulder joint index
constexpr std::size_t FRONT_RIGHT_SHOULDER_IDX = 5;   ///< Front Right shoulder joint index
constexpr std::size_t REAR_LEFT_SHOULDER_IDX   = 6;   ///< Rear Left shoulder joint index
constexpr std::size_t REAR_RIGHT_SHOULDER_IDX  = 7;   ///< Rear Right shoulder joint index
constexpr std::size_t FRONT_LEFT_LEG_IDX       = 8;   ///< Front Left leg joint index
constexpr std::size_t FRONT_RIGHT_LEG_IDX      = 9;   ///< Front Right leg joint index
constexpr std::size_t REAR_LEFT_LEG_IDX        = 10;  ///< Rear Left leg joint index
constexpr std::size_t REAR_RIGHT_LEG_IDX       = 11;  ///< Rear Right leg joint index
constexpr std::size_t FRONT_LEFT_WHEEL_IDX     = 12;  ///< Front Left wheel motor index
constexpr std::size_t FRONT_RIGHT_WHEEL_IDX    = 13;  ///< Front Right wheel motor index
constexpr std::size_t REAR_LEFT_WHEEL_IDX      = 14;  ///< Rear Left wheel motor index
constexpr std::size_t REAR_RIGHT_WHEEL_IDX     = 15;  ///< Rear Right wheel motor index

// ──────────────────────────────────────────────────────────────────────────
// Motor ID Mapping Tables (MID → Joint Index)
// ──────────────────────────────────────────────────────────────────────────

inline constexpr std::array<std::size_t, NUM_MOTORS + 1> MID_TO_OBS_IDX = {
    0,                         // [0]  unused                              OBS IDX
    FRONT_LEFT_HIP_IDX,        // [1]  Motor ID 1  → front left hip        (= 0  )
    FRONT_RIGHT_HIP_IDX,       // [2]  Motor ID 2  → front right hip       (= 1  )
    FRONT_LEFT_SHOULDER_IDX,   // [3]  Motor ID 3  → front left shoulder   (= 4  )
    FRONT_RIGHT_SHOULDER_IDX,  // [4]  Motor ID 4  → front right shoulder  (= 5  )
    FRONT_LEFT_LEG_IDX,        // [5]  Motor ID 5  → front left leg        (= 8  )
    FRONT_RIGHT_LEG_IDX,       // [6]  Motor ID 6  → front right leg       (= 9  )
    FRONT_LEFT_WHEEL_IDX,      // [7]  Motor ID 7  → front left wheel      (= 12 )
    FRONT_RIGHT_WHEEL_IDX,     // [8]  Motor ID 8  → front right wheel     (= 13 )
    REAR_LEFT_HIP_IDX,         // [9]  Motor ID 9  → rear left hip         (= 2  )
    REAR_RIGHT_HIP_IDX,        // [10] Motor ID 10 → rear right hip        (= 3  )
    REAR_LEFT_SHOULDER_IDX,    // [11] Motor ID 11 → rear left shoulder    (= 6  )
    REAR_RIGHT_SHOULDER_IDX,   // [12] Motor ID 12 → rear right shoulder   (= 7  )
    REAR_LEFT_LEG_IDX,         // [13] Motor ID 13 → rear left leg         (= 10 )
    REAR_RIGHT_LEG_IDX,        // [14] Motor ID 14 → rear right leg        (= 11 )
    REAR_LEFT_WHEEL_IDX,       // [15] Motor ID 15 → rear left wheel       (= 14 )
    REAR_RIGHT_WHEEL_IDX       // [16] Motor ID 16 → rear right wheel      (= 15 )
};

/// @brief Maps motor IDs to human-readable joint names
inline constexpr std::array<const char*, NUM_MOTORS + 1> MID_TO_JOINT_NAMES = {
    "unused",               // [0]  unused
    "front_left_hip",       // [1]  Motor ID 1  → "front_left_hip"
    "front_right_hip",      // [2]  Motor ID 2  → "front_right_hip"
    "front_left_shoulder",  // [3]  Motor ID 3  → "front_left_shoulder"
    "front_right_shoulder", // [4]  Motor ID 4  → "front_right_shoulder"
    "front_left_leg",       // [5]  Motor ID 5  → "front_left_leg"
    "front_right_leg",      // [6]  Motor ID 6  → "front_right_leg"
    "front_left_wheel",     // [7]  Motor ID 7  → "front_left_wheel"
    "front_right_wheel",    // [8]  Motor ID 8  → "front_right_wheel"

    "rear_left_hip",        // [9]  Motor ID 9  → "rear_left_hip"
    "rear_right_hip",       // [10] Motor ID 10 → "rear_right_hip"
    "rear_left_shoulder",   // [11] Motor ID 11 → "rear_left_shoulder"
    "rear_right_shoulder",  // [12] Motor ID 12 → "rear_right_shoulder"
    "rear_left_leg",        // [13] Motor ID 13 → "rear_left_leg"
    "rear_right_leg",       // [14] Motor ID 14 → "rear_right_leg"
    "rear_left_wheel",      // [15] Motor ID 15 → "rear_left_wheel"
    "rear_right_wheel"      // [16] Motor ID 16 → "rear_right_wheel"
};


// ──────────────────────────────────────────────────────────────────────────
// Joint Calibration
// ──────────────────────────────────────────────────────────────────────────

/**
 * @brief Position offset calibration values for each joint
 * @note Wheels are not included as they don't require position calibration
 */
inline std::unordered_map<std::string, float> POS_OFFSET = {
    {"front_left_hip",       0.0f}, 
    {"front_right_hip",      0.0f},
    {"front_left_shoulder",  0.0f}, 
    {"front_right_shoulder", 0.0f},
    {"front_left_leg",       0.0f}, 
    {"front_right_leg",      0.0f},

    {"rear_left_hip",        0.0f}, 
    {"rear_right_hip",       0.0f},
    {"rear_left_shoulder",   0.0f}, 
    {"rear_right_shoulder",  0.0f},
    {"rear_left_leg",        0.0f}, 
    {"rear_right_leg",       0.0f},
};

// ════════════════════════════════════════════════════════════════════════════
// Height-map Configuration Constants
// ════════════════════════════════════════════════════════════════════════════

constexpr std::size_t HEIGHT_MAP_SIZE = 144;  

/// @brief Interval for printing repeated warnings (in control cycles)
constexpr std::int32_t WARNING_PRINT_INTERVAL = 200;

// ════════════════════════════════════════════════════════════════════════════
// Exception Types
// ════════════════════════════════════════════════════════════════════════════

/// @brief Exception thrown when emergency stop is triggered
struct RobotEStopError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

/// @brief Exception thrown during robot initialization failures
struct RobotInitError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

/// @brief Exception thrown when gain configuration fails
struct RobotSetGainsError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

// ════════════════════════════════════════════════════════════════════════════
// Robot Control Interface
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class Robot
 * @brief High-level interface for robot motor control and state observation
 * 
 * @details This class provides a safe, high-level API for controlling a robot
 *          with 16 motors (12 limb joints + 4 wheels). It handles:
 *          - Motor gain configuration (PD control)
 *          - State observation (positions, velocities, etc.)
 *          - Safety monitoring and emergency stop
 *          - Communication with motor controllers via FxCli
 */
class Robot {
public:
    /**
     * @brief Constructs and initializes the robot interface
     * @throws RobotInitError if initialization fails
     */
    Robot();

    // ───────────────────────────────────────────────────────────────────────
    // Control Configuration
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Sets PD controller gains for all motors
     * @param kp Proportional gains (size must equal NUM_MOTORS)
     * @param kd Derivative gains (size must equal NUM_MOTORS)
     * @throws RobotSetGainsError if gain configuration fails
     * @note Must be called before do_action() can be used
     */
    void setGains(const std::vector<float>& kp, const std::vector<float>& kd);

    // ───────────────────────────────────────────────────────────────────────
    // State Observation
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Retrieves current robot state observation
     * @return Map containing observation vectors (positions, velocities, etc.)
     * @note Returns a copy; internal buffers are reused for efficiency
     * @throws RobotEStopError if safety checks fail
     */
    std::unordered_map<std::string, std::vector<float>> getObs();

    // ───────────────────────────────────────────────────────────────────────
    // Safety Monitoring
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Performs safety checks on robot state
     * @throws RobotEStopError if any safety condition is violated
     * @details Checks for:
     *          - Communication timeouts
     *          - Disconnection counts
     *          - Hardware errors
     */
    void checkSafety();

    // ───────────────────────────────────────────────────────────────────────
    // Motor Control
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Executes a motor control action
     * @param action Motor commands (size must equal NUM_MOTORS)
     * @param torque_ctrl true for torque control, false for position control
     * @param safe If true, performs safety checks after executing
     * @throws RobotEStopError if safety checks fail (when safe=true)
     * @note Requires set_gains() to be called first
     */
    void doAction(const std::vector<float>& action, bool torque_ctrl, bool safe);

    // ───────────────────────────────────────────────────────────────────────
    // Emergency Control & Diagnostics
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Triggers emergency stop and terminates program
     * @param msg Error message describing the reason for E-stop
     */
    [[noreturn]] void estop(const std::string& msg, bool is_physical_estop=false);

    /**
     * @brief Logs a warning message with throttling
     * @param msg Warning message to log
     */
    void warn(const std::string& msg);

    // ───────────────────────────────────────────────────────────────────────
    // Python Interface Accessors
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Gets current PD controller gains
     * @return Pair of vectors: {kp, kd}
     */
    inline std::pair<std::vector<float>, std::vector<float>> getGains() const {
        return {kp_, kd_};
    }

    /**
     * @brief Checks if gains have been configured
     * @return true if set_gains() has been successfully called
     */
    inline bool gainsReady() const {
        return gains_set_;
    }

private:
    // ───────────────────────────────────────────────────────────────────────
    // Internal Utilities
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Initializes and waits for MCU to be ready for operation
     * @param mcu MCU client to initialize (contains motor IDs and capability flags)
     * @param timeout_ms Maximum wait time in milliseconds (default: 10000ms)
     * @throws RobotInitError if timeout is exceeded or initialization fails
     * @details Performs the following steps:
     *          1. Starts all motors on the MCU
     *          2. Checks motor status for errors
     *          3. Validates emergency stop button (if MCU has_estop)
     *          4. Checks battery level (if MCU has_battery)
     *          5. Gets initial observation data
     *          6. Initializes last_action with current joint positions
     */
    void waitMcu(mcu::McuClient& mcu, std::int32_t timeout_ms=10000);

    /**
     * @brief Parses and updates robot observation state from MCU data
     * @param m MCU client being updated
     * @details Extracts the following data:
     *          - Motor positions and velocities for limb joints (→ dof_pos, dof_vel)
     *          - Motor velocities for wheel motors (→ dof_vel)
     *          - IMU data: angular velocity and projected gravity (if mcu.has_imu)
     */
   void updateMcuObs(mcu::McuClient& m);

    /**
     * @brief Checks if a motor ID corresponds to a limb joint
     * @param mid Motor ID to check
     * @return true if motor is a limb joint (not a wheel)
     */
    bool isLimbMid(std::size_t mid) { return std::find(WHEEL_MIDS.begin(), WHEEL_MIDS.end(), mid) == WHEEL_MIDS.end();}


    std::string getMotorStatusString() const;
    // ───────────────────────────────────────────────────────────────────────
    // State Variables
    // ───────────────────────────────────────────────────────────────────────

    std::vector<float> last_action_;     ///< Last commanded action (for monitoring)
    bool last_torque_ctrl_;              ///< Last control mode used

    // Control gains
    std::vector<float> kp_;              ///< Proportional gains
    std::vector<float> kd_;              ///< Derivative gains
    bool gains_set_;                     ///< Flag indicating if gains are configured

    // Motor configuration
    std::vector<std::string> motor_pattern_;  ///< Motor connection patterns
    std::vector<std::string> motor_err_;      ///< Motor error states

    // Battery monitoring
    std::string battery_voltage_;        ///< Current battery voltage
    std::string battery_soc_;            ///< Battery state of charge (%)

    // Observation data
    std::unordered_map<std::string, std::vector<float>> obs_;  ///< Current observation

    // Hardware interface
    mcu::McuClient front_mcu_;
    mcu::McuClient rear_mcu_;
};

} // namespace robot
