#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

using FxCliMap = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

/// @brief Co-simulation TCP client for W4 MCUs.
/// 실제로는 127.0.0.1:6000(front) / 127.0.0.1:6001(rear)에 붙는다.
class FxCli {
public:
    FxCli(const std::string& ip_addr, uint16_t port);

    bool motor_start (const std::vector<uint8_t>& ids);
    bool motor_stop  (const std::vector<uint8_t>& ids);
    bool motor_estop (const std::vector<uint8_t>& ids);

    bool operation_control(const std::vector<uint8_t>& ids,
                           const std::vector<float>& pos,
                           const std::vector<float>& vel,
                           const std::vector<float>& kp,
                           const std::vector<float>& kd,
                           const std::vector<float>& tau);

    FxCliMap req(const std::vector<uint8_t> &ids);
    FxCliMap status();

private:
    // 실제 접속에 쓰는 주소/포트
    std::string ip_addr_;   ///< 항상 "127.0.0.1"
    uint16_t    port_;      ///< front: 6000, rear: 6001

    bool        is_front_;  ///< 이 인스턴스가 front MCU 인지 여부

    // front / rear 전역 소켓 (프로세스 전체에서 공유)
    static int front_sock_fd_;
    static int rear_sock_fd_;

    /// @brief front / rear 중 해당 side 의 소켓 fd 참조 반환
    int& socket_ref();

    /// @brief 아직 연결되지 않았다면 127.0.0.1:port_ 로 연결 (무한 재시도)
    bool connect_if_needed();

    /// @brief 문자열 전체를 소켓으로 전송 (개행은 호출자가 붙임)
    bool send_all(const std::string& s);

    /// @brief 한 줄(\\n 까지)을 읽어서 out 에 저장
    bool read_line(std::string& out);
};
