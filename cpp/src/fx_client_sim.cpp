#include "fx_client_sim.hpp"

// POSIX networking headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cctype>
#include <vector>
#include <algorithm>

// 전역 소켓
int FxCli::front_sock_fd_ = -1;
int FxCli::rear_sock_fd_  = -1;

// ────────────────────────────────────────────────────────────────
// 아주 단순한 JSON 파서 (두 단계 map만 지원)
// ────────────────────────────────────────────────────────────────
class JsonParser {
public:
    explicit JsonParser(const std::string& s) : src(s), idx(0) {}

    FxCliMap parse() {
        FxCliMap result;
        skip_ws();
        if (!consume('{')) {
            return result;
        }

        skip_ws();
        if (peek() == '}') {
            consume('}');
            return result;
        }

        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) {
                break;
            }

            skip_ws();
            if (!consume(':')) {
                break;
            }

            skip_ws();
            if (peek() == '{') {
                std::unordered_map<std::string, std::string> inner;
                if (!parse_object(inner)) {
                    break;
                }
                result.emplace(std::move(key), std::move(inner));
            } else {
                std::string val;
                if (!parse_value(val)) {
                    break;
                }
                std::unordered_map<std::string, std::string> inner;
                inner["value"] = std::move(val);
                result.emplace(std::move(key), std::move(inner));
            }

            skip_ws();
            if (!consume(',')) {
                break;
            }
        }

        skip_ws();
        consume('}');
        return result;
    }

private:
    const std::string& src;
    size_t idx;

    char peek() const {
        if (idx >= src.size()) return '\0';
        return src[idx];
    }

    char get() {
        if (idx >= src.size()) return '\0';
        return src[idx++];
    }

    void skip_ws() {
        while (idx < src.size() &&
               std::isspace(static_cast<unsigned char>(src[idx]))) {
            idx++;
        }
    }

    bool consume(char ch) {
        if (peek() == ch) {
            ++idx;
            return true;
        }
        return false;
    }

    bool parse_string(std::string& out) {
        out.clear();
        skip_ws();
        if (!consume('"')) {
            return false;
        }
        std::ostringstream oss;
        while (true) {
            char c = get();
            if (c == '\0') {
                return false;
            }
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                char e = get();
                if      (e == '"')  oss << '"';
                else if (e == '\\') oss << '\\';
                else if (e == '/')  oss << '/';
                else if (e == 'b')  oss << '\b';
                else if (e == 'f')  oss << '\f';
                else if (e == 'n')  oss << '\n';
                else if (e == 'r')  oss << '\r';
                else if (e == 't')  oss << '\t';
                else                 oss << e;
            } else {
                oss << c;
            }
        }
        out = oss.str();
        return true;
    }

    bool parse_value(std::string& out) {
        skip_ws();
        out.clear();
        if (peek() == '"') {
            return parse_string(out);
        }

        while (idx < src.size()) {
            char c = src[idx];
            if (c == ',' || c == '}' ||
                std::isspace(static_cast<unsigned char>(c))) {
                break;
            }
            out.push_back(c);
            ++idx;
        }

        while (!out.empty() &&
               std::isspace(static_cast<unsigned char>(out.back()))) {
            out.pop_back();
        }
        return !out.empty();
    }

    bool parse_object(std::unordered_map<std::string, std::string>& m) {
        if (!consume('{')) {
            return false;
        }
        skip_ws();
        if (peek() == '}') {
            consume('}');
            return true;
        }

        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) {
                return false;
            }
            skip_ws();
            if (!consume(':')) {
                return false;
            }
            skip_ws();
            std::string val;
            if (!parse_value(val)) {
                return false;
            }
            m.emplace(std::move(key), std::move(val));
            skip_ws();
            if (!consume(',')) {
                break;
            }
        }
        skip_ws();
        if (!consume('}')) {
            return false;
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────
// FxCli 구현
// ────────────────────────────────────────────────────────────────

FxCli::FxCli(const std::string& ip_addr, uint16_t port)
    : ip_addr_("127.0.0.1")
    , port_(6000)
    , is_front_(true)
{
    if (port == 6000) {
        is_front_ = true;
        port_ = 6000;
    } else if (port == 6001) {
        is_front_ = false;
        port_ = 6001;
    } else {
        if (ip_addr.find("192.168.10.") != std::string::npos) {
            is_front_ = true;
            port_ = 6000;
        } else {
            is_front_ = false;
            port_ = 6001;
        }
    }
}

int& FxCli::socket_ref() {
    return is_front_ ? front_sock_fd_ : rear_sock_fd_;
}

bool FxCli::connect_if_needed() {
    int& sock_fd = socket_ref();
    if (sock_fd != -1) {
        return true;
    }

    int attempt = 0;

    while (true) {
        ++attempt;

        std::cout << "[FxCli] Attempting to connect to simulator at "
                  << ip_addr_ << ":" << port_ << "..." << std::endl;

        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            if (attempt == 1) {
                std::perror("[FxCli] socket");
            }
            ::usleep(500000);
            continue;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(port_);
        if (::inet_pton(AF_INET, ip_addr_.c_str(), &addr.sin_addr) <= 0) {
            if (attempt == 1) {
                std::perror("[FxCli] inet_pton");
            }
            ::close(fd);
            ::usleep(500000);
            continue;
        }

        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            if (attempt % 5 == 1) {
                std::perror("[FxCli] connect");
            }
            ::close(fd);
            ::usleep(500000);
            continue;
        }

        std::cout << "[FxCli] ✓ Connected to simulator at "
                  << ip_addr_ << ":" << port_
                  << " after " << attempt << " attempt(s)" << std::endl;

        int flag = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        sock_fd = fd;
        return true;
    }
}

bool FxCli::send_all(const std::string& s) {
    int& sock_fd = socket_ref();
    if (sock_fd < 0) {
        return false;
    }
    const char* data = s.c_str();
    size_t total = s.size();
    ssize_t sent = 0;
    while (sent < static_cast<ssize_t>(total)) {
        ssize_t n = ::send(sock_fd, data + sent, total - sent, 0);
        if (n <= 0) {
            return false;
        }
        sent += n;
    }
    return true;
}

bool FxCli::read_line(std::string& out) {
    int& sock_fd = socket_ref();
    out.clear();
    if (sock_fd < 0) {
        return false;
    }
    char c;
    while (true) {
        ssize_t n = ::recv(sock_fd, &c, 1, 0);
        if (n <= 0) {
            return false;
        }
        if (c == '\n') {
            break;
        }
        out.push_back(c);
    }
    return true;
}

// Public API

bool FxCli::motor_start(const std::vector<uint8_t>&) {
    return true;
}

bool FxCli::motor_stop(const std::vector<uint8_t>&) {
    return true;
}

bool FxCli::motor_estop(const std::vector<uint8_t>&) {
    return true;
}

bool FxCli::operation_control(const std::vector<uint8_t> &ids,
                              const std::vector<float> &pos,
                              const std::vector<float> &vel,
                              const std::vector<float> &kp,
                              const std::vector<float> &kd,
                              const std::vector<float> &tau) {
    if (!connect_if_needed()) {
        return false;
    }

    std::ostringstream oss;
    oss << "{\"type\":\"control\",";
    // ids
    oss << "\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        oss << static_cast<int>(ids[i]);
        if (i + 1 < ids.size()) oss << ',';
    }
    oss << "],";
    // pos
    oss << "\"pos\":[";
    for (size_t i = 0; i < pos.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << pos[i];
        if (i + 1 < pos.size()) oss << ',';
    }
    oss << "],";
    // vel
    oss << "\"vel\":[";
    for (size_t i = 0; i < vel.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << vel[i];
        if (i + 1 < vel.size()) oss << ',';
    }
    oss << "],";
    // kp
    oss << "\"kp\":[";
    for (size_t i = 0; i < kp.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << kp[i];
        if (i + 1 < kp.size()) oss << ',';
    }
    oss << "],";
    // kd
    oss << "\"kd\":[";
    for (size_t i = 0; i < kd.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << kd[i];
        if (i + 1 < kd.size()) oss << ',';
    }
    oss << "],";
    // tau
    oss << "\"tau\":[";
    for (size_t i = 0; i < tau.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << tau[i];
        if (i + 1 < tau.size()) oss << ',';
    }
    oss << "]}";

    std::string msg = oss.str();
    msg.push_back('\n');
    return send_all(msg);
}

FxCliMap FxCli::req(const std::vector<uint8_t> &ids) {
    FxCliMap out;

    if (!connect_if_needed()) {
        return out;
    }

    std::ostringstream oss;
    oss << "{\"type\":\"req\",";
    oss << "\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        oss << static_cast<int>(ids[i]);
        if (i + 1 < ids.size()) oss << ',';
    }
    oss << "]}";
    std::string msg = oss.str();
    msg.push_back('\n');

    if (!send_all(msg)) {
        return out;
    }

    std::string line;
    if (!read_line(line)) {
        return out;
    }

    try {
        JsonParser parser(line);
        out = parser.parse();
    } catch (...) {
        out.clear();
    }
    return out;
}

// status() 더미: front 는 M1~M8, rear 는 M9~M16 채워줌
FxCliMap FxCli::status() {
    static std::int32_t request_count = 0;
    static float voltage = 56.0f;
    static float soc = 100.0f;

    ++request_count;
    voltage = std::max(voltage - 0.0000672f, 0.0f);
    soc     = std::max(soc - 0.00056f, 0.0f);

    std::string cnt_str     = std::to_string(request_count);
    std::string voltage_str = std::to_string(voltage);
    std::string soc_str     = std::to_string(soc);

    FxCliMap st;
    st["ACK"] = { {"STATUS", "true"} };
    st["MCU"] = {
        {"robot", "2w2l_pro"},
        {"fw",    "v3.0.1"},
        {"proto", "ATv1"},
        {"uptime", "88"}
    };
    st["NET"] = {
        {"status", "up"},
        {"ip",     ip_addr_},
        {"gw",     "127.0.0.1"},
        {"mask",   "255.255.255.0"}
    };

    const int first_mid = is_front_ ? 1 : 9;
    const int last_mid  = is_front_ ? 8 : 16;
    for (int mid = first_mid; mid <= last_mid; ++mid) {
        std::string key = "M" + std::to_string(mid);
        st[key] = { {"pattern", "2"}, {"err", "None"} };
    }

    st["EMERGENCY"] = { {"value", "OFF"} };
    st["IMU"] = { {"value", "N/A"} };
    st["BATT"] = {
        {"V",   voltage_str},
        {"I",   "0.049"},
        {"P",   "2.38"},
        {"T",   "26.8"},
        {"SOC", soc_str}
    };
    st["SEQ_NUM"] = { {"cnt", cnt_str} };
    return st;
}
