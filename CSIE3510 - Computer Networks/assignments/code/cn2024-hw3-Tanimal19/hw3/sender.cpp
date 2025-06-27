#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <time.h>

#include <zlib.h>

#include "def.h"
#include "util.h"

using namespace std;

void setIP(char *dst, char *src){
    if(strcmp(src, "0.0.0.0") == 0 || strcmp(src, "local") == 0 || strcmp(src, "localhost") == 0){
        sscanf("127.0.0.1", "%s", dst);
    }
    else{
        sscanf(src, "%s", dst);
    }
    return;
}

/* Socket Setting */
int sock_fd;
struct sockaddr_in recv_addr;
struct sockaddr_in addr;

/* Sender Init */
enum class FSMState {
    SLOW_START,
    CONGESTION_AVOIDANCE
};

FSMState state = FSMState::SLOW_START;
class SegmentQueue *transmit_queue = new SegmentQueue(); // the whole sequence of data segments waiting to be sent, ordered by seqNumber
int base = 1;
double cwnd = 1;
int thresh = 16;
int dup_ack_count = 0;

/* Timer */
class Timer {
    public:
        Timer() : start_time(std::chrono::steady_clock::now()) {}

        void reset() {
            start_time = std::chrono::steady_clock::now();
        }

        double elapsed() const {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = now - start_time;
            return elapsed_seconds.count() * 1000;
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> start_time;
};


// (Re)transmit segments in transmit_queue[0:cwnd], if state UNSENT
void transmitNew() {
    for (int i = 0; i < cwnd && i < transmit_queue->get_size(); ++i) {
        struct segment_node *node = transmit_queue->get_by_index(i);
        if (node->state == SegmentState::UNSENT) {
            struct segment sgmt = node->sgmt;
            sendto(sock_fd, &sgmt, sizeof(sgmt), 0, (struct sockaddr *)&recv_addr, sizeof(sockaddr));
            node->state = SegmentState::SENT_UNACKED;
            printf("send\tdata\t#%d,\twinSize = %d\n", sgmt.head.seqNumber , int(cwnd));
        }
    }
}

// (Re)transmit the first segment in transmit_queue
void transmitMissing() {
    struct segment_node *node = transmit_queue->get_by_index(0);
    struct segment sgmt = node->sgmt;
    sendto(sock_fd, &sgmt, sizeof(sgmt), 0, (struct sockaddr *)&recv_addr, sizeof(sockaddr));
    node->state = SegmentState::SENT_UNACKED;
    printf("resnd\tdata\t#%d,\twinSize = %d\n", sgmt.head.seqNumber , int(cwnd));
}

// Called on every new cumulative ACK. Update the base, transmit queue, and window s.t. everything up until ackNumber is cumulative acknowledged
void updateBase(int ack_number) {
    base = transmit_queue->get_by_index(0)->sgmt.head.seqNumber;

    for (int i = 0; i < transmit_queue->get_size(); ++i) {
        struct segment_node *node = transmit_queue->get_by_index(i);
        if (node->sgmt.head.seqNumber <= ack_number) {
            transmit_queue->remove(node->sgmt.head.seqNumber);
        }
    }
}

// ./sender <send_ip> <send_port> <agent_ip> <agent_port> <src_filepath>
int main(int argc, char *argv[]) {
    // parse arguments
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <send_ip> <send_port> <agent_ip> <agent_port> <src_filepath>" << endl;
        exit(1);
    }

    int send_port, agent_port;
    char send_ip[50], agent_ip[50];

    // read argument
    setIP(send_ip, argv[1]);
    sscanf(argv[2], "%d", &send_port);

    setIP(agent_ip, argv[3]);
    sscanf(argv[4], "%d", &agent_port);

    char *filepath = argv[5];

    // make socket related stuff
    sock_fd = socket(PF_INET, SOCK_DGRAM, 0);

    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(agent_port);
    recv_addr.sin_addr.s_addr = inet_addr(agent_ip);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(send_port);
    addr.sin_addr.s_addr = inet_addr(send_ip);
    memset(addr.sin_zero, '\0', sizeof(addr.sin_zero));
    bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr));

    struct timeval timeout;
    timeout.tv_sec = TIMEOUT_MILLISECONDS / 1000;
    timeout.tv_usec = (TIMEOUT_MILLISECONDS % 1000) * 1000;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        cerr << "Error setting socket options" << endl;
        close(sock_fd);
        return 1;
    }

    // read file to transmit queue
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        cerr << "Error: Unable to open file " << filepath << endl;
        exit(1);
    }

    char buffer[MAX_SEG_SIZE];
    size_t bytes_read;
    int final_seq_number = 1;
    while ((bytes_read = fread(buffer, 1, MAX_SEG_SIZE, file)) > 0) {
        struct segment sgmt{};
        sgmt.head.length = bytes_read;
        sgmt.head.seqNumber = final_seq_number;
        bzero(sgmt.data, sizeof(char) * MAX_SEG_SIZE);
        memcpy(sgmt.data, buffer, bytes_read);
        sgmt.head.checksum = crc32(0L, (const Bytef *)sgmt.data, MAX_SEG_SIZE);

        transmit_queue->add(sgmt);
        final_seq_number += 1;
    }
    fclose(file);
    cerr << "File read complete. " << transmit_queue->get_size() << " segments to send." << endl;

    // init
    transmitNew();
    base = transmit_queue->get_by_index(0)->sgmt.head.seqNumber;
    Timer timer;
    timer.reset();

    int recv_result;

    while (true) {
        // cout << "cwnd = " << cwnd << ", thresh = " << thresh << ", base = " << base << endl;
        // cout << "transmit_queue: ";
        // for (int i = 0; i < transmit_queue->get_size(); ++i) {
        //     struct segment_node *node = transmit_queue->get_by_index(i);
        //     if (node != nullptr) {
        //         cout << node->sgmt.head.seqNumber << " ";
        //     }
        // }
        // cout << endl;

        // timeout
        if (timer.elapsed() > TIMEOUT_MILLISECONDS) {
            printf("time\tout ,\tthreshold = %d,\twinSize = %d\n", thresh , int(cwnd));

            thresh = int(cwnd / 2);
            if (thresh < 1) thresh = 1;
            cwnd = 1;
            dup_ack_count = 0;
            transmitMissing();
            timer.reset();

            if (state == FSMState::CONGESTION_AVOIDANCE) {
                state = FSMState::SLOW_START;
            }
        }

        fflush(stdout);

        // receive ACK
        socklen_t recv_addr_sz = sizeof(recv_addr);
        struct segment recv_sgmt{};
        recv_result = recvfrom(sock_fd, &recv_sgmt, sizeof(recv_sgmt), 0, (struct sockaddr *)&recv_addr, &recv_addr_sz);

        // timeout
        if (recv_result < 0) continue;

        if (recv_sgmt.head.ack == 0) continue;

        // received FINACK
        if (recv_sgmt.head.fin == 1) {
            printf("recv\tfinack\n");
            break;
        }

        int ack_number = recv_sgmt.head.ackNumber;
        int sack_number = recv_sgmt.head.sackNumber;
        printf("recv\tack\t#%d,\tsack\t#%d\n", ack_number , sack_number);

        // Called at the start of every receive (dupACK or newACK). Remove the segment with seqNumber from transmit_queue.
        struct segment_node* remove_node = transmit_queue->get_by_seqNumber(sack_number);
        if (remove_node != nullptr) {
            remove_node->state = SegmentState::ACKED;
            transmit_queue->remove(sack_number);
        }

        // send FIN if all segments are ACKed
        if (transmit_queue->is_empty()) {
            struct segment fin_sgmt{};
            fin_sgmt.head.seqNumber = final_seq_number + 1;
            fin_sgmt.head.fin = 1;
            bzero(fin_sgmt.data, sizeof(char) * MAX_SEG_SIZE);
            fin_sgmt.head.checksum = crc32(0L, (const Bytef *)fin_sgmt.data, MAX_SEG_SIZE);

            sendto(sock_fd, &fin_sgmt, sizeof(fin_sgmt), 0, (struct sockaddr *)&recv_addr, sizeof(sockaddr));
            printf("send\tfin\n");
        }

        else {
            // recv new cumulative ACK
            if (ack_number >= base) {
                dup_ack_count = 0;

                if (state == FSMState::SLOW_START) {
                    cwnd += 1;
                }
                else if (state == FSMState::CONGESTION_AVOIDANCE) {
                    cwnd += double(1) / int(cwnd);
                }

                transmitNew();
                timer.reset();
                updateBase(ack_number);
            }

            // recv duplicate cumulative ACK
            else if (ack_number < base) {
                dup_ack_count += 1;
                transmitNew();
                if (dup_ack_count == 3) {
                    transmitMissing();
                }
            }

            if (cwnd >= thresh) {
                state = FSMState::CONGESTION_AVOIDANCE;
            }
        }
    }

    close(sock_fd);
    return 0;
}