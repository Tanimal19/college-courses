#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstring>
#include <openssl/evp.h>
#include <string>
#include <sstream>
#include <iomanip>

#include <zlib.h>

#include "def.h"

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

string hexDigest(const void *buf, int len) {
    const unsigned char *cbuf = static_cast<const unsigned char *>(buf);
    ostringstream hx{};

    for (int i = 0; i != len; ++i)
        hx << hex << setfill('0') << setw(2) << (unsigned int)cbuf[i];

    return hx.str();
}


struct segment receive_buffer[MAX_SEG_BUF_SIZE];
int buffer_range_start = 1;
int buffer_range_end = MAX_SEG_BUF_SIZE;
int buffer_size = 0;

int last_ack = 0; // last cumulative ACK's seqNumber

FILE *output_file = nullptr;

EVP_MD_CTX *sha256 = EVP_MD_CTX_new();
int total_bytes_received = 0;



// flush the receive buffer to the file, when buffer is full, or FIN packet is received
void flush_receive_buffer(int is_fin) {
    if (output_file == nullptr) {
        return;
    }

    for (int i = 0; i < MAX_SEG_BUF_SIZE; ++i) {
        if (receive_buffer[i].head.seqNumber != buffer_range_start + i) {
            break;
        }

        fwrite(receive_buffer[i].data, sizeof(char), receive_buffer[i].head.length, output_file);

        EVP_DigestUpdate(sha256, receive_buffer[i].data, receive_buffer[i].head.length);
        total_bytes_received += receive_buffer[i].head.length;

        receive_buffer[i] = {}; // clear buffer
    }

    // reset buffer
    buffer_range_start += MAX_SEG_BUF_SIZE;
    buffer_range_end += MAX_SEG_BUF_SIZE;
    buffer_size = 0;
    last_ack = buffer_range_start - 1;

    printf("flush\n");

    // calculate hash
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;
    EVP_MD_CTX *tmp_sha256 = EVP_MD_CTX_new();
    EVP_MD_CTX_copy_ex(tmp_sha256, sha256);
    EVP_DigestFinal_ex(tmp_sha256, hash, &hash_len);
    EVP_MD_CTX_free(tmp_sha256);

    printf("sha256\t%d\t%s\n", total_bytes_received , hexDigest(hash, hash_len).c_str());

    if (is_fin == 1) {
        printf("finsha\t%s\n", hexDigest(hash, hash_len).c_str());
        fclose(output_file);
    }
}



// ./receiver <recv_ip> <recv_port> <agent_ip> <agent_port> <dst_filepath>
int main(int argc, char *argv[]) {
    // parse arguments
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <recv_ip> <recv_port> <agent_ip> <agent_port> <dst_filepath>" << endl;
        exit(1);
    }

    int recv_port, agent_port;
    char recv_ip[50], agent_ip[50];

    // read argument
    setIP(recv_ip, argv[1]);
    sscanf(argv[2], "%d", &recv_port);

    setIP(agent_ip, argv[3]);
    sscanf(argv[4], "%d", &agent_port);

    char *filepath = argv[5];

    // open file, create if not exist, truncate if exist
    output_file = fopen(filepath, "ab");
    if (output_file == nullptr) {
        cerr << "Error: Unable to open file " << filepath << endl;
        exit(1);
    }
    ftruncate(fileno(output_file), 0);
    fseek(output_file, 0, SEEK_SET);

    // make socket related stuff
    int sock_fd = socket(PF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in recv_addr;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(agent_port);
    recv_addr.sin_addr.s_addr = inet_addr(agent_ip);

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(recv_port);
    addr.sin_addr.s_addr = inet_addr(recv_ip);
    memset(addr.sin_zero, '\0', sizeof(addr.sin_zero));
    bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr));

    // init sha256
    EVP_DigestInit_ex(sha256, EVP_sha256(), NULL);


    while (true) {
        // cout << endl;
        // cout << "buffer: ";
        // for (int i = 0; i < MAX_SEG_BUF_SIZE; ++i) {
        //     int seqNumber = receive_buffer[i].head.seqNumber;
        //     cout << seqNumber << " ";
        // }
        // cout << endl;

        int should_flush = 0;

        fflush(stdout);

        // receive data segment
        socklen_t recv_addr_sz;
        struct segment recv_sgmt{};
        recvfrom(sock_fd, &recv_sgmt, sizeof(recv_sgmt), 0, (struct sockaddr *)&recv_addr, &recv_addr_sz);

        // receive FIN
        if (recv_sgmt.head.fin == 1) {
            printf("recv\tfin\n");

            // send a FINACK
            struct segment finack_sgmt{};
            finack_sgmt.head.fin = 1;
            finack_sgmt.head.ack = 1;
            sendto(sock_fd, &finack_sgmt, sizeof(finack_sgmt), 0, (struct sockaddr *)&recv_addr, sizeof(sockaddr));
            printf("send\tfinack\n");

            flush_receive_buffer(1);
            break;
        }

        int seqNumber = recv_sgmt.head.seqNumber;

        struct segment ack_sgmt{};
        ack_sgmt.head.ack = 1;

        // drop segment: corrupted
        unsigned int expect_checksum = recv_sgmt.head.checksum;
        unsigned int data_checksum = crc32(0L, (const Bytef *)recv_sgmt.data, MAX_SEG_SIZE);
        if (data_checksum != expect_checksum) {
            printf("drop\tdata\t#%d\t(corrupted)\n", seqNumber);

            ack_sgmt.head.ackNumber = last_ack;
            ack_sgmt.head.sackNumber = last_ack;
        }

        // drop segment: buffer overflow, i.e. above buffer range
        else if (seqNumber > buffer_range_end) {
            printf("drop\tdata\t#%d\t(buffer overflow)\n", seqNumber);

            ack_sgmt.head.ackNumber = last_ack;
            ack_sgmt.head.sackNumber = last_ack;
        }

        // store segment: out of order
        else if (seqNumber != last_ack + 1) {
            printf("recv\tdata\t#%d\t(out of order, sack-ed)\n", seqNumber);

            // under buffer range
            if (seqNumber < buffer_range_start) {
                ack_sgmt.head.ackNumber = last_ack;
                ack_sgmt.head.sackNumber = seqNumber;
            }
            // in buffer range
            else {
                // if not stored before
                if (receive_buffer[seqNumber - buffer_range_start].head.seqNumber != seqNumber) {
                    receive_buffer[seqNumber - buffer_range_start] = recv_sgmt;
                    buffer_size += 1;
                }

                ack_sgmt.head.ackNumber = last_ack;
                ack_sgmt.head.sackNumber = seqNumber;
            }

        }

        // store segment: in order
        else if (seqNumber == last_ack + 1) {
            printf("recv\tdata\t#%d\t(in order)\n", seqNumber);

            receive_buffer[seqNumber - buffer_range_start] = recv_sgmt;
            buffer_size += 1;

            ack_sgmt.head.ackNumber = seqNumber;
            ack_sgmt.head.sackNumber = seqNumber;

            int new_last_ack = last_ack;
            for (int i = last_ack + 1; i < buffer_range_end; ++i) {
                if (receive_buffer[i - buffer_range_start].head.seqNumber != i) {
                    break;
                }

                new_last_ack += 1;
            }
            last_ack = new_last_ack;

            if (buffer_size == MAX_SEG_BUF_SIZE) {
                should_flush = 1;
            }
        }

        // send ACK
        sendto(sock_fd, &ack_sgmt, sizeof(ack_sgmt), 0, (struct sockaddr *)&recv_addr, sizeof(sockaddr));
        printf("send\tack\t#%d,\tsack\t#%d\n", ack_sgmt.head.ackNumber , ack_sgmt.head.sackNumber);

        if (should_flush == 1) {
            flush_receive_buffer(0);
        }
    }

    close(sock_fd);
    return 0;
}