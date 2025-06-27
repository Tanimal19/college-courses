#include "hw1.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <sys/select.h>
#include <sys/time.h>

#define ERR_EXIT(a) do { perror(a); exit(1); } while(0)
#define BUFFER_SIZE 512

typedef struct {
    char hostname[512];  // server's hostname
    unsigned short port;  // port to listen
    int listen_fd;  // fd to wait for a new connection
} server;

typedef struct {
    char host[512];  // client's host
    int conn_fd;  // fd to talk with client
    char buf[BUFFER_SIZE];  // data sent by/to client
    size_t buf_len;  // bytes used by buf
    int id;
    int status; // 0:ready 1:post 2:pull
	int post_idx;
} request;

server svr;  // server
request* requestP = NULL;  // point to a list of requests
int maxfd;  // size of open file descriptor table, size of request list

// initailize a server, exit for error
static void init_server(unsigned short port);

// initailize a request instance
static void init_request(request* reqP);

// free resources used by a request instance
static void free_request(request* reqP);

int main(int argc, char** argv) {
	// Parse args.
    if (argc != 2) {
        ERR_EXIT("usage: [port]");
        exit(1);
    }

    struct sockaddr_in cliaddr;  // used by accept()
    int clilen;

    int conn_fd;  // fd for a new connection with client
    int file_fd;  // fd for file that we open for reading
    char buf[BUFFER_SIZE];
    int buf_len;

    // Initialize server
    init_server((unsigned short) atoi(argv[1]));

    // Loop for handling connections
    fprintf(stderr, "\nstarting on %.80s, port %d, fd %d, maxconn %d...\n", svr.hostname, svr.port, svr.listen_fd, maxfd);

	// Init readSet
    fd_set readSet;
    FD_ZERO(&readSet);
    FD_SET(svr.listen_fd, &readSet);

    struct timeval timeout;
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;

	// Bulletin Board
	int bulletin = open(RECORD_PATH, O_RDWR);
	int last = RECORD_NUM-1;
	struct flock LOCK;
	LOCK.l_type = F_WRLCK;
	LOCK.l_whence = SEEK_SET;
	LOCK.l_len = sizeof(record);

	int record_lock[RECORD_NUM];
	memset(record_lock, 0, sizeof(record_lock));

    while (1) {
        // TODO: Add IO multiplexing
	
		fd_set tmp_readSet = readSet;
		int ready_fds = select(maxfd+1, &tmp_readSet, NULL, NULL, &timeout);
		//fprintf(stderr, "ready fds: %d\n", ready_fds);

		if(ready_fds > 0){
			if(FD_ISSET(svr.listen_fd, &tmp_readSet)){
				ready_fds--;

       			// Check new connection
       			clilen = sizeof(cliaddr);
        		conn_fd = accept(svr.listen_fd, (struct sockaddr*)&cliaddr, (socklen_t*)&clilen);
        		if (conn_fd < 0) {
            		if (errno == EINTR || errno == EAGAIN) continue;  // try again
            		if (errno == ENFILE) {
                		(void) fprintf(stderr, "out of file descriptor table ... (maxconn %d)\n", maxfd);
                		continue;
            		}
            		ERR_EXIT("accept");
        		}
        		requestP[conn_fd].conn_fd = conn_fd;
        		strcpy(requestP[conn_fd].host, inet_ntoa(cliaddr.sin_addr));
        		fprintf(stderr, "getting a new request... fd %d from %s\n", conn_fd, requestP[conn_fd].host);
				
				// Add new connection into readSet
				FD_SET(requestP[conn_fd].conn_fd, &readSet);
			}

			// Check which client fd is ready to read
			for(int i=0 ; i<maxfd+1 && ready_fds>0 ; i++){
				if(FD_ISSET(requestP[i].conn_fd, &tmp_readSet)){

					// TODO: handle requests from clients
					if(requestP[i].status == 0){
						requestP[i].buf_len = read(requestP[i].conn_fd, requestP[i].buf, BUFFER_SIZE);
						//fprintf(stderr, "Receive Content: %s\n", requestP[i].buf);
					
						if(strcmp(requestP[i].buf, "post") == 0){
							// Check record lock
							int index = last;
							for(int j=0 ; j<RECORD_NUM ; j++){
								index++;
								if(index >= RECORD_NUM) index = 0;

								LOCK.l_type = F_WRLCK;
								LOCK.l_whence = SEEK_SET;
								LOCK.l_start = sizeof(record) * index;
								LOCK.l_len = sizeof(record);
								LOCK.l_pid = getpid();

								fcntl(bulletin, F_GETLK, &LOCK);
								// If record can lock
								if(LOCK.l_type == F_UNLCK && record_lock[index] == 0){
									//fprintf(stderr, "record %d is not locked, post ACCEPT\n", index);
									
									requestP[i].post_idx = index;
									// Lock record
									LOCK.l_type = F_WRLCK;
									fcntl(bulletin, F_SETLK, &LOCK);
									record_lock[requestP[i].post_idx] = 1;

									// Send ACCEPT to client
									requestP[i].status = 1;
									char respond[10] = "ACCEPT";
									lseek(requestP[i].conn_fd, 0, SEEK_SET);
									write(requestP[i].conn_fd, respond, sizeof(respond));

									// Lock record
									break;
								}
								//fprintf(stderr, "record %d is locked\n", index);
							}

							// All RECORD_NUM posts is locked, Send ERROR to client
							if(index == last){
								char respond[10] = "ERROR";
								lseek(requestP[i].conn_fd, 0, SEEK_SET);
								write(requestP[i].conn_fd, respond, sizeof(respond));
							}
						}
						else if(strcmp(requestP[i].buf, "pull") == 0){
							int lock_posts = 0;

							memset(buf, 0, BUFFER_SIZE);
							//ftruncate(requestP[i].conn_fd, 0);
							int offset = 0;

							for(int j=0 ; j<RECORD_NUM ; j++){
								LOCK.l_type = F_RDLCK;
								LOCK.l_whence = SEEK_SET;
								LOCK.l_start = sizeof(record) * j;
								LOCK.l_len = sizeof(record);
								LOCK.l_pid = getpid();
								
								fcntl(bulletin, F_GETLK, &LOCK);
								// Read record content into buf
								if(LOCK.l_type == F_UNLCK && record_lock[j] == 0){
									//fprintf(stderr, "Record %d is not locked\n", j);

									lseek(bulletin, (sizeof(record) * j), SEEK_SET);
									record post;
									// Check if record is empty
									if(read(bulletin, &post, sizeof(record)) > 0){
										memcpy(buf + (sizeof(record) * offset), &post, sizeof(record));
										offset++;
									}
								}
								else{
									//fprintf(stderr, "Record %d is locked\n", j);
									lock_posts++;
								}
							}
							if(lock_posts > 0){
								fprintf(stdout, "[Warning] Try to access locked post - %d\n", lock_posts);
								fflush(stdout);
							}

							lseek(requestP[i].conn_fd, 0, SEEK_SET);
							write(requestP[i].conn_fd, buf, BUFFER_SIZE);
						}
						else if(strcmp(requestP[i].buf, "exit") == 0){
							// Disconnect
							FD_CLR(requestP[i].conn_fd, &readSet);
							close(requestP[i].conn_fd);
							free_request(&requestP[i]);
						}
					}
					else if(requestP[i].status == 1){
						requestP[i].status = 0;

						record input = {.From = "", .Content = ""};
						read(requestP[i].conn_fd, &input, sizeof(record));

						fprintf(stdout, "[Log] Receive post from %s\n", input.From);
						fflush(stdout);
						//fprintf(stderr, "Content: %s\n", input.Content);

						// Write to bulletin board
						//fprintf(stderr, "post to record %d\n", requestP[i].post_idx);
						lseek(bulletin, (requestP[i].post_idx * sizeof(record)), SEEK_SET);
						write(bulletin, &input, sizeof(record));
						last = requestP[i].post_idx;

						// Release record lock
						LOCK.l_type = F_UNLCK;
						LOCK.l_start = requestP[i].post_idx * sizeof(record);
						fcntl(bulletin, F_SETLK, &LOCK);
						record_lock[requestP[i].post_idx] = 0;
						requestP[i].post_idx = -1;
					}

					ready_fds--;
				}
			}
		}
    }

    free(requestP);
    return 0;
}

// ======================================================================================================
// You don't need to know how the following codes are working

static void init_request(request* reqP) {
    reqP->conn_fd = -1;
    reqP->buf_len = 0;
    reqP->id = 0;
    reqP->status = 0;
	reqP->post_idx = -1;
}

static void free_request(request* reqP) {
    init_request(reqP);
}

static void init_server(unsigned short port) {
    struct sockaddr_in servaddr;
    int tmp;

    gethostname(svr.hostname, sizeof(svr.hostname));
    svr.port = port;

    svr.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (svr.listen_fd < 0) ERR_EXIT("socket");

    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);
    tmp = 1;
    if (setsockopt(svr.listen_fd, SOL_SOCKET, SO_REUSEADDR, (void*)&tmp, sizeof(tmp)) < 0) {
        ERR_EXIT("setsockopt");
    }
    if (bind(svr.listen_fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        ERR_EXIT("bind");
    }
    if (listen(svr.listen_fd, 1024) < 0) {
        ERR_EXIT("listen");
    }

    // Get file descripter table size and initialize request table
    maxfd = getdtablesize();
    requestP = (request*) malloc(sizeof(request) * maxfd);
    if (requestP == NULL) {
        ERR_EXIT("out of memory allocating all requests");
    }
    for (int i = 0; i < maxfd; i++) {
        init_request(&requestP[i]);
    }
    requestP[svr.listen_fd].conn_fd = svr.listen_fd;
    strcpy(requestP[svr.listen_fd].host, svr.hostname);

    return;
}
