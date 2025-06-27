#include "hw1.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define ERR_EXIT(a) do { perror(a); exit(1); } while(0)
#define BUFFER_SIZE 512

typedef struct {
    char* ip; // server's ip
    unsigned short port; // server's port
    int conn_fd; // fd to talk with server
    char buf[BUFFER_SIZE]; // data sent by/to server
    size_t buf_len; // bytes used by buf
} client;

client cli;
static void init_client(char** argv);

char divider[30] = "==============================";

int main(int argc, char** argv){
    
    // Parse args.
    if(argc!=3){
        ERR_EXIT("usage: [ip] [port]");
    }

    // Handling connection
    init_client(argv);
    fprintf(stderr, "connect to %s %d\n", cli.ip, cli.port);

	// Print welcom msg
    fprintf(stdout, "%s\n", divider);
	fprintf(stdout, "Welcome to CSIE Bulletin board\n");
	fprintf(stdout, "%s\n", divider);

	// Pull first
	char str[10] = "pull";
	write(cli.conn_fd, str, sizeof(str));

	lseek(cli.conn_fd, 0, SEEK_SET);
	cli.buf_len = read(cli.conn_fd, cli.buf, BUFFER_SIZE);
	int count = 0;
	do{
		record post;
		memcpy(&post, cli.buf + (sizeof(record) * count), sizeof(record));
		count++;
		if(strcmp(post.From, "") != 0 && strcmp(post.Content, "") != 0){
			fprintf(stdout, "FROM: %s\nCONTENT:\n%s\n", post.From, post.Content);
		}
	}while(count < RECORD_NUM);
	fprintf(stdout, "%s\n", divider);

    while(1){
        // TODO: handle user's input
		// read input and send to server
		memset(cli.buf, 0, sizeof(cli.buf));
		fprintf(stdout, "Please enter your command (post/pull/exit): ");
		fscanf(stdin, "%s", cli.buf);
		cli.buf_len = strlen(cli.buf);
		
		write(cli.conn_fd, cli.buf, cli.buf_len);

		// Post
		if(strcmp(cli.buf, "post") == 0){
			lseek(cli.conn_fd, 0, SEEK_SET);
			cli.buf_len = read(cli.conn_fd, cli.buf, BUFFER_SIZE);
			if(strcmp(cli.buf, "ACCEPT") == 0){
				// Read user input
				record input;
				fprintf(stdout, "FROM: ");
				fscanf(stdin, "%s", input.From);
				fprintf(stdout, "CONTENT:\n");
				fscanf(stdin, "%s", input.Content);

				size_t from_len = strlen(input.From);
				size_t content_len = strlen(input.Content);
				for(int i=from_len ; i<FROM_LEN ; i++)
					input.From[i] = '\0';
				for(int i=content_len ; i<CONTENT_LEN ; i++)
					input.Content[i] = '\0';

				lseek(cli.conn_fd, 0, SEEK_SET);
				write(cli.conn_fd, &input, sizeof(record));
			}
			else if(strcmp(cli.buf, "ERROR") == 0){
				fprintf(stdout, "[Error] Maximum posting limit exceeded\n");
			}
		}
		// Pull
		else if(strcmp(cli.buf, "pull") == 0){
			fprintf(stdout, "%s\n", divider);
			lseek(cli.conn_fd, 0, SEEK_SET);
			cli.buf_len = read(cli.conn_fd, cli.buf, BUFFER_SIZE);

			int count = 0;
			do{
				record post;
				memcpy(&post, cli.buf + (sizeof(record) * count), sizeof(record));
				count++;
				if(strcmp(post.From, "") != 0 && strcmp(post.Content, "") != 0){
					fprintf(stdout, "FROM: %s\nCONTENT:\n%s\n", post.From, post.Content);
				}
			}while(count < RECORD_NUM);
			fprintf(stdout, "%s\n", divider);
		}
		// Exit
		else if(strcmp(cli.buf, "exit") == 0){
			return 1;
		}
    }
 
}

static void init_client(char** argv){
    
    cli.ip = argv[1];

    if(atoi(argv[2])==0 || atoi(argv[2])>65536){
        ERR_EXIT("Invalid port");
    }
    cli.port=(unsigned short)atoi(argv[2]);

    struct sockaddr_in servaddr;
    cli.conn_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(cli.conn_fd<0){
        ERR_EXIT("socket");
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(cli.port);

    if(inet_pton(AF_INET, cli.ip, &servaddr.sin_addr)<=0){
        ERR_EXIT("Invalid IP");
    }

    if(connect(cli.conn_fd, (struct sockaddr*)&servaddr, sizeof(servaddr))<0){
        ERR_EXIT("connect");
    }

    return;
}
