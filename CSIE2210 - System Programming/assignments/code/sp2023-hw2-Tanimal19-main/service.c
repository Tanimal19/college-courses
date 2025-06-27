#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>

#include "util.h"

#define ERR_EXIT(s) perror(s), exit(errno);
#define MSG_LEN 30
#define SECRET_LEN 20
#define RD 0
#define WR 1

char msg[MSG_LEN];
char success_msg[MSG_LEN] = "success";
char failed_msg[MSG_LEN] = "failed";
char suicide_msg[MSG_LEN] = "suicide";

struct node {
	service data;
	struct node *next;
};
typedef struct node Node;

/* set up self info */
struct service_self {
	pid_t 	pid;
    int 	p_read_fd;	// for parent
    int 	p_write_fd;	// for parent
    char 	name[MAX_SERVICE_NAME_LEN];
	char	secret[SECRET_LEN];
	int 	is_manager;
	int		child_count;
	Node	*head;
	Node	*tail;
};
struct service_self self;

/* func: print message */
void print_not_exist(char *service_name) {
    printf("%s doesn't exist\n", service_name);
}

void print_receive_command(char *service_name, char *cmd) {
	printf("%s has received %s\n", service_name, cmd);
}

void print_spawn(char *parent_name, char *child_name) {
    printf("%s has spawned a new service %s\n", parent_name, child_name);
}

void print_kill(char *target_name, int decendents_num) {
    printf("%s and %d child services are killed\n", target_name, decendents_num);
}

void print_acquire_secret(char *service_a, char *service_b, char *secret) {
    printf("%s has acquired a new secret from %s, value: %s\n", service_a, service_b, secret);
}

void print_exchange(char *service_a, char *service_b) {
    printf("%s and %s have exchanged their secrets\n", service_a, service_b);
}

/* split string by space to at most 3 parts */
void SPLIT_STR(char *str, char *token1, char *token2, char *token3){
	char str_copy[MAX_CMD_LEN];
	strcpy(str_copy, str);

	char *token;
	token = strtok(str_copy, " ");
	if(token != NULL) strcpy(token1, token);
	
	token = strtok(NULL, " ");
	if(token != NULL) strcpy(token2, token);
	
	token = strtok(NULL, " ");
	if(token != NULL) strcpy(token3, token);
}

/* func: command */
void SPAWN(Node* child){

	if(self.is_manager){
		dup2(STDIN_FILENO, PARENT_READ_FD);
		dup2(STDOUT_FILENO, PARENT_WRITE_FD);
	}

	/* create pipe */
	int p_to_c[2];
	int c_to_p[2];
	pipe2(p_to_c, O_CLOEXEC);
	pipe2(c_to_p, O_CLOEXEC);
			
	/* fork new child */
	pid_t new_pid;

	if((new_pid = fork()) < 0){
		printf("fork error\n");
	}
	else if(new_pid == 0){
		/* child */
		close(p_to_c[WR]);
		close(c_to_p[RD]);
		dup2(p_to_c[RD], PARENT_READ_FD);
		dup2(c_to_p[WR], PARENT_WRITE_FD);
		
		char *argv[] = {"./service", child->data.name, NULL};
		if(execve("./service", argv, NULL) == -1){
			printf("exceve error\n");
		}
	}
	else {
		/* parent */
		child->data.pid = new_pid;
				
		close(c_to_p[WR]);
		close(p_to_c[RD]);
		if(self.is_manager){
			close(PARENT_READ_FD);
			close(PARENT_WRITE_FD);
		}
		child->data.read_fd = c_to_p[RD];
		child->data.write_fd = p_to_c[WR];
	}
}

int KILL_CHILD(){
	int count = 0;

	Node *cur = self.head;
	int i = 0;
	while(cur != NULL && i <= self.child_count){
		write(cur->data.write_fd, suicide_msg, MSG_LEN);

		memset(msg, 0, MSG_LEN);
		read(cur->data.read_fd, msg, MSG_LEN);
		//printf("recieve %s from %s\n", msg, cur->data.name);
		waitpid(cur->data.pid, NULL, 0);
		close(cur->data.write_fd);
		close(cur->data.read_fd);
		count += atoi(msg);
		cur = cur->next;
		i++;
	}
	return count;
}


int main(int argc, char *argv[]) {    
	if (argc != 2) {
        fprintf(stderr, "Usage: ./service [service_name]\n");
        return 0;
    }
	
	/* set self info */
	self.pid = getpid();
    srand(self.pid);
    unsigned long secret = rand();
	sprintf(self.secret, "%lu", secret);

    setvbuf(stdout, NULL, _IONBF, 0);
	strncpy(self.name, argv[1], MAX_SERVICE_NAME_LEN);
	printf("%s has been spawned, pid: %d, secret: %s\n", self.name, self.pid, self.secret);
	
	self.is_manager = (strcmp(self.name, "Manager") == 0)? 1:0;
	self.p_read_fd = PARENT_READ_FD;
	self.p_write_fd = PARENT_WRITE_FD;

	self.head = NULL;
	self.tail = NULL;
	self.child_count = 0;
	
	if(!self.is_manager)
		write(self.p_write_fd, success_msg, MSG_LEN);

	/* main program */

	char cmd[MAX_CMD_LEN];
	while(1){
		memset(cmd, 0, sizeof(cmd));
		
		/* read cmd */
		if(self.is_manager){
			fgets(cmd, sizeof(cmd), stdin);
			cmd[strcspn(cmd, "\n")] = '\0';
		}
		else{
			read(self.p_read_fd, cmd, sizeof(cmd));
		}

		/* interpret command */
		char command[MSG_LEN],
			argv1[MAX_SERVICE_NAME_LEN],
			argv2[MAX_SERVICE_NAME_LEN];
		SPLIT_STR(cmd, command, argv1, argv2);

		/* spawn */
		if(strcmp(command, "spawn") == 0){
			print_receive_command(self.name, cmd);

			if(strcmp(argv1, self.name) == 0){
				/* spawn new service */
				Node *new = (Node *)malloc(sizeof(Node));
				strcpy(new->data.name, argv2);
				new->next = NULL;
				SPAWN(new);

				/* connect new service to child list's tail */
				if(self.child_count == 0) self.head = new;	
				if(self.tail != NULL) self.tail->next = new;
				self.tail = new;
				self.child_count++;

				/* wait for new service spawn */
				memset(msg, 0, MSG_LEN);
				read(new->data.read_fd, msg, MSG_LEN);
				if(strcmp(msg, success_msg) != 0) printf("spawn error\n");

				if(self.is_manager){
					print_spawn(argv1, argv2);	
				}
				else{
					write(self.p_write_fd, success_msg, MSG_LEN);
				}
			}
			else{
				/* traverse to find argv1 */
				int result = 0;
				Node *cur = self.head;
				int i = 0;
				while(cur != NULL && i <= self.child_count){
					//printf("send [%s] to %d\n", cmd, cur->data.write_fd);
					write(cur->data.write_fd, cmd, strlen(cmd));
					
					memset(msg, 0, MSG_LEN);
					read(cur->data.read_fd, msg, MSG_LEN);
					if(strcmp(msg, success_msg) == 0){
						result = 1;
						break;
					}
					cur = cur->next;
					i++;
				}
				
				/* deal with traverse result */ 
				if(self.is_manager){
					if(result == 1){
						print_spawn(argv1, argv2);
					}
					else{
						print_not_exist(argv1);
					}
				}
				else{
					if(result == 1){
						write(self.p_write_fd, success_msg, MSG_LEN);	
					}
					else{
						write(self.p_write_fd, failed_msg, MSG_LEN);
					}
				}
			}
		}
		/* kill */
		else if(strcmp(command, "kill") == 0){
			print_receive_command(self.name, cmd);

			if(strcmp(argv1, self.name) == 0){
				int count = KILL_CHILD();

				if(self.is_manager){
					print_kill(argv1, count);
				}
				else{
					char str[MSG_LEN] = "exited ";
					char count_str[MSG_LEN];
					sprintf(count_str, "%d", count);
					strcat(str, count_str); 
					write(self.p_write_fd, str, MSG_LEN);
				}
				close(self.p_write_fd);
				close(self.p_read_fd);
				exit(0);
			}
			else{
				/* traverse to find argv1 */
				int result = 0;
				int count = 0;
				char command[MSG_LEN], count_str[MSG_LEN];

				Node *cur = self.head;
				Node *prev = NULL;
				int i = 0;
				while(cur != NULL && i <= self.child_count){
					write(cur->data.write_fd, cmd, strlen(cmd));
					
					memset(msg, 0, MSG_LEN);
					memset(command, 0, MSG_LEN);
					memset(count_str, 0, MSG_LEN);
					read(cur->data.read_fd, msg, MSG_LEN);
					SPLIT_STR(msg, command, count_str, NULL);

					//printf("%s recieve [%s] from %s\n", self.name, msg, cur->data.name);
					
					if(strcmp(command, success_msg) == 0){
						result = 1;
						break;
					}
					else if(strcmp(command, "exited") == 0){
						waitpid(cur->data.pid, NULL, 0); // wait for argv1 exit
						close(cur->data.write_fd);
						close(cur->data.read_fd);

						/* connect child before and after argv1 */
						if(prev == NULL){
							self.head = cur->next;
						}
						else{
							prev->next = cur->next;
						}

						if(self.tail == cur){
							self.tail = prev;
						}
						
						self.child_count --;
						result = 1;
						break;
					}

					prev = cur;
					cur = cur->next;
					i++;
				}

				/* deal with traverse result */ 
				if(self.is_manager){
					if(result == 1){
						count = atoi(count_str);
						print_kill(argv1, count);
					}
					else{
						print_not_exist(argv1);
					}
				}
				else{
					if(result == 1){
						char str[MSG_LEN] = "success ";
						strcat(str, count_str);
						write(self.p_write_fd, str, MSG_LEN);
					}
					else{
						write(self.p_write_fd, failed_msg, MSG_LEN);
					}
				}

			}
		}
		/* suicide */
		else if(strcmp(command, suicide_msg) == 0){
			int count = KILL_CHILD() + 1;
			char count_str[MSG_LEN];
			sprintf(count_str, "%d", count);
			write(self.p_write_fd, count_str, MSG_LEN);
			close(self.p_write_fd);
			close(self.p_read_fd);
			exit(0);
		}
		/* exchange */
		else if(strcmp(command, "exchange") == 0){
			print_receive_command(self.name, cmd);

			int find;
			char find_msg[MSG_LEN];
			
			/* set up FIFO name */
			char a_to_b[MAX_FIFO_NAME_LEN],
				b_to_a[MAX_FIFO_NAME_LEN];
				
			memset(a_to_b, 0, MAX_FIFO_NAME_LEN);
			strcat(a_to_b, argv1);
			strcat(a_to_b, "_to_");
			strcat(a_to_b, argv2);
				
			memset(b_to_a, 0, MAX_FIFO_NAME_LEN);
			strcat(b_to_a, argv2);
			strcat(b_to_a, "_to_");
			
			/* manager make two FIFOs */
			if(self.is_manager){
				mkfifo(a_to_b, S_IRWXU | S_IRWXG | S_IRWXO);
				mkfifo(b_to_a, S_IRWXU | S_IRWXG | S_IRWXO);
				find = 0;
			}
			/* not manager wait for second msg */
			else{
				write(self.p_write_fd, " ", 1);		/* send space */
				read(self.p_read_fd, find_msg, MSG_LEN);
				find = atoi(find_msg);
			}
			
			if(strcmp(self.name, argv1) == 0 || strcmp(self.name, argv2) == 0)
				find++;
 			 
			/* traverse until find 2 target */
			Node *cur = self.head;
			int i = 0;
			while(cur != NULL && i <= self.child_count && find < 2){
				write(cur->data.write_fd, cmd, strlen(cmd));
				read(cur->data.read_fd, msg, MSG_LEN);		/* recieve space */

				memset(find_msg, 0, MSG_LEN);
				sprintf(find_msg, "%d", find);
				write(cur->data.write_fd, find_msg, MSG_LEN);
					
				memset(find_msg, 0, MSG_LEN);
				read(cur->data.read_fd, find_msg, MSG_LEN);
				find = atoi(find_msg);
				cur = cur->next;
				i++;
			}

			/* send find # to parent */
			if(!self.is_manager){
				memset(find_msg, 0, MSG_LEN);
				sprintf(find_msg, "%d", find);
				write(self.p_write_fd, find_msg, MSG_LEN);
			}

			int fd;
			if(strcmp(self.name, argv1) == 0){
				/* recieve secret from argv2 */
				fd = open(b_to_a, O_RDONLY);
				char secret2[SECRET_LEN];
				memset(secret2, 0, SECRET_LEN);
				read(fd, secret2, SECRET_LEN);
				close(fd);
				print_acquire_secret(argv1, argv2, secret2);

				/* send secret to argv2 */
				fd = open(a_to_b, O_WRONLY);
				write(fd, self.secret, SECRET_LEN);
				close(fd);

				/* change secret */
				memset(self.secret, 0, SECRET_LEN);
				strcat(self.secret, secret2);
			}
			else if(strcmp(self.name, argv2) == 0){
				/* send secret to argv1 */
				fd = open(b_to_a, O_WRONLY);
				write(fd, self.secret, SECRET_LEN);
				close(fd);

				/* recieve secret from argv1 */
				fd = open(a_to_b, O_RDONLY);
				char secret1[SECRET_LEN];
				memset(secret1, 0, SECRET_LEN);
				read(fd, secret1, SECRET_LEN);
				close(fd);
				print_acquire_secret(argv2, argv1, secret1);

				/* change secret */
				memset(self.secret, 0, SECRET_LEN);
				strcat(self.secret, secret1);
			}
			
			/* manager remove FIFOs */
			if(self.is_manager){
				/* check 2 targets finish exchange */
				find = 0;
				if(strcmp(self.name, argv1) == 0 || strcmp(self.name, argv2) == 0)
					find++;

				char finish_msg[MSG_LEN] = "finish ";
				memset(find_msg, 0, MSG_LEN);
				sprintf(find_msg, "%d", find);
				strcat(finish_msg, find_msg);
				cur = self.head;
				i = 0;
				while(cur != NULL && i <= self.child_count && find < 2){
					write(cur->data.write_fd, finish_msg, MSG_LEN);
						
					memset(find_msg, 0, MSG_LEN);
					read(cur->data.read_fd, find_msg, MSG_LEN);
					find = atoi(find_msg);
					cur = cur->next;
					i++;
				}

				/* remove FIFOs */
				unlink(a_to_b);
				unlink(b_to_a);
				print_exchange(argv1, argv2);
			}
		}
		/* finish (manager will not get this) */
		else if(strcmp(command, "finish") == 0){
			char finish_msg[MSG_LEN], find_msg[MSG_LEN];
			int find = atoi(argv1);			
			Node *cur = self.head;
			int i = 0;
			while(cur != NULL && i <= self.child_count && find < 2){
				memset(find_msg, 0, MSG_LEN);
				memset(finish_msg, 0, MSG_LEN);
				sprintf(find_msg, "%d", find);
				strcat(finish_msg, "finish ");
				strcat(finish_msg, find_msg);
				write(cur->data.write_fd, finish_msg, MSG_LEN);
						
				memset(find_msg, 0, MSG_LEN);
				read(cur->data.read_fd, find_msg, MSG_LEN);
				find = atoi(find_msg);
				cur = cur->next;
				i++;
			}

			memset(find_msg, 0, MSG_LEN);
			sprintf(find_msg, "%d", find);
			write(self.p_write_fd, find_msg, MSG_LEN);
		}
	}
	
    return 0;
}
