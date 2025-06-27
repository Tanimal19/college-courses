#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#include <netdb.h>
#include <dirent.h>
#include <sys/stat.h>
#include "utils/percent.c"
#include "utils/base64.c"

#define MAX_FILE_SIZE 209715200 // 200MB
#define BUFSIZE 1024
#define ERR_EXIT(a) \
    {               \
        perror(a);  \
        exit(1);    \
    }
#define BOUNDARY "abce11902038"
#define MULTIPART_FORM_HEADER "multipart/form-data; boundary=" BOUNDARY

char *_host_name, *_host_ip, *_authorization_code;
int _port, _sockfd;
char* _root_dir;

typedef struct http_response
{
    char *status_code;
    char *headers;
    char *content;
    size_t content_length;
} http_response;

typedef struct http_request
{
    char *content;
    size_t content_length;
} http_request;

typedef struct rfile
{
    char *content;
    size_t content_length;
} rfile;

struct http_request *generate_http_request(char *method, char *url, char *connection, char *content_type_str, size_t content_length, char *content);
int recv_http_response(struct http_response *response);
void decode_http_header(struct http_response *response, char *header);
char *get_http_header_value(char *headers, char *key);
void append_http_header(char *content, char *key, char *value);
struct rfile *read_file(char *filename);
void create_dir_if_not_exist(char *dir_name);
void generate_multipart_form_data(struct rfile *file, char *filename, char *content_type);
void download_file(char *filename, char *content_type, size_t content_length, char *content);
char *change_to_absolute_path(char *path);

int main(int argc, char *argv[])
{
    char path[PATH_MAX];
    if (realpath(argv[0], path) == NULL) {
        perror("realpath");
        return -1;
    }
    _root_dir = dirname(path);

    if (argc != 3 && argc != 4)
    {
        fprintf(stderr, "Usage: ./client [host] [port] [username:password]\n");
        return -1;
    }
    else
    {
        // get host name and host ip
        _host_name = argv[1];
        struct hostent *lh = gethostbyname(_host_name);
        if (lh)
            _host_ip = inet_ntoa(*((struct in_addr *)lh->h_addr_list[0]));

        // get port
        char *endptr;
        _port = strtol(argv[2], &endptr, 10);

        if (*endptr != '\0')
        {
            fprintf(stderr, "Usage: ./client [host] [port] [username:password]\n");
            return -1;
        }

        if (_port < 1024 || _port > 65535)
        {
            fprintf(stderr, "Usage: ./client [host] [port] [username:password]\n");
            return -1;
        }

        if (argc == 4)
        {
            // set authorization code
            size_t output_length;
            char *encoded_username_password = (char *)base64_encode((unsigned char *)argv[3], strlen(argv[3]), &output_length);

            char *tmp = malloc(30);
            strcpy(tmp, "Basic ");
            strcat(tmp, encoded_username_password);
            _authorization_code = tmp;
        }
    }

    struct sockaddr_in addr;

    // Get socket file descriptor
    if ((_sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        ERR_EXIT("socket()");

    // Set server address
    bzero(&addr, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(_host_ip);
    addr.sin_port = htons(_port);

    // Connect to the server
    if (connect(_sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        ERR_EXIT("connect()");

    // Authentication
    struct http_request *request = generate_http_request("GET", "/upload/file", "Keep-Alive", NULL, 0, NULL);
    if (send(_sockfd, request->content, request->content_length, 0) < 0)
        ERR_EXIT("send()");

    struct http_response *response = malloc(sizeof(struct http_response));
    int status = recv_http_response(response);

    if (status <= 0)
    {
        return -1;
    }

    if (strcmp(response->status_code, "401") == 0)
    {
        perror("Invalid user or wrong password.\n");
        return -1;
    }

    char *input = malloc(BUFSIZE);
    while (1)
    {
        printf("> ");
        fgets(input, BUFSIZE, stdin);

        if (strcmp(input, "\n") == 0)
            continue;

        if ((strlen(input) > 0) && (input[strlen(input) - 1] == '\n'))
            input[strlen(input) - 1] = '\0';

        if (strcmp(input, "quit") == 0)
        {
            printf("Bye.\n");
            break;
        }
        else
        {
            char *command, *filename;
            command = strtok(input, " ");

            struct http_request *request = NULL;

            if (strcmp(command, "put") == 0)
            {
                filename = strtok(NULL, "\0");
                if (filename == NULL)
                {
                    fprintf(stderr, "Usage: put [file]\n");
                    continue;
                }

                struct rfile *file = read_file(filename);
                if (file != NULL)
                {
                    generate_multipart_form_data(file, filename, "text/plain");
                    request = generate_http_request("POST", "/api/file", "Keep-Alive", MULTIPART_FORM_HEADER, file->content_length, file->content);

                    if (send(_sockfd, request->content, request->content_length, 0) < 0)
                        ERR_EXIT("send()");

                    struct http_response *response = malloc(sizeof(struct http_response));
                    int status = recv_http_response(response);

                    if (status > 0)
                    {
                        if (strcmp(response->status_code, "200") == 0)
                        {
                            fprintf(stdout, "Command succeeded.\n");
                            continue;
                        }
                    }
                }
            }
            else if (strcmp(command, "putv") == 0)
            {
                filename = strtok(NULL, "\0");
                if (filename == NULL)
                {
                    fprintf(stderr, "Usage: putv [file]\n");
                    continue;
                }

                struct rfile *file = read_file(filename);
                if (file != NULL)
                {
                    generate_multipart_form_data(file, filename, "video/mp4");
                    request = generate_http_request("POST", "/api/video", "Keep-Alive", MULTIPART_FORM_HEADER, file->content_length, file->content);

                    if (send(_sockfd, request->content, request->content_length, 0) < 0)
                        ERR_EXIT("send()");

                    struct http_response *response = malloc(sizeof(struct http_response));
                    int status = recv_http_response(response);

                    if (status > 0)
                    {
                        if (strcmp(response->status_code, "200") == 0)
                        {
                            fprintf(stdout, "Command succeeded.\n");
                            continue;
                        }
                    }
                }
            }
            else if (strcmp(command, "get") == 0)
            {
                filename = strtok(NULL, "\0");
                if (filename == NULL)
                {
                    fprintf(stderr, "Usage: get [file]\n");
                    continue;
                }

                char *url = malloc(strlen(filename) + 11);
                strcpy(url, "/api/file/");
                strcat(url, filename);
                request = generate_http_request("GET", url, "Keep-Alive", NULL, 0, NULL);

                if (send(_sockfd, request->content, request->content_length, 0) < 0)
                    ERR_EXIT("send()");

                struct http_response *response = malloc(sizeof(struct http_response));
                int status = recv_http_response(response);

                if (status > 0 && strcmp(response->status_code, "200") == 0)
                {
                    char *content_type = get_http_header_value(response->headers, "Content-Type");
                    char *content_length_str = get_http_header_value(response->headers, "Content-Length");
                    size_t content_length = atoi(content_length_str);

                    download_file(filename, content_type, content_length, response->content);

                    fprintf(stdout, "Command succeeded.\n");
                    continue;
                }
            }
            else
            {
                fprintf(stderr, "Command Not Found.\n");
                continue;
            }

            fprintf(stderr, "Command failed.\n");
        }
    }

    free(input);
    close(_sockfd);

    return 0;
}

struct http_request *generate_http_request(char *method, char *url, char *connection, char *content_type_str, size_t content_length, char *content)
{
    http_request *request = malloc(sizeof(http_request));
    request->content = malloc(BUFSIZE);

    strcpy(request->content, method);
    strcat(request->content, " ");
    strcat(request->content, url);
    strcat(request->content, " HTTP/1.1\r\n");

    append_http_header(request->content, "Host", _host_name);
    append_http_header(request->content, "User-Agent", "CN2024Client/1.0");
    if (_authorization_code != NULL)
        append_http_header(request->content, "Authorization", _authorization_code);
    append_http_header(request->content, "Connection", connection);

    if (content != NULL)
    {
        char content_length_str[20];
        sprintf(content_length_str, "%ld", content_length);

        append_http_header(request->content, "Content-Type", content_type_str);
        append_http_header(request->content, "Content-Length", content_length_str);
        strcat(request->content, "\r\n");

        size_t total_length = strlen(request->content) + content_length;
        if (total_length > BUFSIZE)
        {
            request->content = realloc(request->content, total_length + 1);
        }
        memcpy(request->content + strlen(request->content), content, content_length);

        request->content_length = total_length;
    }
    else
    {
        append_http_header(request->content, "Content-Length", "0");
        strcat(request->content, "\r\n");

        request->content_length = strlen(request->content);
    }

    // printf("\n-- generate request --\n");
    // fwrite(request->content, request->content_length, 1, stdout);

    return request;
}

int recv_http_response(struct http_response *response)
{
    int nbytes;
    char buffer[BUFSIZE];

    nbytes = recv(_sockfd, buffer, BUFSIZE, 0);
    if (nbytes <= 0)
    {
        return -1;
    }

    // decode http response
    int content_length, recv_length;
    char *header, *body;

    char *separator = strstr(buffer, "\r\n\r\n");
    if (separator == NULL)
    {
        perror("recv response wrong format\n");
        return 0;
    }

    header = buffer;
    body = separator + 4;
    *separator = '\0';

    recv_length = nbytes - strlen(header) - 4;

    // set header
    decode_http_header(response, header);

    // get Content-Length
    char *s = get_http_header_value(response->headers, "Content-Length");
    if (s == NULL)
    {
        content_length = 0;
    }
    else
    {
        content_length = atoi(s);
    }
    response->content_length = content_length;

    // copy content
    response->content = malloc(content_length + 1);
    memcpy(response->content, body, recv_length);

    // get rest of the content
    while (recv_length < content_length)
    {
        nbytes = recv(_sockfd, buffer, BUFSIZE, 0);

        if (nbytes < 0)
        {
            return -1;
        }

        // copy content
        memcpy(&response->content[recv_length], buffer, nbytes);
        recv_length += nbytes;

        if (recv_length > MAX_FILE_SIZE)
        {
            perror("file too large\n");
            return 0;
        }
    }
    response->content[content_length] = '\0';

    // printf("\n-- recv response --\n");
    // printf("status code: %s\n", response->status_code);
    // printf("headers: %s\n", response->headers);
    // printf("content length: %ld\n", response->content_length);
    // printf("content: %s\n", response->content);

    return 1;
}

void decode_http_header(struct http_response *response, char *header)
{
    char *tmp;

    strtok(header, " "); // skip HTTP/1.1

    tmp = strtok(NULL, " ");
    response->status_code = malloc(strlen(tmp) + 1);
    strcpy(response->status_code, tmp);

    strtok(NULL, "\r\n"); // skip status code

    tmp = strtok(NULL, "");
    response->headers = malloc(strlen(tmp) + 3);
    strcpy(response->headers, tmp);
    strcat(response->headers, "\r\n");
}

char *get_http_header_value(char *headers, char *key)
{
    char *line = strstr(headers, key);
    if (line == NULL)
    {
        return NULL;
    }

    char *start, *end;
    start = strstr(line, ": ");
    start += 2;
    end = strstr(start, "\r\n");
    if (end == NULL)
    {
        return NULL;
    }

    size_t len = end - start;
    char *value = malloc(len + 1);
    memcpy(value, start, len);
    value[len] = '\0';

    return value;
}

void append_http_header(char *content, char *key, char *value)
{
    strcat(content, key);
    strcat(content, ": ");
    strcat(content, value);
    strcat(content, "\r\n");
}

struct rfile *read_file(char *filename)
{
    char *filepath = change_to_absolute_path(filename);

    struct stat file_stat;
    if (stat(filepath, &file_stat) != 0)
    {
        // file not exist
        return NULL;
    }
    if (!S_ISREG(file_stat.st_mode))
    {
        // not a regular file
        return NULL;
    }

    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    struct rfile *ret = malloc(sizeof(struct rfile));

    fseek(fp, 0, SEEK_END);
    ret->content_length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    ret->content = malloc(ret->content_length + 1);
    fread(ret->content, 1, ret->content_length, fp);
    fclose(fp);
    ret->content[ret->content_length] = '\0';

    return ret;
}

void create_dir_if_not_exist(char *dir_name)
{
    struct stat st = {0};
    if (stat(dir_name, &st) == -1)
    {
        mkdir(dir_name, 0700);
    }
}

void generate_multipart_form_data(struct rfile *file, char *filename, char *content_type)
{
    char *modify_content = malloc(BUFSIZE);
    char *end_boundary = "\r\n--" BOUNDARY "--\r\n";

    strcpy(modify_content, "--" BOUNDARY "\r\n");

    strcat(modify_content, "Content-Disposition: form-data; name=\"upfile\"; filename=\"");
    strcat(modify_content, filename);
    strcat(modify_content, "\"\r\n");

    strcat(modify_content, "Content-Type: ");
    strcat(modify_content, content_type);
    strcat(modify_content, "\r\n\r\n");

    size_t total_length = strlen(modify_content) + file->content_length + strlen(end_boundary) + 1;
    if (total_length > BUFSIZE)
    {
        modify_content = realloc(modify_content, total_length);
    }

    size_t cur_length = strlen(modify_content);
    memcpy(modify_content + cur_length, file->content, file->content_length);
    cur_length += file->content_length;
    memcpy(modify_content + cur_length, end_boundary, strlen(end_boundary));

    file->content = modify_content;
    file->content_length = total_length;
}

void download_file(char *filename, char *content_type, size_t content_length, char *content)
{

    char *dir_name = change_to_absolute_path("files");
    create_dir_if_not_exist(dir_name);

    char filepath[100];
    strcpy(filepath, dir_name);
    strcat(filepath, "/");
    strcat(filepath, filename);

    FILE *fp = fopen(filepath, "w");
    if (fp != NULL)
    {
        if (strcmp(content_type, "text/plain") == 0)
        {
            fwrite(content, sizeof(char), content_length, fp);
        }
        else
        {
            unsigned char *binary_content = (unsigned char *)content;
            fwrite(binary_content, sizeof(unsigned char), content_length, fp);
        }
        fclose(fp);
    }
}

char *change_to_absolute_path(char *path)
{
    char *absolute_path = malloc(strlen(_root_dir) + strlen(path) + 1);
    strcpy(absolute_path, _root_dir);
    strcat(absolute_path, "/");
    strcat(absolute_path, path);

    return absolute_path;
}