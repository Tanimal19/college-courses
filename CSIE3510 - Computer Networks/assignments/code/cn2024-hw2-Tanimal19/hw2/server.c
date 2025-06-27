#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <poll.h>
#include <dirent.h>
#include <libgen.h>
#include <limits.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include "utils/percent.c"
#include "utils/base64.c"

#define MAX_FILE_SIZE 209715200 // 200MB
#define BUFSIZE 1024
#define ERR_EXIT(a) \
    {               \
        perror(a);  \
        exit(1);    \
    }
#define MAXCONN 101 // include listenfd

typedef struct http_request
{
    char *method;
    char *url;
    char *headers;
    size_t content_length;
    char *content;
    char *response;
} http_request;

typedef struct http_response
{
    char *content;
    size_t content_length;
} http_response;

typedef struct rfile
{
    char *content;
    size_t content_length;
} rfile;

int _port;
struct pollfd _clients[MAXCONN];
int _max_index;
char* _root_dir;

// 1: success, 0: internal server error, -1: client disconnected
int recv_http_request(struct http_request *request, int idx);

/* http request handler */
struct http_response *handle_http_request(struct http_request *request);
// 1: success, 0: internal server error
int handle_multipart_form_data_file(struct http_request *request, char *boundary);
int handle_multipart_form_data_video(struct http_request *request, char *boundary);
int is_authenticated(http_request *request);
void decode_http_header(struct http_request *request, char *header);
char *get_http_header_value(char *headers, char *key);

/* http response generator */
struct http_response *generate_http_response_200(char *content_type_str, size_t s_content_length, char *content);
struct http_response *generate_http_response_401();
struct http_response *generate_http_response_404();
struct http_response *generate_http_response_405(char *allow_method);
struct http_response *generate_http_response_500();
void append_http_header(char *response, char *key, char *value);

/* helper functions */
void disconnect_client(int idx);
struct rfile *read_file(char *filename);
char *get_mime_type(char *filename);
void exec_video_convert(char *video_name);
void create_dir_if_not_exist(char *dir_name);
char *modify_html(char *html, char *placeholder, char *content);
char *change_to_absolute_path(char *path);
char *extrac_filename_from_path(char *path);

int main(int argc, char *argv[])
{
    char path[PATH_MAX];
    if (realpath(argv[0], path) == NULL) {
        perror("realpath");
        return -1;
    }
    _root_dir = dirname(path);
    // printf("Server directory: %s\n", _root_dir);


    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./server [port]\n");
        return -1;
    }
    else
    {
        char *endptr;

        _port = strtol(argv[1], &endptr, 10);

        if (*endptr != '\0')
        {
            fprintf(stderr, "Usage: ./server [port]\n");
            return -1;
        }

        if (_port < 1024 || _port > 65535)
        {
            fprintf(stderr, "Usage: ./server [port]\n");
            return -1;
        }
    }

    int listenfd, connfd;
    struct sockaddr_in server_addr;

    // Get socket file descriptor
    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        ERR_EXIT("socket()");

    // Set server address information
    bzero(&server_addr, sizeof(server_addr)); // erase the data
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(_port);

    // Bind the server file descriptor to the server address
    if (bind(listenfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        ERR_EXIT("bind()");

    // Listen on the server file descriptor
    if (listen(listenfd, 3) < 0)
        ERR_EXIT("listen()");

    // Initialize pollfd struct
    _clients[0].fd = listenfd;
    _clients[0].events = POLLIN;

    for (int i = 1; i < MAXCONN; i++)
        _clients[i].fd = -1;

    _max_index = 0;

    // Start polling
    int nready;
    struct sockaddr_in client_addr;
    int client_addr_len;

    while (1)
    {
        nready = poll(_clients, MAXCONN, -1);

        if (_clients[0].revents & POLLIN)
        {
            // Accept the client and get client file descriptor
            client_addr_len = sizeof(client_addr);
            if ((connfd = accept(listenfd, (struct sockaddr *)&client_addr, (socklen_t *)&client_addr_len)) < 0)
                ERR_EXIT("accept()");

            int i;
            for (i = 0; i < MAXCONN; i++)
            {
                if (_clients[i].fd < 0)
                {
                    _clients[i].fd = connfd;
                    _clients[i].events = POLLIN;
                    // printf("Accept client at PORT %d, dispatch to fd %d\n", ntohs(client_addr.sin_port), clients[i].fd);
                    break;
                }
            }

            if (i >= MAXCONN)
            {
                // fprintf(stderr, "Too many clients\n");
                close(connfd);
            }
            else
            {
                if (i > _max_index)
                    _max_index = i;
            }

            nready--;
            if (nready <= 0)
                continue;
        }

        for (int i = 1; i <= _max_index; i++)
        {
            if (_clients[i].fd < 0)
                continue;

            if (_clients[i].revents & POLLIN)
            {
                struct http_response *response = NULL;
                struct http_request *request = malloc(sizeof(struct http_request));
                int status = recv_http_request(request, i);

                switch (status)
                {
                case 1:
                    response = handle_http_request(request);
                    break;
                case 0:
                    response = generate_http_response_500();
                    break;
                case -1:
                    free(request);
                    continue;
                }

                send(_clients[i].fd, response->content, response->content_length, 0);
                free(request);

                nready--;
                if (nready <= 0)
                    break;
            }
        }
    }

    close(connfd);
    close(listenfd);
}

int recv_http_request(struct http_request *request, int idx)
{
    int nbytes;
    char buffer[BUFSIZE];

    nbytes = recv(_clients[idx].fd, buffer, BUFSIZE, 0);
    if (nbytes <= 0)
    {
        disconnect_client(idx);
        return -1;
    }

    // decode http request
    int content_length, recv_length;
    char *header, *body;

    char *separator = strstr(buffer, "\r\n\r\n");
    if (separator == NULL)
    {
        perror("recv request wrong format\n");
        return 0;
    }

    header = buffer;
    body = separator + 4;
    *separator = '\0';

    recv_length = nbytes - strlen(header) - 4;

    // set header
    decode_http_header(request, header);

    // get Content-Length
    char *s = get_http_header_value(request->headers, "Content-Length");
    if (s == NULL)
    {
        content_length = 0;
    }
    else
    {
        content_length = atoi(s);
    }
    request->content_length = content_length;

    // copy content
    request->content = malloc(content_length + 1);
    memcpy(request->content, body, recv_length);

    // get rest of the content
    while (recv_length < content_length)
    {
        nbytes = recv(_clients[idx].fd, buffer, BUFSIZE, 0);

        if (nbytes < 0)
        {
            disconnect_client(idx);
            return -1;
        }

        // copy content
        memcpy(&request->content[recv_length], buffer, nbytes);
        recv_length += nbytes;

        if (recv_length > MAX_FILE_SIZE)
        {
            perror("file too large\n");
            return 0;
        }
    }
    request->content[content_length] = '\0';

    // printf("\n-- recv request --\n");
    // printf("method: %s\n", request->method);
    // printf("url: %s\n", request->url);
    // printf("headers: %s\n", request->headers);
    // fwrite(request->content, request->content_length, 1, stdout);

    return 1;
}

struct http_response *handle_http_request(struct http_request *request)
{
    if (strcmp(request->url, "/") == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            struct rfile *file = read_file("web/index.html");
            if (file != NULL)
            {
                return generate_http_response_200("text/html", file->content_length, file->content);
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/upload/file") == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            if (is_authenticated(request))
            {
                printf("Authenticated\n");

                struct rfile *file = read_file("web/uploadf.html");
                if (file != NULL)
                {
                    return generate_http_response_200("text/html", file->content_length, file->content);
                }
                else
                {
                    return generate_http_response_404();
                }
            }
            else
            {
                return generate_http_response_401();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/upload/video") == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            if (is_authenticated(request))
            {
                struct rfile *file = read_file("web/uploadv.html");
                if (file != NULL)
                {
                    return generate_http_response_200("text/html", file->content_length, file->content);
                }
                else
                {
                    return generate_http_response_404();
                }
            }
            else
            {
                return generate_http_response_401();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/file/") == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            struct rfile *file = read_file("web/listf.rhtml");
            if (file != NULL)
            {
                // Replace <?FILE_LIST?>
                DIR *dir;
                struct dirent *ent;
                char *rows = malloc(1);
                rows[0] = '\0';

                char* dir_name = change_to_absolute_path("web/files");
                create_dir_if_not_exist(dir_name);

                if ((dir = opendir(dir_name)) != NULL)
                {
                    while ((ent = readdir(dir)) != NULL)
                    {
                        if (ent->d_type == DT_REG)
                        { // is a regular file
                            size_t row_len = strlen(ent->d_name) * 2 + 50;

                            char *row = malloc(row_len);
                            sprintf(row, "<tr><td><a href=\"/api/file/%s\">%s</a></td></tr>\n", ent->d_name, ent->d_name);

                            rows = realloc(rows, strlen(rows) + row_len + 1);
                            strcat(rows, row);
                        }
                    }
                    closedir(dir);

                    char *new_content = modify_html(file->content, "<?FILE_LIST?>", rows);

                    return generate_http_response_200("text/html", strlen(new_content), new_content);
                }
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/video/") == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            struct rfile *file = read_file("web/listv.rhtml");
            if (file != NULL)
            {
                // Replace <?VIDEO_LIST?>
                DIR *dir;
                struct dirent *ent;
                char *rows = malloc(1);
                rows[0] = '\0';

                char* dir_name = change_to_absolute_path("web/videos");
                create_dir_if_not_exist(dir_name);

                if ((dir = opendir(dir_name)) != NULL)
                {
                    while ((ent = readdir(dir)) != NULL)
                    {
                        if (ent->d_type == DT_DIR)
                        { // is a directory
                            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                            {
                                continue;
                            }

                            size_t row_len = strlen(ent->d_name) * 2 + 50;
                            char *name_without_ext = strtok(ent->d_name, ".");

                            char *row = malloc(row_len);
                            sprintf(row, "<tr><td><a href=\"/video/%s\">%s</a></td></tr>\n", name_without_ext, name_without_ext);

                            rows = realloc(rows, strlen(rows) + row_len + 1);
                            strcat(rows, row);
                        }
                    }
                    closedir(dir);

                    char *new_content = modify_html(file->content, "<?VIDEO_LIST?>", rows);

                    return generate_http_response_200("text/html", strlen(new_content), new_content);
                }
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strncmp(request->url, "/video/", 7) == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            char *video_name = request->url + 7;
            struct rfile *file = read_file("web/player.rhtml");
            if (file != NULL)
            {
                char *new_content;
                char mpd_path[PATH_MAX];
                strcpy(mpd_path, "\"/api/video/");
                strcat(mpd_path, video_name);
                strcat(mpd_path, "/dash.mpd\"");

                new_content = modify_html(file->content, "<?VIDEO_NAME?>", video_name);
                new_content = modify_html(new_content, "<?MPD_PATH?>", mpd_path);

                return generate_http_response_200("text/html", strlen(new_content), new_content);
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/api/file") == 0)
    {
        if (strcmp(request->method, "POST") == 0)
        {
            if (is_authenticated(request))
            {
                char *content_type = get_http_header_value(request->headers, "Content-Type");

                if (content_type != NULL && strstr(content_type, "multipart/form-data") != NULL)
                {
                    // get boundary
                    char *b = strstr(content_type, "boundary=");
                    b += 9; // skip "boundary="

                    char *boundary = malloc(strlen(b) + 3);
                    strcpy(boundary, "--"); // add prefix "--"
                    strcat(boundary, b);

                    handle_multipart_form_data_file(request, boundary);

                    char *msg = "File Uploaded\n";
                    return generate_http_response_200("text/plain", strlen(msg), msg);
                }
            }
            else
            {
                return generate_http_response_401();
            }
        }
        else
        {
            return generate_http_response_405("POST");
        }
    }
    else if (strncmp(request->url, "/api/file/", 10) == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            char *filename = request->url + 10;
            char filepath[PATH_MAX];
            strcpy(filepath, "web/files/");
            strcat(filepath, filename);

            struct rfile *file = read_file(filepath);
            if (file != NULL)
            {
                return generate_http_response_200(get_mime_type(filename), file->content_length, file->content);
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else if (strcmp(request->url, "/api/video") == 0)
    {
        if (strcmp(request->method, "POST") == 0)
        {
            if (is_authenticated(request))
            {
                char *content_type = get_http_header_value(request->headers, "Content-Type");

                if (content_type != NULL && strstr(content_type, "multipart/form-data") != NULL)
                {
                    // get boundary
                    char *b = strstr(content_type, "boundary=");
                    b += 9; // skip "boundary="

                    char *boundary = malloc(strlen(b) + 3);
                    strcpy(boundary, "--"); // add prefix "--"
                    strcat(boundary, b);

                    handle_multipart_form_data_video(request, boundary);

                    char *msg = "Video Uploaded\n";
                    return generate_http_response_200("text/plain", strlen(msg), msg);
                }
            }
            else
            {
                return generate_http_response_401();
            }
        }
        else
        {
            return generate_http_response_405("POST");
        }
    }
    else if (strncmp(request->url, "/api/video/", 11) == 0)
    {
        if (strcmp(request->method, "GET") == 0)
        {
            char *filename = request->url + 11;
            char filepath[PATH_MAX];
            strcpy(filepath, "web/videos/");
            strcat(filepath, filename);

            struct rfile *file = read_file(filepath);
            if (file != NULL)
            {
                return generate_http_response_200(get_mime_type(filename), file->content_length, file->content);
            }
            else
            {
                return generate_http_response_404();
            }
        }
        else
        {
            return generate_http_response_405("GET");
        }
    }
    else
    {
        return generate_http_response_404();
    }

    return generate_http_response_500();
}

int handle_multipart_form_data_file(struct http_request *request, char *boundary)
{
    // printf("\n-- content --\n");
    // printf("%s", request->content);

    // get first block start
    char *block = strstr(request->content, boundary);
    block += strlen(boundary);

    // find block end
    char *block_end = NULL;
    for (int i = 0; i < request->content_length; i++)
    {
        if (strncmp(&block[i], boundary, strlen(boundary)) == 0)
        {
            block_end = &block[i];
            break;
        }
    }
    if (block_end == NULL)
    {
        perror("multipart/form wrong format\n");
        return 0;
    }

    // split into headers and content
    char *headers;
    char *content;
    unsigned char *binary_content;
    size_t content_length;

    char *separator = strstr(block, "\r\n\r\n");
    if (separator == NULL)
    {
        perror("multipart/form wrong format\n");
        return 0;
    }
    *separator = '\0';
    content = separator + 4;
    content_length = block_end - content - 2; // remove tailed \r\n

    // parse headers
    headers = block;
    strcat(headers, "\r\n");

    char *content_disposition = get_http_header_value(headers, "Content-Disposition");
    if (content_disposition == NULL)
    {
        perror("content_disposition not found\n");
        return 0;
    }

    char *filename = strstr(content_disposition, "filename=\"");
    if (filename == NULL)
    {
        perror("filename not found\n");
        return 0;
    }
    filename += 10; // skip "filename=\""
    filename = strtok(filename, "\"");
    filename = extrac_filename_from_path(filename);

    char *content_type = get_http_header_value(headers, "Content-Type");
    if (content_type == NULL)
    {
        perror("content type not found\n");
        return 0;
    }

    char *dir_name = change_to_absolute_path("web/files");
    create_dir_if_not_exist(dir_name);

    // save file
    char filepath[PATH_MAX];
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
            binary_content = (unsigned char *)content;
            fwrite(binary_content, sizeof(unsigned char), content_length, fp);
        }
        fclose(fp);
    }

    return 1;
}

int handle_multipart_form_data_video(struct http_request *request, char *boundary)
{
    // printf("\n-- content --\n");
    // printf("%s", request->content);

    // get first block start
    char *block = strstr(request->content, boundary);
    block += strlen(boundary);

    // find block end
    char *block_end = NULL;
    for (int i = 0; i < request->content_length; i++)
    {
        if (strncmp(&block[i], boundary, strlen(boundary)) == 0)
        {
            block_end = &block[i];
            break;
        }
    }
    if (block_end == NULL)
    {
        perror("multipart/form wrong format\n");
        return 0;
    }

    // split into headers and content
    char *headers;
    char *content;
    unsigned char *binary_content;
    size_t content_length;

    char *separator = strstr(block, "\r\n\r\n");
    if (separator == NULL)
    {
        perror("multipart/form wrong format\n");
        return 0;
    }
    *separator = '\0';
    content = separator + 4;
    content_length = block_end - content - 2; // remove tailed \r\n

    // parse headers
    headers = block;
    strcat(headers, "\r\n");

    char *content_disposition = get_http_header_value(headers, "Content-Disposition");
    if (content_disposition == NULL)
    {
        perror("content_disposition not found\n");
        return 0;
    }

    char *filename = strstr(content_disposition, "filename=\"");
    if (filename == NULL)
    {
        perror("filename not found\n");
        return 0;
    }
    filename += 10; // skip "filename=\""
    filename = strtok(filename, "\"");
    filename = extrac_filename_from_path(filename);

    char *content_type = get_http_header_value(headers, "Content-Type");
    if (content_type == NULL)
    {
        perror("content type not found\n");
        return 0;
    }

    char *dir_name = change_to_absolute_path("web/tmp");
    create_dir_if_not_exist(dir_name);

    // save file
    char filepath[PATH_MAX];
    strcpy(filepath, dir_name);
    strcat(filepath, "/");
    strcat(filepath, filename);

    FILE *fp = fopen(filepath, "w");
    if (fp != NULL)
    {
        binary_content = (unsigned char *)content;
        fwrite(binary_content, sizeof(unsigned char), content_length, fp);
        // printf("write %ld video file: %s\n", content_length, filename);
        fclose(fp);
    }

    exec_video_convert(filename);

    return 1;
}

int is_authenticated(http_request *request)
{
    char *authorization_code = get_http_header_value(request->headers, "Authorization");
    if (authorization_code == NULL)
    {
        return 0;
    }
    authorization_code += 6; // skip "Basic "

    size_t input_length = strlen(authorization_code);
    size_t output_length;
    char *username_password = (char *)base64_decode(authorization_code, input_length, &output_length);

    int valid = 0;

    char *secret_path = change_to_absolute_path("secret");
    FILE *fp = fopen(secret_path, "r");
    char line[256];
    while (fgets(line, sizeof(line), fp))
    {
        line[strcspn(line, "\n")] = '\0';
        if (strcmp(username_password, line) == 0)
        {
            valid = 1;
            break;
        }
    }
    fclose(fp);

    return valid;
}

void decode_http_header(struct http_request *request, char *header)
{
    char *tmp;

    tmp = strtok(header, " ");
    request->method = malloc(strlen(tmp) + 1);
    strcpy(request->method, tmp);

    tmp = strtok(NULL, " ");
    request->url = malloc(strlen(tmp) + 1);
    strcpy(request->url, percent_decode(tmp));

    strtok(NULL, "\r\n"); // skip HTTP/1.1

    tmp = strtok(NULL, "");
    request->headers = malloc(strlen(tmp) + 3);
    strcpy(request->headers, tmp);
    strcat(request->headers, "\r\n");
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

struct http_response *generate_http_response_200(char *content_type_str, size_t body_length, char *body)
{
    struct http_response *response = malloc(sizeof(struct http_response));
    response->content = malloc(BUFSIZE);

    strcpy(response->content, "HTTP/1.1 200 OK\r\n");
    append_http_header(response->content, "Server", "CN2024Server/1.0");

    if (body != NULL)
    {
        char body_length_str[20];
        sprintf(body_length_str, "%ld", body_length);

        append_http_header(response->content, "Content-Type", content_type_str);
        append_http_header(response->content, "Content-Length", body_length_str);
        strcat(response->content, "\r\n");

        size_t total_length = strlen(response->content) + body_length;
        if (total_length > BUFSIZE)
        {
            response->content = realloc(response->content, total_length + 1);
        }
        memcpy(response->content + strlen(response->content), body, body_length);

        response->content_length = total_length;
    }
    else
    {
        append_http_header(response->content, "Content-Length", "0");
        strcat(response->content, "\r\n");

        response->content_length = strlen(response->content);
    }

    return response;
}

struct http_response *generate_http_response_401()
{
    struct http_response *response = malloc(sizeof(struct http_response));
    response->content = malloc(BUFSIZE);

    strcpy(response->content, "HTTP/1.1 401 Unauthorized\r\n");
    append_http_header(response->content, "Server", "CN2024Server/1.0");
    append_http_header(response->content, "WWW-Authenticate", "Basic realm=\"B11902038\"");
    append_http_header(response->content, "Content-Type", "text/plain");
    append_http_header(response->content, "Content-Length", "13");
    strcat(response->content, "\r\nUnauthorized\n");

    response->content_length = strlen(response->content);

    return response;
}

struct http_response *generate_http_response_404()
{
    struct http_response *response = malloc(sizeof(struct http_response));
    response->content = malloc(BUFSIZE);

    strcpy(response->content, "HTTP/1.1 404 Not Found\r\n");
    append_http_header(response->content, "Server", "CN2024Server/1.0");
    append_http_header(response->content, "Content-Type", "text/plain");
    append_http_header(response->content, "Content-Length", "10");
    strcat(response->content, "\r\nNot Found\n");

    response->content_length = strlen(response->content);

    return response;
}

struct http_response *generate_http_response_405(char *allow_method)
{
    struct http_response *response = malloc(sizeof(struct http_response));
    response->content = malloc(BUFSIZE);

    strcpy(response->content, "HTTP/1.1 405 Method Not Allowed\r\n");
    append_http_header(response->content, "Server", "CN2024Server/1.0");
    append_http_header(response->content, "Allow", allow_method);
    append_http_header(response->content, "Content-Length", "0");
    strcat(response->content, "\r\n");

    response->content_length = strlen(response->content);

    return response;
}

struct http_response *generate_http_response_500()
{
    struct http_response *response = malloc(sizeof(struct http_response));
    response->content = malloc(BUFSIZE);

    strcpy(response->content, "HTTP/1.1 500 Internal Server Error\r\n");
    append_http_header(response->content, "Server", "CN2024Server/1.0");
    append_http_header(response->content, "Content-Length", "0");
    strcat(response->content, "\r\n");

    response->content_length = strlen(response->content);

    return response;
}

void append_http_header(char *content, char *key, char *value)
{
    strcat(content, key);
    strcat(content, ": ");
    strcat(content, value);
    strcat(content, "\r\n");
}

void disconnect_client(int idx)
{
    // printf("client %d disconnected\n", _clients[idx].fd);
    close(_clients[idx].fd);
    _clients[idx].fd = -1;

    if (idx == _max_index)
    {
        while (_clients[_max_index].fd < 0)
            _max_index--;
    }
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

    printf("read file %s with %ld btye\n", filepath, ret->content_length);

    return ret;
}

char *get_mime_type(char *filename)
{
    char *extension = strrchr(filename, '.');

    if (extension == NULL)
    {
        return "text/plain";
    }

    if (strcmp(extension, ".html") == 0)
    {
        return "text/html";
    }
    else if (strcmp(extension, ".mp4") == 0)
    {
        return "video/mp4";
    }
    else if (strcmp(extension, ".m4v") == 0)
    {
        return "video/mp4";
    }
    else if (strcmp(extension, ".m4s") == 0)
    {
        return "video/iso.segment";
    }
    else if (strcmp(extension, ".m4a") == 0)
    {
        return "audio/mp4";
    }
    else if (strcmp(extension, ".mpd") == 0)
    {
        return "application/dash+xml";
    }
    else
    {
        return "text/plain";
    }
}

void exec_video_convert(char *video_name)
{
    pid_t pid = fork();

    if (pid == 0)
    {
        char input_path[PATH_MAX];
        char* tmp_dir_name = change_to_absolute_path("web/tmp/");
        strcpy(input_path, tmp_dir_name);
        strcat(input_path, video_name);

        char* video_dir_name = change_to_absolute_path("web/videos/");
        create_dir_if_not_exist(video_dir_name);

        char output_dir[PATH_MAX];
        strcpy(output_dir, video_dir_name);
        strcat(output_dir, strtok(video_name, "."));
        create_dir_if_not_exist(output_dir);

        char output_path[PATH_MAX];
        strcpy(output_path, output_dir);
        strcat(output_path, "/dash.mpd");

        execlp("ffmpeg",
               "ffmpeg", "-re", "-i", input_path, "-c:a", "aac", "-c:v", "libx264",
               "-map", "0", "-b:v:1", "6M", "-s:v:1", "1920x1080", "-profile:v:1", "high",
               "-map", "0", "-b:v:0", "144k", "-s:v:0", "256x144", "-profile:v:0", "baseline",
               "-bf", "1", "-keyint_min", "120", "-g", "120", "-sc_threshold", "0", "-b_strategy", "0",
               "-ar:a:1", "22050", "-use_timeline", "1", "-use_template", "1",
               "-adaptation_sets", "id=0,streams=v id=1,streams=a", "-f", "dash", output_path,
               (char *)NULL);

        perror("exec failed");
        return;
    }
    else if (pid > 0)
    { // 父進程
      // printf("video conversion started: %s\n", video_name);
    }
    else
    {
        perror("fork failed");
    }

    return;
}

void create_dir_if_not_exist(char *dir_name)
{
    struct stat st = {0};
    if (stat(dir_name, &st) == -1)
    {
        mkdir(dir_name, 0700);
    }
}

char *modify_html(char *html, char *placeholder, char *content)
{
    char *new_html = malloc(strlen(html) + strlen(content) + 1);

    char *separator = strstr(html, placeholder);
    if (separator != NULL)
    {
        char *pre_content = html;
        char *post_content = separator + strlen(placeholder);
        *separator = '\0';

        strcpy(new_html, pre_content);
        strcat(new_html, content);
        strcat(new_html, post_content);
    }

    return new_html;
}

char *change_to_absolute_path(char *path)
{
    char *absolute_path = malloc(strlen(_root_dir) + strlen(path) + 1);
    strcpy(absolute_path, _root_dir);
    strcat(absolute_path, "/");
    strcat(absolute_path, path);
    return absolute_path;
}

char *extrac_filename_from_path(char *path)
{
    char *filename = strrchr(path, '/');
    if (filename == NULL)
    {
        return path;
    }
    return filename + 1;
}