#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fs.h"

#define INPUTSIZE 20
#define NAMELEN 11
#define DEPTH 5
#define RD 0
#define WR 1

typedef struct {
    int dirNum;
    int fileNum;
} DATA;

DATA getEntry(char *path, char *key, DATA ret);
char* strcat(char *str1, char *str2);

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("argument num incorrect\n");
        exit(0);
    }

    char rpath[NAMELEN], key[NAMELEN];
    strcpy(rpath, argv[1]);
    strcpy(key, argv[2]);

    /* FORK */
    int CtoP[2];
    if (pipe(CtoP) < 0)
        printf("pipe error\n");

    int pid;
    if ((pid = fork()) < 0) {
		printf("fork error\n");
	} else if (pid == 0) {
		/* CHILD */
        close(CtoP[RD]);

        DATA ret;
        ret.dirNum = 0;
        ret.fileNum = 0;

        /* CHECK ROOT */
        int rfd;
        struct stat rst;

        if ((rfd = open(rpath, 0)) < 0) {
            printf("%s [error opening dir]\n", rpath);
        }
        else {
            fstat(rfd, &rst);
            if (rst.type == T_DIR) {
                ret = getEntry(rpath, key, ret);
                ret.dirNum--;
            }
            else {
                printf("%s [error opening dir]\n", rpath);
            }
        }
        
        /* SEND TO PARENT */
        write(CtoP[WR], &ret, sizeof(DATA));
        printf("\n");

        exit(0);

	} else {
		/* PARENT */
        close(CtoP[WR]);

        DATA ans;
        read(CtoP[RD], &ans, sizeof(DATA));

        printf("%d directories, %d files\n", ans.dirNum, ans.fileNum);
        
        int status;
        wait(&status);

        exit(0);
	}

    exit(0);
}

DATA getEntry(char *path, char *key, DATA ret) {

    /* COUNT KEY */
    int count = 0;
    for (int i=0 ; i<strlen(path) ; i++) {
        if (path[i] == key[0])
            count++;
    }
    printf("%s %d\n", path, count);

    /* CHECK FILE TYPE */
    int fd;
    struct dirent entry;
    struct stat st;

    fd = open(path, 0);
    fstat(fd, &st);

    if (st.type == T_FILE) {
        ret.fileNum++;
    }
    else if (st.type == T_DIR) {
        ret.dirNum++;
        
        while(read(fd, &entry, sizeof(entry)) == sizeof(entry)){
            if (strcmp(entry.name, ".") == 0 || strcmp(entry.name, "..") == 0)
			    continue;
      
            char newpath[NAMELEN*(DEPTH+1)];
            strcpy(newpath, path);
            strcat(newpath, "/");
            strcat(newpath, entry.name);

            ret = getEntry(newpath, key, ret);
        }
    }

    return ret;
}

char* strcat(char *str1, char *str2) {
    str1 = str1 + strlen(str1);
    strcpy(str1, str2);
    return str1;
} 