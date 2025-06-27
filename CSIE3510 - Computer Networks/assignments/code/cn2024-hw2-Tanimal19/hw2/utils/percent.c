#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char* percent_encode(char* str);
char* percent_decode(char* str);

char *percent_encode(char *str) {
    size_t len = strlen(str);
    size_t new_len = len * 3 + 1;
    char *encoded = malloc(new_len);

    char *p = encoded;
    for (size_t i = 0; i < len; i++) {
        if (isalnum((unsigned char)str[i]) || strchr("-_.~", str[i]) != NULL) {
            *p++ = str[i];
        } else {
            p += sprintf(p, "%%%02X", (unsigned char)str[i]);
        }
    }
    *p = '\0';
    return encoded;
}

char *percent_decode(char *str) {
    size_t len = strlen(str);
    char *decoded = malloc(len + 1);

    char *p = decoded;
    for (size_t i = 0; i < len; i++) {
        if (str[i] == '%' && isxdigit((unsigned char)str[i + 1]) && isxdigit((unsigned char)str[i + 2])) {
            char hex[3] = { str[i + 1], str[i + 2], '\0' };
            *p++ = (char)strtol(hex, NULL, 16);
            i += 2;
        } else {
            *p++ = str[i];
        }
    }
    *p = '\0';
    return decoded;
}
