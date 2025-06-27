#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SIZE 1000

int slen, plen;
char spell[SIZE], pattern[SIZE];
int ans;
int match[SIZE];

// Fills lps[] for given pattern pat[0..M-1]
void computeLPS(char* pat, int patlen, int* lps){
    int len = 0;
    lps[0] = 0;
 
    int i = 1;
    while(i < patlen){
        if(pat[i] == pat[len]){
            len++;
            lps[i] = len;
            i++;
        }
        else{
            if(len != 0){
                len = lps[len - 1];
            }
            else{
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch_SP(char* pat, int patlen, char* pat2, int patlen2){
    int lps[patlen];
    computeLPS(pat, patlen, lps);
 
    int head, tail, len, i;

    int s = 0; // index for spell[]
    int p = 0; // index for pat[]
    while((slen - s) >= (patlen - p)){
        if(pat[p] == spell[s]){
            s++;
            p++;
        }
 
        if(p == patlen){
        	// match at spell[s-p]
        	head = s - p; // head of matched pat
        	tail = s - 1; // tail of matched pat
        	len = patlen;

			i = 1;
        	while(tail+i < slen && i <= patlen2){
        		if(spell[tail+i] != pat2[i-1])
        			break;

        		len++;
        		i++;
        	}

        	if(len == plen && match[s-p] != -1){
        		match[s-p] = -1;
        		ans ++;
        	}

        	i = 1;
        	while(head-i >= 0 && i <= patlen2){
        		if(spell[head-i] != pat2[patlen2-i])
        			break;

        		if(len >= plen-i && match[head-i] != -1){
        			match[head-i] = -1;
        			ans ++;
        		}

        		i++;
        	}

            p = lps[p - 1];
        }
        else if (s < slen && pat[p] != spell[s]) {
            if (p != 0)
                p = lps[p - 1];
            else
                s++;
        }
    }
}

void KMPSearch(char* pat, int patlen, char* pat2, int patlen2){
    int lps[patlen];
    computeLPS(pat, patlen, lps);
 
    int head, tail, len, i;

    int s = 0; // index for spell[]
    int p = 0; // index for pat[]
    while((slen - s) >= (patlen - p)){
        if(pat[p] == spell[s]){
            s++;
            p++;
        }
 
        if(p == patlen){
        	// match at spell[s-p]
        	head = s - p; // head of matched pat
        	tail = s - 1; // tail of matched pat
        	len = patlen;

			i = 1;
        	while(tail+i < slen && i <= patlen2){
        		if(spell[tail+i] != pat2[i-1])
        			break;

        		len++;
        		i++;
        	}

        	if(len == plen && match[s-p] != -1){
        		match[s-p] = -1;
        		ans ++;
        	}

        	i = 1;
        	while(head-i >= 0 && i < patlen2){
        		if(spell[head-i] != pat2[patlen2-i])
        			break;

        		if(len >= plen-i && match[head-i] != -1){
        			match[head-i] = -1;
        			ans ++;
        		}

        		i++;
        	}

            p = lps[p - 1];
        }
        else if (s < slen && pat[p] != spell[s]) {
            if (p != 0)
                p = lps[p - 1];
            else
                s++;
        }
    }
}

int main(){
	ans = 0;

	scanf("%d%d", &slen, &plen);
	scanf("%s", spell);
	scanf("%s", pattern);

	int halflen = (plen+1)/2;
	char* firsthalf = &pattern[0];
	char* seconhalf = (plen%2 == 0) ? &pattern[halflen] : &pattern[halflen-1];
	char* another;

	another = &pattern[halflen];
	if(plen%2 == 0){
		KMPSearch(firsthalf, halflen, another, plen-halflen);
	}
	else{
		KMPSearch_SP(firsthalf, halflen, another, plen-halflen);
	}
	
	another = &pattern[0];
	KMPSearch(seconhalf, halflen, another, plen-halflen);

	printf("%d", ans);
	return 0;
}