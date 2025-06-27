#include <stdio.h>
#include <stdlib.h>

int Find_Pair(int A[], int l){
	int now = A[1];
	int begin = 1;
	int cursor;

	for(int i=1 ; i<=l ; i++){
		if(A[i] >= now*2){
			cursor = begin;
			begin = i;
			now = now*2;
		}

		if(A[i] % 2 == 0 && cursor < begin){
			while(A[cursor] == 0){
				if(cursor >= begin)
					break;
				cursor ++;
			}
			if(A[cursor]*2 == A[i]){
				A[cursor] = 0;
				A[i] = 0;
				cursor++;
			}
		}
	}

	for(int i=1 ; i<=l ; i++){
		if(A[i] != 0) return 0;
	}

	return 1;
}


int main(){
	int l;
	scanf("%d", &l);
	int A[100];
	for(int i=1 ; i<=l ; i++){
		scanf("%d", &A[i]);
	}

	printf("%d", Find_Pair(A, l));

	return 0;
}