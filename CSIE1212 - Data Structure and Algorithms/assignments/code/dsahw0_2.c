#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define MAX 25
#define n 0
#define m 1

int N, M;
int board[MAX][MAX];
int Ncule[MAX][MAX], Mcule[MAX][MAX];

bool recur(int board[MAX][MAX], int now_n, int now_m){

	if(now_n >= N-1 && now_m >= M-1){
		// #ifdef DEBUG
		/* if board valid */
		int segment, length;

		/* check N */
		for(int i=0 ; i<N ; i++){
			segment = 0;
			length = 0;
			for(int j=0 ; j<M ; j++){
				if(board[i][j] == 2) length ++;
				if(board[i][j] == 1 || j == M-1){
					if(length != 0){
						if(length != Ncule[i][segment+1]) return false;
						segment ++;
					}
					length = 0;
				}
			}
			if(segment != Ncule[i][0]) return false;
		}

		/* check M */
		for(int i=0 ; i<M ; i++){
			segment = 0;
			length = 0;
			for(int j=0 ; j<N ; j++){
				if(board[j][i] == 2) length ++;
				if(board[j][i] == 1 || j == N-1){
					if(length != 0){
						if(length != Mcule[i][segment+1]) return false;
						segment ++;
					}
					length = 0;
				}
			}
			if(segment != Mcule[i][0]) return false;
		}
		// #endif

		for(int i=0 ; i<N ; i++){
			for(int j=0 ; j<M ; j++){
				printf("%c", board[i][j] == 2 ? 'o':'_');
			}
			if(i != N-1) printf("\n");
		}
		return true;
	}
	else{
		
		/* find next empty cell */
		if(now_m == M-1){
			now_m = 0;
			now_n ++; 
		}
		else{
			now_m ++;
		}

		/* paint black */
		board[ now_n ][ now_m ] = 2;
		if( recur(board, now_n, now_m) ) return true;

		/* paint white */
		board[ now_n ][ now_m ] = 1;
		if( recur(board, now_n, now_m) ) return true;

		/* make empty i.e. recover */
		board[ now_n ][ now_m ] = 0;
		return false;
	}
}

int main(){
	scanf("%d%d", &N, &M);
	for(int i=0 ; i<N ; i++){
		scanf("%d", &Ncule[i][0]);
		for(int j=0 ; j<Ncule[i][0] ; j++)
			scanf("%d", &Ncule[i][j+1]);
	}
	for(int i=0 ; i<M ; i++){
		scanf("%d", &Mcule[i][0]);
		for(int j=0 ; j<Mcule[i][0] ; j++)
			scanf("%d", &Mcule[i][j+1]);
	}

	recur(board, 0, -1);
	return 0;
}