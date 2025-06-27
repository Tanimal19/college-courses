#include <stdio.h>
#include <stdlib.h>

typedef struct player
{
	int index;
	int atk;
}Player;


int N, M;


int main(){
	scanf("%d%d", &N, &M);

	Player alive[M];
	int tail = 0, alen = 0;
	Player enter;
	for(int round=1 ; round<=N ; round++){
		scanf("%d", &enter.atk);
		enter.index = round;
		printf("Round %d:", round);
		
		int c = alen;
		while(c > 0){
			if(enter.atk <= alive[tail].atk){
				tail++;
				if(tail >= M){
					tail = 0;
				}
				break;
			}

			printf(" %d", alive[tail].index);
			alive[tail].index = 0;
			alive[tail].atk = 0;
			alen--;
			tail--;
			if(tail < 0){
				tail = M-1;
			}

			c--;
		}
		
		alen++;
		if(alen > M){
			// Revolution
			printf(" %d", alive[tail].index);
			alen--;
		}
		alive[tail] = enter;
		
		printf("\n");

		#ifdef PRINT
		c = alen;
		int j = tail;
		while(c > 0){
			printf("[%d][%d %d] -- ", j, alive[j].index, alive[j].atk);
			j--;
			if(j < 0){
				j = M-1;
			}
			c--;
		}
		printf("\n");
		#endif
	}

	printf("Final:");
	while(alen > 0){
		printf(" %d", alive[tail].index);
		tail--;
		if(tail < 0){
			tail = M-1;
		}
		alen--;
	}
	
	return 0;
}