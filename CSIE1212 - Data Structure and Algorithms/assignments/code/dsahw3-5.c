#include <stdio.h>
#include <stdlib.h>
#define MAXN 1000

struct set;
struct node;

struct set{
	int location;
	int time;
	int atksum;
	int minushel;
	struct node* head;
	struct node* tail;
};

struct node{
	int index, hel, atk, time, alive;
	struct node* next;
	struct set* parent;
};

int n, m;
struct node knight[MAXN];

void Make_Set(int i){
	struct set* newset = malloc(sizeof(struct set));
	newset->location = i+1;
	newset->time = 0;
	newset->atksum = knight[i].atk;
	newset->minushel = 0;
	newset->head = &knight[i];
	newset->tail = newset->head;
	knight[i].parent = newset;
}

int main(){
	scanf("%d%d", &n, &m);

	for(int i = 0; i < n ; i++){
		scanf("%d", &knight[i].hel);
		knight[i].index = i+1;
		knight[i].time = 0;
		knight[i].next = NULL;
		knight[i].alive = 1;
	}

	for(int i = 0; i < n ; i++){
		scanf("%d", &knight[i].atk);
		Make_Set(i);
	}
	
	for(int i = 0; i < m ; i++){
		int attack, target;
		scanf("%d%d", &attack, &target);
		attack --;
		target --;

		// set1 攻擊 set2
		struct set* set1 = knight[attack].parent;
		struct set* set2 = knight[target].parent;

		knight[attack].hel += set1->minushel;
		if(knight[attack].hel <= 0){
			knight[attack].alive = 0;
		}

		knight[attack].hel += set2->minushel;
		if(knight[attack].hel <= 0){
			knight[attack].alive = 0;
		}

		if(knight[attack].alive == 1 && knight[target].alive == 1 && set1->location != set2->location){
			// 將 set1 併入 set2
			// 雖然我們沒有把 set1 裡面的元素的 parent 改掉，但我們把它串在一起，所以之後在算 set2 時還是會算到
			

			(set2->tail)->next = set1->head;
			set2->tail = set1->tail;

			set2->minushel -= set1->atksum;
			set2->atksum += set1->atksum; 
			//set1->minushel = set2->minushel;
		}

		for(int j = 0; j < n; j++)
			printf("knight[%d] hel/atk:%d/%d time:%d\nset:%d atksum:%d minushel:%d\n", 
				knight[j].index, knight[j].hel, knight[j].atk, knight[j].time, knight[j].parent->location, knight[j].parent->atksum, knight[j].parent->minushel);
	}

	#ifdef DEBUG
	for(int i = 0; i < n; i++){
		if(knight[i].alive == 1)
			knight[i].time += knight[i].parent->time;
		printf("%d ", knight[i].time);
	}
	#endif

	return 0;
}