#include <stdio.h>
#include <stdlib.h>
#define MAXN 1000

struct set;
struct node;

struct set{
	int location;
	int time;
	//int size;
	int atksum;
	//int maxhel;
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
	//newset->size = 1;
	newset->atksum = knight[i].atk;
	//newset->maxhel = knight[i].hel;
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

		if(knight[attack].alive == 1 && knight[target].alive == 1 && set1->location != set2->location){
			set1->time ++;

			// 先將 set2 併入 set1 ，然後再將 set1 的 location 改成 set2 的 location 
			// 每個 set2 裡面的 knight 扣血
			struct node* cur = set2->head;
			while(cur != NULL){
				if(cur->alive == 1){
					cur->hel -= set1->atksum;
					
					if(cur->hel <= 0){
						set2->atksum -= cur->atk;
						cur->time += set2->time;
						cur->alive = 0;
						
					}
					else{
						cur->time -= set1->time;
						cur->time += set2->time;
						cur->parent = set1;
					}
				}
				
				cur = cur->next;
			}

			(set1->tail)->next = set2->head;
			set1->tail = set2->tail;
			
			set1->atksum += set2->atksum;
			set1->location = set2->location;
			//set1->size += set2->size;
			//set2->maxhel = (set2->maxhel > set1->maxhel) ? set2->maxhel : set1->maxhel;
		
			free(set2);
		}

		//for(int j = 0; j < n; j++)
			//printf("knight[%d] hel/atk:%d/%d time:%d set:%d\n", 
				//knight[j].index, knight[j].hel, knight[j].atk, knight[j].time, knight[j].parent->location);
	}

	for(int i = 0; i < n; i++){
		if(knight[i].alive == 1)
			knight[i].time += knight[i].parent->time;
		printf("%d ", knight[i].time);
	}

	return 0;
}