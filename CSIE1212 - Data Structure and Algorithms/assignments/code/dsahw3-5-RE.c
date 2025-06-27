#include <stdio.h>
#include <stdlib.h>
#define MAXN 1000

struct set;
struct node;

struct set{
	int location;
	// int size;
	int atksum;
	struct node* head;
	struct node* tail;
};

struct node{
	int index, hel, atk, time, alive;
	struct node* next;
	struct node* prev;
	struct set* parent;
};

int n, m;
struct node knight[MAXN];

void Make_Set(int i){
	struct set* newset = malloc(sizeof(struct set));
	newset->location = i+1;
	// newset->size = 1;
	newset->atksum = knight[i].atk;
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
		knight[i].prev = NULL;
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

			// 每個 set2 裡面的 knight 扣血
			struct node* cur = set2->head;
			while(cur != NULL){
				cur->hel -= set1->atksum;
						
				if(cur->hel <= 0){
					set2->atksum -= cur->atk;
					cur->alive = 0;
				
					// 把 cur 從 set 中移除
					if(cur->prev == NULL){  // cur 是第一個
						set2->head = cur->next;
						if(cur->next != NULL) cur->next->prev = NULL;
					}
					else{
						cur->prev->next = cur->next;
						if(cur->next != NULL) cur->next->prev = cur->prev;
					}
				}

				cur = cur->next;
			}

			// 更改 set1 裡面的 knight 的位置
			cur = set1->head;
			while(cur != NULL){
				if(cur->alive == 1){
					cur->time ++;
					cur->parent = set2;
				}

				cur = cur->next;
			}

			if(set2->head == NULL) set2->head = set1->head;
			(set2->tail)->next = set1->head;
			(set1->head)->prev = set2->tail;
			set2->tail = set1->tail;

			//set2->size += set1->size;
			set2->atksum += set1->atksum;

			printf("set[%d]:",set1->location);
			cur = set1->head;
			while(cur != NULL){
				printf("%d ", cur->index);
				cur = cur->next;
			}
			printf("\n");
			printf("set[%d]:",set2->location);
			cur = set2->head;
			while(cur != NULL){
				printf("%d ", cur->index);
				cur = cur->next;
			}
			printf("\n");

			free(set1);
		}

		//for(int j = 0; j < n; j++)
			//printf("knight[%d] hel/atk:%d/%d time:%d set:%d\n", 
				//knight[j].index, knight[j].hel, knight[j].atk, knight[j].time, knight[j].parent->location);
	}

	for(int i = 0; i < n; i++)
		printf("%d ", knight[i].time);

	return 0;
}