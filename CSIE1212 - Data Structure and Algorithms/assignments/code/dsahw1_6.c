#include <stdio.h>
#include <stdlib.h>
#define mSIZE 10
#define dSIZE 10

typedef struct node{
	int rank;
	int toast;
	int i;
	int j;
	struct node* left;
	struct node* right;
	struct node* up;
	struct node* down;
}Node;

int n, m;
int t1, R, t2;

// -top 0 -left 1 -bottom 2 -right 3
Node* bakery[mSIZE][mSIZE];

int c = 0;
Node* closed[mSIZE*mSIZE];
int v = 0;
Node* visited[mSIZE*mSIZE];

Node* search(int r){
	for(int i=1 ; i<n+1 ; i++)
		for(int j=1 ; j<m+1 ; j++)
			if(bakery[i][j]->rank == r){	
				return bakery[i][j];
			}
}

// left 0 right 1 up 2 down 3
int minrank(Node* gather){
	int min = n*m, dir = -1;
	
	if(gather->left != NULL && gather->left->rank < min){
		dir = 0;
		min = gather->left->rank;
	}
	if(gather->right != NULL && gather->right->rank < min){
		dir = 1;
		min = gather->right->rank;
	}
	if(gather->up != NULL && gather->up->rank < min){
		dir = 2;
		min = gather->up->rank;
	}
	if(gather->down != NULL && gather->down->rank < min){
		dir = 3;
		min = gather->down->rank;
	}
		
	return dir;
}

void closebakery(Node* n){
	int i = n->i, j = n->j;
	if(bakery[i][j-1] != NULL) bakery[i][j-1]->right = n->right;
	if(bakery[i][j+1] != NULL) bakery[i][j+1]->left = n->left;
	if(bakery[i-1][j] != NULL) bakery[i-1][j]->down = n->down;
	if(bakery[i+1][j] != NULL) bakery[i+1][j]->up = n->up;
}

void training(int r, int l, int s){
	Node* gather = search(r);
	Node* visit;
	for(int i=0 ; i<l ; i++){
		switch(minrank(gather)){
		case -1: // can't find
			return;
		case 0: // left
			visit = gather->left;
			visit->right = gather->right;
			break;
		case 1: // right
			visit = gather->right;
			visit->left = gather->left;
			break;
		case 2: // up
			visit = gather->up;
			visit->down = gather->down;
			break;
		case 3: // down
			visit = gather->down;
			visit->up = gather->up;
			break;
		}

		visit->toast -= s;
		// printf("[%d %d]\n", visit->rank, visit->toast);

		if(gather->toast <= 0){
			closebakery(gather);
			closed[c] = gather;
			c++;
		}
		else{
			visited[v] = gather;
			v++;
		}

		gather = visit;
	}

	if(gather->toast <= 0){
		closebakery(gather);
		closed[c] = gather;
		c++;
	}
	else{
		visited[v] = gather;
		v++;
	}
}

void recover(Node* n){
	int i = n->i, j = n->j;
	n->left = bakery[i][j-1];
	n->right = bakery[i][j+1];
	n->up = bakery[i-1][j];
	n->down = bakery[i+1][j];
}

int main(){

	scanf("%d%d", &n, &m);
	for(int i=0 ; i<n+2 ; i++){
		for(int j=0 ; j<m+2 ; j++){
			if(i==0 || i==n+1 || j==0 || j==m+1){
				bakery[i][j] = NULL;
			}
			else{
				bakery[i][j] = malloc(sizeof(Node));
				bakery[i][j]->i = i;
				bakery[i][j]->j = j;
				scanf("%d", &bakery[i][j]->rank);
			}
		}
	}

	for(int i=1 ; i<n+1 ; i++){
		for(int j=1 ; j<m+1 ; j++){
			bakery[i][j]->left = bakery[i][j-1];
			bakery[i][j]->right = bakery[i][j+1];
			bakery[i][j]->up = bakery[i-1][j];
			bakery[i][j]->down = bakery[i+1][j];
			scanf("%d", &bakery[i][j]->toast);
		}
	}


	scanf("%d%d%d", &t1, &R, &t2);
	int r, l, s;
	for(int k=0 ; k<t1 ; k++){
		scanf("%d%d%d", &r, &l, &s);
		training(r, l, s);

		for(int i=0 ; i<v ; i++)
			recover(visited[i]);
		v = 0;

		for(int i=0 ; i<c ; i++)
			closebakery(closed[i]);
	}

	#ifdef DEBUG
	for(int i=0 ; i<R ; i++){
		scanf("%d%d%d", &r, &l);
	}
	#endif
	
	for(int i=0 ; i<t2 ; i++){
		scanf("%d%d%d", &r, &l, &s);
		training(r, l, s);

		for(int i=0 ; i<v ; i++)
			recover(visited[i]);
		v = 0;

		for(int i=0 ; i<c ; i++)
			closebakery(closed[i]);
	}
	
	for(int i=1 ; i<n+1 ; i++){
		for(int j=1 ; j<m+1 ; j++){
			printf("%d ", bakery[i][j]->toast <= 0 ? 0 : bakery[i][j]->toast);
		}
		printf("\n");
	}

	return 0;
}