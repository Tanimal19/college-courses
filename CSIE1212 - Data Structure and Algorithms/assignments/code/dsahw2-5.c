#include <stdio.h>
#include <stdlib.h>

struct NODE{
	int appetite;
	int color;
	int index;
	int Aindex;
};

struct COLOR{
	int color;
	int total;
	struct NODE* arr;
};

int N, M;

void mergeA(struct NODE* arr[N], int head, int mid, int tail){
	int n = tail-head+1;
	struct NODE* temp[n];

	int l = head;
	int r = mid+1;
	int i = 0;

	while(l <= mid && r <= tail){
		if(arr[l]->appetite <= arr[r]->appetite){
			temp[i] = arr[l];
			l++;
			i++;
		}
		else{
			temp[i] = arr[r];
			r++;
			i++;
		}
	}

	if(l > mid && r <= tail){
		while(r <= tail){
			temp[i] = arr[r];
			r++;
			i++;
		}
	}
	else if(r > tail && l <= mid){
		while(l <= mid){
			temp[i] = arr[l];
			l++;
			i++;
		}
	}

	for(int i = 0; i < n; i++){
		arr[head+i] = temp[i];
		arr[head+i]->Aindex = head+i;
	}
}

void mergeSortA(struct NODE* arr[N], int head, int tail){
	if (head < tail) {
		int mid = (head+tail)/2;
		mergeSortA(arr, head, mid);
		mergeSortA(arr, mid+1, tail);
		mergeA(arr, head, mid, tail);
	}
}

int bsColor(struct COLOR arr[], int l, int r, int key){
	while(l <= r){
		int m = l + (r - l) / 2;
		if (arr[m].color == key)
			return m;
		if (arr[m].color < key)
			l = m + 1;
		else
			r = m - 1;
	}
	return -1;
}

int main(){
	scanf("%d%d", &N, &M);
	struct NODE cats[N];
	struct NODE* A[N];
	struct COLOR C[N];
	int allcolor = 0;

	for(int i = 0; i < N; i++){
		scanf("%d", &cats[i].appetite);
		cats[i].index = i;
		cats[i].Aindex = i;
		A[i] = &cats[i];
	}
	// mergeSort by appetite
	mergeSortA(A, 0, N-1);

	for(int i = 0; i < N; i++){
		int key;
		scanf("%d", &key);
		int find = -1;
		for(int j = 0; j < allcolor; j++){
			if(C[j].color == key){
				find = j;
				break;
			}
		}
		printf("result[%d at %d]", key, find);
		if(find == -1){
			C[allcolor].color = key;
			C[allcolor].total = 0;
			C[allcolor].arr = malloc( (C[allcolor].total+1) * sizeof(struct NODE) );
			find = allcolor;
			allcolor ++;
		}
		else{
			C[find].arr = malloc( (C[find].total+1) * sizeof(struct NODE) );
		}
		C[find].arr[C[find].total] = cats[i];
		C[find].total ++;
		cats[i].color = key;
	}


	for(int i = 0; i < allcolor; i++){
		printf("[%d %d]\n", C[i].color, C[i].total);
	}

	return 0;
}

