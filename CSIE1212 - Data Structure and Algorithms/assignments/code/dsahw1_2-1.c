#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int find[2];

int *findmissing(int A[], int l){
	
	for(int i=0 ; i<2 ; i++){
		int tail = l;
		int head = 1;
		int mid;

		while( head <= tail ){
			mid = (head+tail) / 2;
			printf("%d %d %d\n", head, mid, tail);
			if(A[mid] > mid+i){
				tail = mid-1;
			}
			else if(A[mid] <= mid+i){
				head = mid+1;
			}
		}
		find[i] = (A[mid] == mid+i)? mid+1+i : mid+i;
	}
	return find;
}

int main(){
	int l;
	scanf("%d", &l);
	int A[100];
	for(int i=1 ; i<=l ; i++){
		scanf("%d", &A[i]);
	}

	int *ans = findmissing(A, l);
	printf("%d %d", ans[0], ans[1]);

	return 0;
}