#include <stdio.h>
#include <stdlib.h>
#define MAXN 1000

int N, M;

struct ctable{
    int color;
    int head; // index of C[i]
    int tail;
};
int total_color;

struct cat{
	int index;
	int Aindex; // index of A[i]
	int appetite;
	int Cindex; // index of C[i]
	int color;
};

void merge_appetite(struct cat* arr[], int l, int m, int r){
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    struct cat* L[n1];
    struct cat* R[n2];
 
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while(i < n1 && j < n2){
        if(L[i]->appetite <= R[j]->appetite){
            arr[k] = L[i];
            i++;
        }
        else{
            arr[k] = R[j];
            j++;
        }
        arr[k]->Aindex = k;
        k++;
    }

    while(i < n1){
        arr[k] = L[i];
        arr[k]->Aindex = k;
        i++;
        k++;
    }
    while(j < n2){
        arr[k] = R[j];
        arr[k]->Aindex = k;
        j++;
        k++;
    }
}

void mergeSort_appetite(struct cat* arr[], int l, int r){
    if(l < r){
        int m = l + (r - l) / 2;
        mergeSort_appetite(arr, l, m);
        mergeSort_appetite(arr, m + 1, r);
        merge_appetite(arr, l, m, r);
    }
}

void merge_color(struct cat* arr[], int l, int m, int r){
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    struct cat* L[n1];
    struct cat* R[n2];
 	
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while(i < n1 && j < n2){
        if(L[i]->color < R[j]->color){
            arr[k] = L[i];
            i++;
            arr[k]->Cindex = k;
        	k++;
        }
        else if(L[i]->color > R[j]->color){
            arr[k] = R[j];
            j++;
            arr[k]->Cindex = k;
        	k++;
        }
        else{
        	arr[k] = L[i];
            i++;
            arr[k]->Cindex = k;
        	k++;
        	arr[k] = R[j];
            j++;
            arr[k]->Cindex = k;
        	k++;
        }
    }

    while(i < n1){
        arr[k] = L[i];
        arr[k]->Cindex = k;
        i++;
        k++;
    }
    while(j < n2){
        arr[k] = R[j];
        arr[k]->Cindex = k;
        j++;
        k++;
    }
}

void mergeSort_color(struct cat* arr[], int l, int r){
    if(l < r){
        int m = l + (r - l) / 2;
        mergeSort_color(arr, l, m);
        mergeSort_color(arr, m + 1, r);
        merge_color(arr, l, m, r);
    }
}

int find_color(struct ctable ctable[], int color){
	int l = 0;
	int r = total_color;
	while(l <= r){
        int m = (l+r)/2;
 
        if(ctable[m].color == color)
            return m;
        if(ctable[m].color < color)
            l = m + 1;
        else
            r = m - 1;
    }
    // no corresponding color
    return -1;
}

int main(){
	scanf("%d%d", &N, &M);

	struct cat cats[N];
	struct cat* A[N];
	struct cat* C[N];
    struct ctable ctable[N];

	// get appetite and sort A[] with appetite
	for(int i = 0; i < N; i++){
		cats[i].index = i;
		cats[i].Aindex = i;
		scanf("%d", &cats[i].appetite);
		A[i] = &cats[i];
	}
	mergeSort_appetite(A, 0, N-1);

    // get color and sort A[] with color
    // each color section is sorted by appetite  
	for(int i = 0; i < N; i++){
		cats[i].Cindex = i;
		scanf("%d", &cats[i].color);
		C[i] = A[i];
	}
	mergeSort_color(C, 0, N-1);

    // setting color table
    int current = C[0]->color;
    int index = 0;
    ctable[0].color = C[0]->color;
    ctable[0].head = 0;
    for(int i = 1; i < N; i++){
        if(C[i]->color != current){
            current = C[i]->color;
            ctable[index].tail = i-1;
            index++;
            ctable[index].color = current;
            ctable[index].head = i; 
        }
    }
    ctable[index].tail = N-1;
    total_color = index+1;

    for(int i = 0; i < M; i++){
        int c, l, r, k, s, t;
        
        int ti;
        int ans = 0;

        int mode;
        scanf("%d", &mode);
        switch(mode){
        // question
        case 1:
            scanf("%d%d%d", &c, &l, &r);
            ti = find_color(ctable, c);
            ans = 0;
            if(ti != -1){
                for(int j = ctable[ti].head; j <= ctable[ti].tail; j++){
                    if(C[j]->appetite >= l && C[j]->appetite <= r)
                        ans++;
                }
            }
            printf("%d\n", ans);
            break;

        // greedy cat
        case 2:
            break;
        // magic cat
        case 3:
            break;
        }
    }


	return 0;
}