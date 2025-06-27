// malloc realloc 會花太久時間 

#include <stdio.h>
#include <stdlib.h>

// every company formed a min heap
struct node{
    long long price;
    long long expired;
};

// all companies formed a tree
struct company{
    long long parent;
    long long* childlist;
    long long childlen;
    struct node* heap;
    long long heaplen;
    long long minprice;
    long long melon;
};

long long N, M, C; // companies, sales event, money

void swap(struct node* A, struct node* B){
    long long tprice, texpired;
    tprice = A->price;
    texpired = A->expired;
    A->price = B->price;
    A->expired = B->expired;
    B->price = tprice;
    B->expired = texpired;
}

long long calmelon(long long* tree, struct company* dsa, long long i, long long count){
    for(long long j = 0; j < dsa[i].childlen; j++){
        count = calmelon(tree, dsa, dsa[i].childlist[j], count);
        dsa[i].melon += dsa[dsa[i].childlist[j]].melon;
    }
    tree[count] = i;
    count ++;
    return count;
}

signed main(){

    scanf("%lld%lld%lld", &N, &M, &C);
    struct company DSA[N];
    long long tree[N];

    // initialize companies DSA
    for(long long i = 0; i < N; i++){
        DSA[i].childlen = 0;
        DSA[i].childlist = malloc(DSA[i].childlen * sizeof(long long));
        DSA[i].heaplen = 0;
        DSA[i].heap = malloc(M * sizeof(struct node));
        DSA[i].minprice = 0;
        DSA[i].melon = 1;
    }

    for(long long i = 0; i < N; i++){
        if(i > 0){
            long long p;
            scanf("%lld", &p);
            p = p-1;
            DSA[i].parent = p;
            DSA[p].childlen ++;
            DSA[p].childlist = realloc(DSA[p].childlist, (DSA[p].childlen) * sizeof(long long));
            DSA[p].childlist[DSA[p].childlen-1] = i;
        }
    }

    calmelon(tree, DSA, 0, 0);

    // each day
    for(long long j = 0; j < M; j++){
        // each company
        for(long long i = 0; i < N; i++){
            // insert sales plan into company heap 
            long long period;
            scanf("%lld%lld", &DSA[i].heap[DSA[i].heaplen].price, &period);
            // expired day = current day (j) + period
            DSA[i].heap[DSA[i].heaplen].expired = j + period;
            
            long long cursor = DSA[i].heaplen;
            while(cursor > 0){
                long long parent = (cursor%2 == 0)? (cursor-2)/2 : (cursor-1)/2;
                if(DSA[i].heap[cursor].price < DSA[i].heap[parent].price){
                    swap(&DSA[i].heap[cursor], &DSA[i].heap[parent]);
                    cursor = parent;
                }
                else{
                    break;
                }
            }
            DSA[i].heaplen++;

            // get minprice (root of heap)
            // check if expired
            while(DSA[i].heap[0].expired < j){
                // remove root and copy tail to root
                DSA[i].heaplen --;
                DSA[i].heap[0].price = DSA[i].heap[DSA[i].heaplen].price;
                DSA[i].heap[0].expired = DSA[i].heap[DSA[i].heaplen].expired;

                // heapify
                cursor = 0;
                while(cursor < DSA[i].heaplen){
                    long long min = cursor;
                    long long l = cursor*2 + 1;
                    long long r = cursor*2 + 2;

                    if(l < DSA[i].heaplen && DSA[i].heap[l].price < DSA[i].heap[min].price)
                        min = l;
                    if(r < DSA[i].heaplen && DSA[i].heap[r].price < DSA[i].heap[min].price)
                        min = r;
                    if(min != cursor){
                        swap(&DSA[i].heap[cursor], &DSA[i].heap[min]);
                        cursor = min;
                    }
                    else{
                        break;
                    }
                } 
            }
            
            DSA[i].minprice = DSA[i].heap[0].price;
        }

        long long maxmelon = 0;
        for(long long i = 0; i < N; i++){
            if(tree[i] != 0)
                DSA[DSA[tree[i]].parent].minprice += DSA[tree[i]].minprice;
            if(DSA[tree[i]].minprice <= C && DSA[tree[i]].melon > maxmelon)
                maxmelon = DSA[tree[i]].melon;
        }

        printf("%lld\n", maxmelon);
    }

    return 0;
}
