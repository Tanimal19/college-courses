#include <stdio.h>
#include <stdlib.h>
#define divisor 1000000007

typedef struct Node{
	int color;  // 0 = red, 1 = black
	long long gap;
    long long price1;
    long long price2;
    long long pricesumA;
    long long pricesumB;
    int leftsize;
    int rightsize;
    int type;
	struct Node *parent;
    struct Node *left;
    struct Node *right;
}TreeNode;

int N, M;
long long Dumpling[1001];     // Dumpling[type] = gap
TreeNode *root;
TreeNode *NIL;

void InorderTraverse(TreeNode *cur, int deepth);
void Left_Rotation(TreeNode *x);
void Right_Rotation(TreeNode *y);
void Insert(long long price1, long long price2, int type);
void InsertFix(TreeNode *cur);
void Delete(int type);
void DeleteFix(TreeNode *cur);


int main(){
    // initialize
    NIL = malloc(sizeof(TreeNode));
    NIL->color = 1;
    NIL->pricesumA = 0;
    NIL->pricesumB = 0;
    root = NIL;
    root->parent = NIL;

    scanf("%d%d", &N, &M);

    for(int i = 0; i < N; i++){
        long long price1, price2;
        scanf("%lld%lld", &price1, &price2);

        if(price1 > price2){
            long long temp = price1;
            price1 = price2;
            price2 = temp;
        }

        Insert(price1, price2, i+1);
    }

    //#ifdef DEBUG
    long long cost = (root->pricesumA < root->pricesumB) ? root->pricesumA : root->pricesumB;
    printf("%lld\n", cost);

    for(int i = 0; i < (M-1); i++){
        int t, c, d, e, f;
        scanf("%d%d%d%d%d", &t, &c, &d, &e, &f);

        Delete(t);

        long long newprice1 = (c * cost + d) % divisor;
        long long newprice2 = (e * cost + f) % divisor;
        if(newprice1 > newprice2){
            long long temp = newprice1;
            newprice1 = newprice2;
            newprice2 = temp;
        }

        Insert(newprice1, newprice2, t);

        cost = (root->pricesumA < root->pricesumB) ? root->pricesumA : root->pricesumB;
        printf("%lld\n", cost);
    }
    //#endif

    return 0;
}


void InorderTraverse(TreeNode *cur, int deepth){
    
    if(cur == NIL){
        return;
    }
    
    InorderTraverse(cur->left, deepth+1);
    
    printf("[%d] type:%d gap:%d price1:%d, price2:%d\n    pricesumA:%d pricesumB:%d leftsize:%d rightsize:%d parent:%d\n",
    deepth, cur->type, cur->gap, cur->price1, cur->price2, cur->pricesumA, cur->pricesumB, cur->leftsize, cur->rightsize, cur->parent->type);
    
    InorderTraverse(cur->right, deepth+1);
}

void Left_Rotation(TreeNode *x){

    TreeNode *y = x->right;

    x->pricesumA -= y->pricesumB;
    x->pricesumB -= y->pricesumA;

    x->right = y->left;
    if(x->right != NIL){
        y->pricesumA -= x->right->pricesumB;
        y->pricesumB -= x->right->pricesumA;
        x->pricesumA += x->right->pricesumB;
        x->pricesumB += x->right->pricesumA;
        x->right->parent = x;
    }

    y->parent = x->parent;
    if(x->parent == NIL){
        root = y;
    }
    else if(x == x->parent->left){
        x->parent->left = y;
    }
    else{
        x->parent->right = y;
    }

    y->left = x;
    x->parent = y;

    y->pricesumA += x->pricesumB;
    y->pricesumB += x->pricesumA;
    
    y->leftsize += (x->leftsize + 1);
    x->rightsize -= (y->rightsize + 1); 
}

void Right_Rotation(TreeNode *y){

    TreeNode *x = y->left;

    y->pricesumA -= x->pricesumB;
    y->pricesumB -= x->pricesumA;

    y->left = x->right;
    if(y->left != NIL){
        x->pricesumA -= y->left->pricesumB;
        x->pricesumB -= y->left->pricesumA;
        y->pricesumA += y->left->pricesumB;
        y->pricesumB += y->left->pricesumA;
        y->left->parent = y;
    }

    x->parent = y->parent;
    if(y->parent == NIL){
        root = x;
    }
    else if(y == y->parent->left){
        y->parent->left = x;
    }
    else{
        y->parent->right = x;
    }

    x->right = y;
    y->parent = x;

    x->pricesumA += y->pricesumB;
    x->pricesumB += y->pricesumA;

    y->leftsize -= (x->leftsize + 1);
    x->rightsize += (y->rightsize + 1); 
}

void Insert(long long price1, long long price2, int type){
    
    TreeNode *cur = root;
    TreeNode *cur_p = NIL;

    // init new node
    TreeNode *newnode = malloc(sizeof(TreeNode));
    newnode->color = 0;
    newnode->gap = price2 - price1;
    newnode->price1 = price1;
    newnode->price2 = price2;
    newnode->pricesumA = price1;
    newnode->pricesumB = price2;
    newnode->leftsize = 0;
    newnode->rightsize = 0;
    newnode->type = type;
    newnode->left = NIL;
    newnode->right = NIL;

    Dumpling[type]= newnode->gap;

    // find the place to insert (gap大的在左邊)
    while(cur != NIL){
        cur_p = cur;
        if(newnode->gap > cur->gap){
            cur->leftsize ++;
            cur = cur->left;
        }
        else{
            cur->rightsize ++;
            cur = cur->right;
        }
    }
    newnode->parent = cur_p;

    // attach newnode to tree
    if(cur_p == NIL){
        root = newnode;
    }
    else if(newnode->gap > cur_p->gap){
        cur_p->left = newnode;
    }
    else{
        cur_p->right = newnode;
    }

    // fixup pricesum
    cur = newnode;
    while(cur->parent != NIL){
        cur = cur->parent;

        cur->pricesumA = cur->price1;
        cur->pricesumB = cur->price2;
        if(cur->left->rightsize % 2 == 0){
            cur->pricesumA += cur->left->pricesumB;
            cur->pricesumB += cur->left->pricesumA;    
        }
        else{
            cur->pricesumA += cur->left->pricesumA;
            cur->pricesumB += cur->left->pricesumB;  
        }
        if(cur->right->leftsize % 2 == 0){
            cur->pricesumA += cur->right->pricesumB;
            cur->pricesumB += cur->right->pricesumA;    
        }
        else{
            cur->pricesumA += cur->right->pricesumA;
            cur->pricesumB += cur->right->pricesumB;  
        }
    }
            
    InsertFix(newnode);  // 對紅紅相連做修正
}

void InsertFix(TreeNode *cur){
    
    while(cur->parent->color == 0){     // when parent is red, continue

        if(cur->parent == cur->parent->parent->left){
            TreeNode *uncle = cur->parent->parent->right;

            if(uncle->color == 0){      // case1 
                cur->parent->color = 1;
                uncle->color = 1;
                cur->parent->parent->color = 0;
                cur = cur->parent->parent;
            }
            else{
                if(cur == cur->parent->right){       // case2
                    cur = cur->parent;
                    Left_Rotation(cur);
                }
                // case3
                cur->parent->color = 1;
                cur->parent->parent->color = 0;
                Right_Rotation(cur->parent->parent);
            }
        }

        else{
            TreeNode *uncle = cur->parent->parent->left;

            if(uncle->color == 0){      // case1 
                cur->parent->color = 1;
                uncle->color = 1;
                cur->parent->parent->color = 0;
                cur = cur->parent->parent;
            }
            else{
                if(cur == cur->parent->left){       // case2
                    cur = cur->parent;
                    Right_Rotation(cur);
                }
                // case3
                cur->parent->color = 1;
                cur->parent->parent->color = 0;
                Left_Rotation(cur->parent->parent);
            }
        }

    }

    root->color = 1;    // make root black
}

void Delete(int type){
    
    // find the node to delete
    TreeNode *cur = root;
    int key = Dumpling[type];

    while(cur != NIL){
        if(cur->type == type){
            break;
        }
        else if(cur->gap >= key){
            cur->rightsize --;
            cur = cur->right;
        }
        else{
            cur->leftsize --;
            cur = cur->left;
        }
    }

    if(cur == NIL){
        while(cur != NIL){
            if(cur->type == type){
                break;
            }
            else if(cur->gap >= key){
                cur->rightsize --;
                cur = cur->right;
            }
            else{
                cur->leftsize --;
                cur = cur->left;
            }
        }
    }

    TreeNode *keynode = cur;    // the node we want to delete
    TreeNode *del = 0;          // the node we actually delete
    TreeNode *del_c = 0;        // child of deleted node

    if(keynode->left == NIL || keynode->right == NIL){
        del = keynode;
    }
    else{
        // case3 : 找替身 successor
        cur = keynode->right;
        while(cur->left != NIL){
            cur->leftsize --;
            cur = cur->left;
        }
        del = cur;
    }

    if(del->left != NIL){
        del_c = del->left;
    }
    else{
        del_c = del->right;
    }

    del_c->parent = del->parent;

    if(del->parent == NIL){
        root = del_c;
    }
    else if(del == del->parent->left){
        del->parent->left = del_c;
    }
    else{
        del->parent->right = del_c;
    }

    // case 3 fixup
    if(del != keynode){
        keynode->gap = del->gap;
        keynode->type = del->type;
        keynode->price1 = del->price1;
        keynode->price2 = del->price2;
        keynode->leftsize = del->leftsize;
        keynode->rightsize = del->rightsize;
    }

    // fixup pricesum
    cur = del;
    while(cur->parent != NIL){
        cur = cur->parent;

        cur->pricesumA = cur->price1;
        cur->pricesumB = cur->price2;
        if(cur->left->rightsize % 2 == 0){
            cur->pricesumA += cur->left->pricesumB;
            cur->pricesumB += cur->left->pricesumA;    
        }
        else{
            cur->pricesumA += cur->left->pricesumA;
            cur->pricesumB += cur->left->pricesumB;  
        }
        if(cur->right->leftsize % 2 == 0){
            cur->pricesumA += cur->right->pricesumB;
            cur->pricesumB += cur->right->pricesumA;    
        }
        else{
            cur->pricesumA += cur->right->pricesumA;
            cur->pricesumB += cur->right->pricesumB;  
        }
    }

    int color = del->color;
    free(del);

    if(color == 1){
        DeleteFix(del_c);
    }
}

void DeleteFix(TreeNode *cur){
    // if cur is root or is red, turn it into black directly

    while(cur != root && cur->color == 1){

        if(cur == cur->parent->left){
            TreeNode *sibling = cur->parent->right;

            if(sibling->color == 0){        // case 1
                sibling->color = 1;
                cur->parent->color = 0;
                Left_Rotation(cur->parent);
                sibling = cur->parent->right; 
            }

            if(sibling->left->color == 1 && sibling->right->color == 1){        // case 2
                sibling->color = 0;
                cur = cur->parent;
            }
            else{
                // case 3
                if(sibling->right->color == 1){
                    sibling->left->color = 1;
                    sibling->color = 0;
                    Right_Rotation(sibling);
                    sibling = cur->parent->right;
                }
                // case 4
                sibling->color = cur->parent->color;
                cur->parent->color = 1;
                sibling->right->color = 1;
                Left_Rotation(cur->parent);
                cur = root;
            }
        }

        else{
            TreeNode *sibling = cur->parent->left;

            if(sibling->color == 0){        // case 1
                sibling->color = 1;
                cur->parent->color = 0;
                Right_Rotation(cur->parent);
                sibling = cur->parent->left; 
            }

            if(sibling->left->color == 1 && sibling->right->color == 1){        // case 2
                sibling->color = 0;
                cur = cur->parent;
            }
            else{
                // case 3
                if(sibling->left->color == 1){
                    sibling->right->color = 1;
                    sibling->color = 0;
                    Left_Rotation(sibling);
                    sibling = cur->parent->left;
                }
                // case 4
                sibling->color = cur->parent->color;
                cur->parent->color = 1;
                sibling->left->color = 1;
                Right_Rotation(cur->parent);
                cur = root;
            }
        }
    }

    cur->color = 1;
}
