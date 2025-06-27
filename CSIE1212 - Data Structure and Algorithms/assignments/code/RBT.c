#include <stdio.h>
#include <stdlib.h>
#define divisor 1000000007

typedef struct Node{
	int color;  // 0 = red, 1 = black
	int gap;
    int price1;
    int price2;
    int type;
	struct Node *parent;
    struct Node *left;
    struct Node *right;
}TreeNode;

int N, M;
TreeNode *root;
TreeNode *NIL;

TreeNode* Successor(TreeNode *cur);
void InorderTraverse(TreeNode *cur, int deepth);
void Left_Rotation(TreeNode *x);
void Right_Rotation(TreeNode *y);
void Insert(int price1, int price2, int type);
void InsertFix(TreeNode *cur);
void Delete(TreeNode *cur);
void DeleteFix(TreeNode *cur);


int main(){
    // initialize
    NIL = malloc(sizeof(TreeNode));
    NIL->color = 1;
    root = NIL;
    root->parent = NIL;

    return 0;
}
    
    

TreeNode* Successor(TreeNode *cur){

    if(cur->right != NIL){
        cur = cur->right;

        while(cur->left != NULL){
            cur = cur->left;
        }

        return cur;
    }

    TreeNode *newnode = cur->parent;

    while(newnode != NIL && cur == newnode->right){
        cur = newnode;
        newnode = newnode->parent;
    }
    
    return newnode;
}

void InorderTraverse(TreeNode *cur, int deepth){
    
    if(cur == NIL){
        return;
    }
    
    InorderTraverse(cur->left, deepth+1);
    
    printf("[%d] type:%d gap:%d color:%s\n", deepth, cur->type, cur->gap, cur->color == 0 ? "Red":"Black");
    
    InorderTraverse(cur->right, deepth+1);
}

void Left_Rotation(TreeNode *x){

    TreeNode *y = x->right;

    x->right = y->left;
    if(x->right != NIL){
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
}

void Right_Rotation(TreeNode *y){

    TreeNode *x = y->left;

    y->left = x->right;
    if(y->left != NIL){
        y->left->parent = x;
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
}

void Insert(int price1, int price2, int type){
    
    TreeNode *cur = root;
    TreeNode *cur_p = NIL;

    // init new node
    TreeNode *newnode = malloc(sizeof(TreeNode));
    newnode->color = 0;
    newnode->gap = price1 - price2;
    newnode->price1 = price1;
    newnode->price2 = price2;
    newnode->type = type;
    newnode->left = NIL;
    newnode->right = NIL;

    // find the place to insert (gap大的在左邊)
    while(cur != NIL){
        cur_p = cur;
        if(newnode->gap > cur->gap){
            cur = cur->left;
        }
        else{
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

void Delete(TreeNode *cur){
    
    TreeNode *del = 0;          // the node we actually deleted
    TreeNode *del_c = 0;        // child of deleted node

    if(cur->left == NIL || cur->right == NIL){
        del = cur;
    }
    else{
        del = Successor(cur);
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

    if(del != cur){
        cur->gap = del->gap;
        cur->price1 = del->price1;
        cur->price2 = del->price2;
        cur->type = del->type;
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

