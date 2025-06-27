#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#ifdef DEBUG
void printarr(int m[], int len){
	for( int i=0 ; i<len ; i++ )
		printf("%d", m[i]);

	printf("\n");
	return;
}
#endif


/* return 0 if m >= n , otherwise */
int compare(int m[], int n[], int len){

	for(int i=0 ; i<len ; i++){
		if(m[i] > n[i]){
			return 0;
		}
		else if(m[i] < n[i]){
			return 1;
		}
	}

	return 0;
}

/* swap to make a > b */
void swap(int m[], int n[], int len){
	int temp;

	for(int i=0 ; i<len ; i++){
		temp = m[i];
		m[i] = n[i];
		n[i] = temp;
	}

	return;
}

/* add 0 in front of array to align the longer one */
int align(int arr[], int nlen, int mlen){

	if(nlen == mlen) return mlen;

	for(int i=0 ; i<nlen ; i++)
		arr[mlen-1-i] = arr[nlen-1-i];

	for(int i=0 ; i<mlen-nlen ; i++)
		arr[i] = 0;

	return mlen;
}

/* remove zero in front of m */
int removeZERO(int m[], int len){
	int c = 0;
	while( m[c] == 0 ){
		c++;
	}

	len -= c;
	for( int i=0 ; i<len ; i++ )
		m[i] = m[i+c];

	return len;
}

/* m - n */
int substract(int m[], int n[], int len){
	int temp[512];
	int borrow = 0;

	for(int i=len-1 ; i>=0 ; i--){
		if( (m[i]-borrow) >= n[i] ){
			temp[i] = m[i] - borrow - n[i];
			borrow = 0;
		}
		else{
			temp[i] = m[i] + 10 - borrow - n[i];
			borrow = 1;
		}
		// printf("m[i]:%d n[i]:%d temp[i]:%d\n", m[i], n[i], temp[i]);
	}

	for(int i=0 ; i<len ; i++)
		m[i] = temp[i];

	return 0; 
}

/* m / div */
int division(int m[], int div, int len){
	int temp[512];
	int borrow = 0;

	for(int i=0 ; i<len ; i++){
		if( (m[i] + borrow*10) >= div ){
			temp[i] = (m[i] + borrow*10) / div;
			borrow = (m[i] + borrow*10) % div;
		}
		else{
			borrow = m[i];
			temp[i] = 0;
		}
	}

	for(int i=0 ; i<len ; i++)
		m[i] = temp[i];

	return 0;
}

/* m * exp */
int multiplication(int m[], int exp, int len){
	int explen = 0, count = exp;
	while( count != 0 ){
		count /= 10;
		explen ++;
	}

	int ans[512];
	len = align(m, len, len + explen);
	int temp = 0, carry = 0;

	for(int i=len-1 ; i>=0 ; i--){
		temp = (m[i] * exp) + carry;
		ans[i] = temp % 10;
		carry = temp / 10;
	}
	
	len = removeZERO(ans, len);
	for(int i=0 ; i<len ; i++){
		printf("%d", ans[i]);
	}

	return 0;
}


int main(){
	int i;

	/* get input */
	int m[512], n[512];
	int mlen, nlen;
	char in;

	for( i=0 ; i<512 ; i++){
		scanf("%c", &in);
		if(isspace(in) != 0) break;
		m[i] = in - '0'; 
	}
	mlen = i;

	for(i=0 ; i<512 ; i++){
		scanf("%c", &in);
		if(in == '\n') break;
		n[i] = in - '0';
	}
	nlen = i;

	/* align m and m */
	if(mlen > nlen){
		nlen = align(n, nlen, mlen);
	}
	else if(mlen < nlen){
		mlen = align(m, mlen, nlen);
	}

	/* if n > m , swap */
	if(compare(m, n, mlen) == 1) swap(m, n, mlen);


	// Binary Algorithm
	int exp = 1;
	while(mlen > 0 && nlen > 0){

		if(mlen > nlen){
			nlen = align(n, nlen, mlen);
		}
		else if(mlen < nlen){
			mlen = align(m, mlen, nlen);
		}

		if( m[mlen-1] % 2 == 0 && n[nlen-1] % 2 == 0 ){
			exp *= 2;
			division(m, 2, mlen);
			division(n, 2, nlen);
		}
		else if( m[mlen-1] % 2 == 0 ){
			division(m, 2, mlen);
		}
		else if( n[nlen-1] % 2 == 0 ){
			division(n, 2, nlen);
		}

		if(compare(m, n, mlen) == 1) swap(m, n, mlen);

		substract(m, n, mlen);

		mlen = removeZERO(m, mlen);
		nlen = removeZERO(n, nlen);
	}
	multiplication(n, exp, nlen);

	return 0;
}