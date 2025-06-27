B11902038 資工一 鄭博允

---

# 1.
> Reference :
課程簡報
[https://www.runoob.com/linux/linux-comm-tee.html](https://www.runoob.com/linux/linux-comm-tee.html)
[https://www.onejar99.com/linux-command-tee/](https://www.onejar99.com/linux-command-tee/)
[https://man7.org/linux/man-pages/man1/tee.1.html](https://man7.org/linux/man-pages/man1/tee.1.html)

<br>

```c
tee [OPTION].. [FILE]..

// assume stdin is a file descriptor called *rf*
// assume FILE is a pathname of the file to be written
// assume we have each FILE stored in a array called FILES[n]

int b
int wf[n]
char buf[BUFFSIZE] 

// set offset of *rf* to 0
lseek(rf, 0, SEEK_SET)

// open FILE if exist, create if not
for (int i=0 ; i<n ; i++) {
    wf[i] = open(FILES[i], O_RDWR | O_CREAT)
// if -a is given, set offset to the end of FILE; set to 0 if not
    if (OPTION == "-a" || OPTION == "--append") {
        lseek(wf[i], 0, SEEK_END)
    } else {
        lseek(wf[i], 0, SEEK_SET)
    }
}

// read entire *rf* and write into *wf*
while ((b = read(rf, buf, BUFFSIZE)) > 0) {
    for (int i=0 ; i<n ; i++) {
        write(wf[i], buf, BUFFSIZE)
    }
    print(buf)
}

for (int i=0 ; i<n ; i++) {
    close(wf[i])
}
```
<br>

Explanation :

**line 16**
I use `open()` to open `FILE` in `FILES[]` and use `wf[]` to store its file descriptor
also, I use `O_CREAT` to create new file if `FILE` is not exist
<br>

**line 18-22**
first, I check the option `-a` is given or not
then I use `lseek()` to set offset of `wf[]` :
- if `-a` is given, we need to write data from the EOF
    therefore I use flag `SEEK_END` to set the offset be the size of the file
- otherwise, we overwrite the file
    therefore I use flag `SEEK_SET` to set the offset to 0
<br>

**line 26**
I read data from `rf` to `buf` by `read()`
`read()` will return the number of bytes been read, I assigned this value to variable `b`
- if `b` is greater than 0, which means we haven’t read the entire `rf` yet, then we keep reading
- if `b` is less than 0, we stop reading because there’s nothing left to read
<br>

**line 27-29**
I use `for` loop and `write()` to write data from `buf` to each file in `wf[]`
<br>

**line 30**  
directly print out the content in `buf`
<br>

**line 34**
close each file in `wf[]`
<br>
<div style="page-break-after: always;"></div>


# 2.
## 2.1

> Reference :
https://www.runoob.com/linux/linux-shell-io-redirections.html
https://blog.csdn.net/u011630575/article/details/52151995
https://www.hy-star.com.tw/tech/linux/pipe/pipe.html#stdout

<br>

Assume file descriptors and open file table before running the command are like this :
![[School/Course Homeworks/System Programming/png/sp01-2-1.png|350]]
the `stdout` and `stderr` in open file table will be print on terminal

After executing a command, stdout will be deliver to the file that fd1 is pointed (which originally is `stdout`), and stderr will be deliver to the file that fd2 is pointed (which originally is `stderr`).
<br>

1. `./a.out > outfile 2>&1`
	this command is executed in the following order : <br>
	1. run `./a.out` <br>
	2. `> outfile` is equivalent to `1> outfile`, which means to redirect fd1 to `outfile` (red arrow) <br>
	3. `2>&1` will redirect fd2 to the file that fd1 is pointed, which is `outfile` (blue arrow) <br>
	4. it comes out that both **stdout** and **stderr** of `./a.out` will be deliver to `outfile` and nothing will be print on terminal <br>
![[School/Course Homeworks/System Programming/png/sp01-2-2.png|350]]
<br>
<div style="page-break-after: always;"></div>

1. `./a.out 2>&1 > outfile`
	this command is executed in the following order : <br>
	1. run `./a.out` <br>
	2. `2>&1` will redirect fd2 to the file that fd1 is pointed, which is `stdout` (red arrow) <br>
	3. `> outfile` will redirect fd1 to `outfile` (blue arrow) <br>
	4. it comes out that **stdout** of `./a.out` will be deliver to `outfile` ; **stderr** of `./a.out` will be deliver to `stdout` and print on terminal <br>
![[School/Course Homeworks/System Programming/png/sp01-2-3.png|350]]
<br>

## 2.2

we can use `command > file 2>&1`
if we want to put both **stdout** and **stderr** into to same file
<br>
<div style="page-break-after: always;"></div>


# 3.

## 3.1
No.
Open a file with flag `O_APPEND`, that is,  it will set the offset to EOF before each `write()`, which can be seen as a atomic operation. Although we use `lseek()` before `write()`, it will still set the offset to the EOF again.
<br>

## 3.2
No.
Generally, `cd ..` will move you to the parent directory of your current location;
and `cd .` will move you to your current location.
However, if you were currently at root directory, which doesn't have parent directory, then you might stay at current location. In this case, both `cd ..` and `cd .` will lead you to your current location.
<br>

## 3.3
Yes.
Since `ls` is an executable file in linux, you need a new process to execute it.
<br>

## 3.4
No.
The description is wrong.
> Consider the following code example we show in the class.  
Suppose func() is run by two processes in the system simultaneously.  
The reason why we want to make the seek() and write() here atomic is
<mark style="background: #FF5582A6;">because one process could execute "seek()" after another process finishes executing "write()"</mark>.

it should be
>becasue one process could execute "seek()" first, and execute "write()" after another process finishes executing "seek()" and "write()"

<br>

## 3.5
Yes.

```text
dup(fd1)
=
fnctl(fd1, F_DUPFD, 0) 
```

```text
dup2(fd1, fd2)
=
close(fd1)
fnctl(fd1, F_DUPFD, fd2)
```