B11902038 鄭博允

---

# 1.

In my system (WSL2 ubuntu), the access permission set for the *passwd* program is `rwsr-xr-x`.
The third bit `s` is set-user-ID bit, and thus *passwd* is a set-user-ID program. 
When execute *passwd*, normal user's <mark style="background: #FFB8EBA6;">effective user ID</mark> will be set to <mark style="background: #FFB8EBA6;">real user ID</mark> of passwd's owner (which is root).
Therefore, normal user is allowed to execute *passwd*.

<br>

#2.

> Reference :
> https://stackoverflow.com/questions/29484910/reason-of-a-directory-size-being-zero
> https://unix.stackexchange.com/questions/55/what-does-size-of-a-directory-mean-in-output-of-ls-l-command

First, file size 0 is allowed for a regular file, but accroding to the definition of the textbook, a directory file is not a regular file, and file size of a directory file is **usually** a multiple of 16 or 512.

Moreover, according to my knowledge, a directory in linux **cannot** be completely empty, because a directory has at least two entries : `.` and `..`. Therefore, I don't think we should ever see a directory file with `st_size` equals to 0.

<br>

# 3.
1. False
2. False
3. False
4. False

<br>

# 4.

> Reference :
> https://man7.org/linux/man-pages/man1/rm.1.html
> https://man7.org/linux/man-pages/man2/unlink.2.html
> https://stackoverflow.com/questions/21517600/how-does-rm-work-what-does-rm-do

When using the `rm` command to delete a file, it actually makes use of the `unlink()` system call to remove the file from the system.

According to the manual page, `unlink()` doesn't actually remove the _file's data blocks_ but rather removes the _directory entry_ (link) of the file. If the _directory entry_ (link) that we `unlink()` is the last link that points to the file, and if currently no process has the file open, then the _file's data blocks_ can be reclaimed by the system, effectively removing the file.

The reason for printing the message is described in the manual page. When `rm` encounters a read-only file (if the `-f` option is not given), it first prints a message to "inform" the user that the file is write-protected and asks for the user's consent. Then, it will call `unlink()` to remove that file if the user agrees; otherwise, it won't take any action.

The reason we can remove a read-only file is that the permissions required to remove a file are the `w+x` permissions of the directory containing the file, regardless of the `w` permission of the file.
