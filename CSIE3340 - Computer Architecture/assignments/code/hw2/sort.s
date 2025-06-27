.globl sort

# partition(addr, lo, hi)
# t1: left index i
# t2: right index j
# t3: value of pivot
partition:
	# Save registers
	addi	sp, sp, -8
	sd	ra, 0(sp)

	# pivot = A[lo]
	slli	t0, a1, 3
	add	t0, a0, t0
	ld	t3, 0(t0)

	# i = lo - 1
	addi	t1, a1, -1

	# j = hi + 1
	addi	t2, a2, 1

partition_loop:
	left_loop:
		addi	t1, t1, 1
		slli	t0, t1, 3
		add	t0, a0, t0
		ld	t4, 0(t0)	# t4 = A[i]
		blt	t4, t3, left_loop

	right_loop:
		addi	t2, t2, -1
		slli	t0, t2, 3
		add	t0, a0, t0
		ld	t5, 0(t0)	# t5 = A[j]
		bgt	t5, t3, right_loop

	## if i >= j
	bge	t1, t2, partition_exit

	# Swap A[i] with A[j]
	slli	t0, t2, 3
	add	t0, a0, t0
	sd	t4, 0(t0)	# Store t4 to A[j]

	slli	t0, t1, 3
	add	t0, a0, t0
	sd	t5, 0(t0)	# Store t5 to A[i]

	j	partition_loop

partition_exit:
	# Return j in a0
	mv	a0, t2

	# Restore registers
	ld	ra, 0(sp)
	addi	sp, sp, 8

	ret

# quicksort(addr, lo, hi)
quicksort:
	# Save registers
	# s0 = base addr of A
	# s1 = lo
	# s2 = hi
	addi	sp, sp, -32
	sd	ra, 24(sp)
	sd	s0, 16(sp)
	sd	s1, 8(sp)
	sd	s2, 0(sp)

	mv	s0, a0
	mv	s1, a1
	mv	s2, a2

	## if lo < 0 or hi < 0 or lo >= hi then goto exit
	blt	a1, zero, quicksort_exit
	blt	a2, zero, quicksort_exit
	bge	a1, a2, quicksort_exit

	# Call partition(addr, lo, hi)
	jal	ra, partition
	mv	t0, a0		# save result pivot to t0

	# Call quicksort(addr, lo, p)
	mv	a0, s0
	mv	a1, s1
	mv	a2, t0
	jal	ra, quicksort

	# Call quicksort(addr, p+1, hi)
	mv	a0, s0
	addi	a1, t0, 1
	mv	a2, s2
	jal	ra, quicksort

quicksort_exit:
	# Restore registers
	ld	ra, 24(sp)
	ld	s0, 16(sp)
	ld	s1, 8(sp)
	ld	s2, 0(sp)
	addi	sp, sp, 32

	ret

# sort(addr, count)
sort:
	# Save registers
	addi	sp, sp, -8
	sd 	ra, 0(sp)

	# Call quicksort(addr, 0, count-1)
	addi	a2, a1, -1
	mv	a1, zero
	jal	ra, quicksort

	# Restore registers
	ld	ra, 0(sp)
	addi	sp, sp, 8

	ret