b11902038 資工三 鄭博允

---

# Question 1

```
	addi   x22, x0, -1
Loop:
	addi   x22, x22, 1
	slli   x10, x22, 2
	add    x10, x10, x25
	lw     x9, 0(x10)
	beq    x9, x24, Loop
	// exit code..
```

# Question 2

## `0x00A484b3`

1. Binary: `0000 0000 1010 0100 1000 0100 1011 0011`
2. Using R-format: `0000000 01010 01000 100 01001 0110011`
	- Opcode = `0110011`, Funct7 = `0000000`, Funct3 = `100`
	  → instruction `srl`
	- rd = `01001` → register `x9`
	- rs1 = `01000` → register `x8`
	- rs2 = `01010` → register `x10`
3. The instruction is `srl x9, x8, x10`, which shift the value in `x8` right by the number of positions (add zero to the left) specified in `x10`, and store the result in `x9`.

## `0x40A48533`

1. Binary: `0100 0000 1010 0100 1000 0101 0011 0011`
2. Using R-format: `0100000 01010 01000 100 01010 0110011`
    - Opcode = `0110011`, Funct7 = `0100000`, Funct3 = `100` → instruction `sra`
    - rd = `01010` → register `x10`
    - rs1 = `01000` → register `x8`
    - rs2 = `01010` → register `x10`
3. The instruction is `sra x10, x8, x10`, which shifts the value in `x8` right arithmetically by the number of positions (add sign number to the left) specified in `x10`, and stores the result in `x10`.

## `0x40A484B3`

1. Binary: `0100 0000 1010 0100 1000 0100 1011 0011`
2. Using R-format: `0100000 01010 01000 100 01001 0110011`
    - Opcode = `0110011`, Funct7 = `0100000`, Funct3 = `100` → instruction `sra`
    - rd = `01001` → register `x9`
    - rs1 = `01000` → register `x8`
    - rs2 = `01010` → register `x10`
3. The instruction is `sra x9, x8, x10`, which does the same thing as the previous group, but stores the result in `x9`.

> Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf

# Question 3

add these line at the beginning of `Main`:
```
add x5, x10, x0
add x10, x12, x0
add x12, x5, x0
```
which swap the register of x and z.
