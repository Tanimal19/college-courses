# Problem 1

The total miss latency can be calculate as
$$\text{total miss latency} = \text{reference per instruction} \times \text{miss rate} \times \text{miss latency}$$

## (1)

| block size | total miss latency                   |
| ---------- | ------------------------------------ |
| 8          | 1.35 x 0.04 x (20 x 8) = **8.64**    |
| 16         | 1.35 x 0.03 x (20 x 16) = **12.96**  |
| 32         | 1.35 x 0.02 x (20 x 32) = **17.28**  |
| 64         | 1.35 x 0.015 x (20 x 64) = **25.92** |
| 128        | 1.35 x 0.01 x (20 x 128) = **34.56** |

Therefore the optimal block size is 8.

## (2)

| block size | total miss latency                   |
| ---------- | ------------------------------------ |
| 8          | 1.35 x 0.04 x (24 + 8) = **1.728**   |
| 16         | 1.35 x 0.03 x (24 + 16) = **1.62**   |
| 32         | 1.35 x 0.02 x (24 + 32) = **1.512**  |
| 64         | 1.35 x 0.015 x (24 + 64) = **1.782** |
| 128        | 1.35 x 0.01 x (24 + 128) = **2.052** |

Therefore the optimal block size is 32.

## (3)
If the miss latency is constant, based on the formula, the smallest **miss rate** will obtain smallest total miss latency. Therefore the optimal block size is 128.

<div style="page-break-after:always;"></div>

# Problem 2
## (1)
A 64-bit integers is 8 bytes, therefore we can store 16/8 = 2 integers.

## (2)
Temporal locality means a same memory location is accessed multiple time within a short time.

Therefore the variables exhibit temporal locality are `B[I][0]` and `I` and `J`:
- `B[I][0]` and `I` remain the same for an entire inner loop
- `J` is accessed two times inside each second loop

## (3)
Spatial locality means some memory locations which close to each other are accessed in a short time.

Therefore the only variable exhibit spatial locality is `A[I][J]`, since `J` iterates sequentially and each elements with same `J` (that is, same row) are stored contiguously. While `A[J][I]` doesn't exhibit spatial locality, since the elements being accessed are not in the same row and are not stored contiguously.

## (4)
There's total
- `A`: 8 x 8000 = 64000
- `B`: 8 x 1 = 8
64-bit matrix elements.

At (1) we know that each 16-byte cache block can store 2 64-bit integers, therefore we need (64000 + 8) / 2 = 32004 blocks.

<div style="page-break-after:always;"></div>

# Problem 3
For a two-way set associative cache with four one-word blocks, the address map to index is like:
- address 0, 2, 4, 6 -> set 0
- address 1, 3, 5, 7 -> set 1

## (1)
LRU will replace the least recently used block (in the same set) when miss occurs.

| Access | Set   | Hit/Miss | Cache State (Set 0) | Cache State (Set 1) |
| ------ | ----- | -------- | ------------------- | ------------------- |
| 0      | 0     | Miss     | 0                   |                     |
| 1      | 1     | Miss     | 0                   | 1                   |
| 2      | 0     | Miss     | 0, 2                | 1                   |
| 3      | 1     | Miss     | 0, 2                | 1, 3                |
| 4      | 0     | Miss     | 2, 4                | 1, 3                |
| **2**  | **0** | **Hit**  | **2, 4**            | **1, 3**            |
| **3**  | **1** | **Hit**  | **2, 4**            | **1, 3**            |
| **4**  | **0** | **Hit**  | **2, 4**            | **1, 3**            |
| 5      | 1     | Miss     | 2, 4                | 3, 5                |
| 6      | 0     | Miss     | 4, 6                | 3, 5                |
| 7      | 1     | Miss     | 4, 6                | 5, 7                |
| 0      | 0     | Miss     | 0, 6                | 5, 7                |
| 1      | 1     | Miss     | 0, 6                | 1, 7                |
| 2      | 0     | Miss     | 0, 2                | 1, 7                |
| 3      | 1     | Miss     | 0, 2                | 1, 3                |
| 4      | 0     | Miss     | 2, 4                | 1, 3                |
| 5      | 1     | Miss     | 2, 4                | 3, 5                |
| 6      | 0     | Miss     | 4, 6                | 3, 5                |
| 7      | 1     | Miss     | 4, 6                | 5, 7                |
| 0      | 0     | Miss     | 0, 6                | 5, 7                |

<div style="page-break-after:always;"></div>

## (2)
MRU will replace the most recently used block when miss occurs.

| Access | Set   | Hit/Miss | Cache State (Set 0) | Cache State (Set 1) |
| ------ | ----- | -------- | ------------------- | ------------------- |
| 0      | 0     | Miss     | 0                   |                     |
| 1      | 1     | Miss     | 0                   | 1                   |
| 2      | 0     | Miss     | 0, 2                | 1                   |
| 3      | 1     | Miss     | 0, 2                | 1, 3                |
| 4      | 0     | Miss     | 0, 4                | 1, 3                |
| 2      | 0     | Miss     | 0, 2                | 1, 3                |
| **3**  | **1** | **Hit**  | **0, 2**            | **1, 3**            |
| 4      | 0     | Miss     | 0, 4                | 1, 3                |
| 5      | 1     | Miss     | 0, 4                | 1, 5                |
| 6      | 0     | Miss     | 0. 6                | 1, 5                |
| 7      | 1     | Miss     | 0, 6                | 1, 7                |
| **0**  | **0** | **Hit**  | **0, 6**            | **1, 7**            |
| **1**  | **1** | **Hit**  | **0, 6**            | **1, 7**            |
| 2      | 0     | Miss     | 2, 6                | 1, 7                |
| 3      | 1     | Miss     | 2, 6                | 3, 7                |
| 4      | 0     | Miss     | 4, 6                | 3, 7                |
| 5      | 1     | Miss     | 4, 6                | 5, 7                |
| **6**  | **0** | **Hit**  | **4, 6**            | **5, 7**            |
| **7**  | **1** | **Hit**  | **4, 6**            | **5, 7**            |
| 0      | 0     | Miss     | 0, 4                | 5, 7                |

<div style="page-break-after:always;"></div>

## (3)

Replace the block that will be used less frequently in the future, if same, then replace the one with longest time between next access.

| Access | Set   | Hit/Miss | Cache State (Set 0) | Cache State (Set 1) |
| ------ | ----- | -------- | ------------------- | ------------------- |
| 0      | 0     | Miss     | 0                   |                     |
| 1      | 1     | Miss     | 0                   | 1                   |
| 2      | 0     | Miss     | 0, 2                | 1                   |
| 3      | 1     | Miss     | 0, 2                | 1, 3                |
| 4      | 0     | Miss     | 2, 4                | 1, 3                |
| **2**  | **0** | **Hit**  | **2, 4**            | **1, 3**            |
| **3**  | **1** | **Hit**  | **2, 4**            | **1, 3**            |
| **4**  | **0** | **Hit**  | **2, 4**            | **1, 3**            |
| 5      | 1     | Miss     | 2, 4                | 1, 5                |
| 6      | 0     | Miss     | 2, 6                | 1, 5                |
| 7      | 1     | Miss     | 2, 6                | 1, 7                |
| 0      | 0     | Miss     | 0, 2                | 1, 7                |
| **1**  | **1** | **Hit**  | **0, 2**            | **1, 7**            |
| **2**  | **0** | **Hit**  | **0, 2**            | **1, 7**            |
| 3      | 1     | Miss     | 0, 2                | 3, 7                |
| 4      | 0     | Miss     | 0, 4                | 3, 7                |
| 5      | 1     | Miss     | 0, 4                | 5, 7                |
| 6      | 0     | Miss     | 0, 6                | 5, 7                |
| **7**  | **1** | **Hit**  | **0, 6**            | **5, 7**            |
| **0**  | **0** | **Hit**  | **0, 6**            | **5, 7**            |

<div style="page-break-after:always;"></div>

# Problem 4
## (1)

| `x` | `y` | `w` | `z` | Explanation                                              |
| --- | --- | --- | --- | -------------------------------------------------------- |
| 2   | 2   | 1   | 0   | Core 3 & 4 -> Core 1 & 2                                 |
| 2   | 2   | 1   | 4   | Core 3 -> Core 1 & 2 -> Core 4                           |
| 2   | 2   | 1   | 2   | Core 3 -> Core 1 -> Core 4 -> Core 2 (1, 2 can exchange) |
| 2   | 2   | 3   | 0   | Core 4 -> Core 1 -> Core 3 -> Core 2 (1, 2 can exchange) |
| 2   | 2   | 3   | 4   | Core 1 -> Core 3 -> Core 2 -> Core 4 (1, 2 can exchange) |
| 2   | 2   | 3   | 2   | Core 1 -> Core 3 & 4 -> Core 2 (1, 2 can exchange)       |
| 2   | 2   | 5   | 4   | Core 1 & 2 -> Core 3 & 4                                 |
| 2   | 2   | 5   | 0   | Core 4 -> Core 1 & 2 -> Core 3                           |
| 2   | 2   | 5   | 2   | Core 1 -> Core 4 -> Core 2 -> Core 3 (1, 2 can exchange) |

## (2)
Using lock, make sure the reading is after the writing (or before, just try to fix the order of different processor.)

<div style="page-break-after:always;"></div>

# Problem 5
## (1)

| Feature              | CPU                                     | GPU                         |
| -------------------- | --------------------------------------- | --------------------------- |
| **Cores**            | Fewer cores                             | Thousands of smaller cores  |
| **Clock Speed**      | Higher clock speeds per core            | Lower clock speeds per core |
| **Cache**            | Larger cache for low-latency operations | Smaller cache per core      |
| **Memory Bandwidth** | Moderate (for low latency)              | High (for more throughput)  |

## (2)

| Task                                    | Processor | Reason                                                               |
| --------------------------------------- | --------- | -------------------------------------------------------------------- |
| Sorting 10,000 integers (quick sort)    | CPU       | Irregular memory access and recursive calls.                         |
| Sorting 10,000 integers (counting sort) | GPU       | Parallelizable counting operations.                                  |
| Training a large neural network         | GPU       | Heavy matrix operations, optimized for parallel computation.         |
| Summing 2302^{30}230 numbers            | GPU       | Massively parallel reduction algorithm.                              |
| Running a physics simulation            | Depends   | Depends on task complexity (GPU for numerical; CPU for logic-heavy). |
| Running a virtual machine               | CPU       | Requires general-purpose processing and efficient control logic.     |
