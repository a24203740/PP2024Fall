# Q1-1: 

## Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? 

隨著 VECTOR_WIDTH 越大，Vector Utilization 也越來越低

## Why?

因為我的做法，會一直 exponent vector 的各元素一直 -1，然後用 exponent vector 的每個元素的值是否為 0 來產生 mask
（i.e. mask[i] = 0 if exponent_vector[i] == 0, else mask[i] = 1）
可以發現，每一個 vector 要跑幾次 loop，取決於 exponent vector 中最大的元素的值
而每計算一個 vector 會浪費的 vector lane，可以用這個 psuedo code 表示：
```
max_element = max(exponent_vector);
inactive_vector_lane_count = 0;

for (e in exponent_vector)
{
  inactive_vector_lane_count += (max_element - e)
}
```
可以觀察到，exponent vector 內每有一個元素的值 != 最大的元素值，就會浪費 vector lane
而 VECTOR_WIDTH 越長，exponent vector 就越有可能有更多的非最大元素值，浪費更多的 vector lane
所以，VECTOR_WIDTH 越大，Vector Utilization 也會越來越低

下面是 VECTOR_WIDTH 分別為 2 4 8 16 時的執行結果

VECTOR_WIDTH = 2:
```
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              2
Total Vector Instructions: 162515
Vector Utilization:        87.6%
Utilized Vector Lanes:     284622
Total Vector Lanes:        325030
************************ Result Verification *************************
ClampedExp Passed!!!
```

VECTOR_WIDTH = 4:
```
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              4
Total Vector Instructions: 94571
Vector Utilization:        82.3%
Utilized Vector Lanes:     311252
Total Vector Lanes:        378284
************************ Result Verification *************************
ClampedExp Passed!!!
```

VECTOR_WIDTH = 8:
```
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              8
Total Vector Instructions: 51627
Vector Utilization:        79.6%
Utilized Vector Lanes:     328624
Total Vector Lanes:        413016
************************ Result Verification *************************
ClampedExp Passed!!!
```

VECTOR_WIDTH = 16:
```
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              16
Total Vector Instructions: 26967
Vector Utilization:        78.3%
Utilized Vector Lanes:     337864
Total Vector Lanes:        431472
************************ Result Verification *************************
ClampedExp Passed!!!
```


# Q2-1
## Fix the code to make sure it uses aligned moves for the best performance.

加上 -mavx2 之後，可以觀測到產出的 assembely 變成了:
```
  vmovups (%rdi,%rcx,4), %ymm0
  vmovups 32(%rdi,%rcx,4), %ymm1
  vmovups 64(%rdi,%rcx,4), %ymm2
  vmovups 96(%rdi,%rcx,4), %ymm3
```
從 bytes offset 變成 32 bytes 的倍數，可以推測 `vmovups` 是一次 mov 32 bytes
所以我們需要告訴 compiler 我們的 array 是 32 bytes aligned
我們可以透過修改 `__builtin_assume_aligned` 內的數字來達成，也就是：
```
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);
```

# Q2-2

執行時間:
No vectorized: -> 6.933972
vectorized but not with AVX2: -> 1.721069
vectorized with AVX2: -> 1.1472835

## What speedup does the vectorized code achieve over the unvectorized code? 
vectorized 之後，執行的速度變成了 4 倍快
## What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)?
使用 AVX2 vectorized 之後，執行的速度只有變 1.5 倍快
## What can you infer about the bit width of the default vector registers on the PP machines? 
預設的 vectorized 的 Vector width 是 4，這點可以從 Compiler 的 remark 跟產出的 assembely 觀察到
而執行速度也確實變成 4 倍
一個 float 是 4 bytes，所以 vector width = 4 代表 16 bytes
因此推測 PP machines 的 default vector registers 的 bit-width 是 128 bits
## What about the bit width of the AVX2 vector registers?
AVX2 的 vector width 是 8，一樣可以從 Compiler 的 remark 跟產出的 assembely 觀察到
雖然執行速度沒有變 8 倍，而是只有 6 倍，可能是 AVX2 指令的 overhead 等因素
但仍能推測 AVX2 vector registers 的 bit-width 是 256 bits

# Q2-3
## Provide a theory for why the compiler is generating dramatically different assemblies.
打 Patch 前，Compiler 產出的 assembely 是 non-vectorized 版本
Compiler remark 說是因為有 unsafe dependent memory operations in loop，所以無法 vectorized

打 Patch 之後，Compiler 產出的 assembely 是 vectorized 的版本

assembely 會差這麼多的原因顯然就是因為有無 vectorized 的差別
而導致第一個版本的 code 沒有 vectorized 的原因，我的推測是這樣：

在第二個版本中，compiler 是根據 `if (b[j] > a[j])` 的結果，決定要把 **a[j] 跟 b[j] 的哪一個** assign 給 c[j]
所以 compiler 可以使用 maxps 這個 SIMD instruction，各自選出 a[j:j+3] 和 b[j:j+3] 的每個元素中比較大的值，把結果塞進 c[j:j+3]

但是第一個版本，compiler 是先把 a assign 給 c
然後再根據 `if (b[j] > a[j])` 的結果，決定**要不要把 b[j]** assign 給 c[j]
*或許我猜可能* 沒有一個 SIMD instruction 是可以做有條件的 mov（如果滿足條件就 mov，否則保留原值）

就算想要效仿上面第二個版本的做法，用 maxps 去達成類似的效果，但 if 內的條件是 `b[j] > a[j]`，但這次是要決定 **c[j] 跟 b[j] 的哪一個** assign 給 c[j]
雖然我們知道 c[j] == a[j]，但可能 Compiler 優化沒辦法辨認這件事情？
我試過把 if 內的條件改成 `b[j] > c[j]`，就看到 remark 顯示 loop 有被 vectorized，某種程度上能證明我上面的想法
雖然這樣修改讓他 vectorized 了，但產出的 code 仍然比第二版複雜非常多，我沒辦法搞懂造成這兩者差異的原因


