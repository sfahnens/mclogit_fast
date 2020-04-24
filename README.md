# mclogit_fast

Efficiently implemented estimators for the conditional logit model.\
(Approach based on/mostly compatible to: https://cran.r-project.org/package=mclogit).

## Setup:

Install from source with devtools or clone the repository and `R CMD INSTALL .`

## Basic Usage:

The interface of `mylogit_fast` is compatible to `mclogit`:
```R
library(mclogit_fast)
m <-  mclogit_fast(cbind(Y,CASE)~VAR_1+VAR2:VAR3, data = data)
summary(m)
```
See also [demo/demo.dense.R](demo/demo.dense.R) and [demo/demo.sparse.R](demo/demo.sparse.R).

## Variants:

- `mclogit_fast(..., parallel=FALSE, sparse=FALSE)` Baseline variant for dense problems, straightforward IRLS algorithm following mclogit in C++14 (see [src/mclogit_fast_dense_fit.h](src/mclogit_fast_dense_fit.h)).
- `mclogit_fast(..., parallel=TRUE, sparse=FALSE)` Parallelized (and more memory efficient) variant for dense problems (see [src/mclogit_fast_dense_fit_pll.h](src/mclogit_fast_dense_fit_pll.h))
- `mclogit_fast(..., parallel=TRUE, sparse=TRUE)` Optimized (memory efficient and parallelized) for sparse problems which occur in highly interacted models of the form `cbind(Y, CASE)~A:B:C:D:E`, where the one hot encoding of factors leads to a sparse design matrix (see [src/mclogit_fast_sparse_fit_pll.h](src/mclogit_fast_sparse_fit_pll.h)).

## Performance:

Code: [demo/bench.driver.R](demo/bench.driver.R)\
System: AMD Ryzen 7 3700X 8-Core, 32GB (+2GB swap), Ubuntu 20.04 LTS, R 3.6.3, GCC 9

**Dense problem**: `cbind(Y,CASE)~A+B+C+D+E` (5 columns, averages of 3 repeated runs)

NAME|NROW|USER TIME[s]|WALL TIME[s]|MAX RSS[MB]|DATASET[MB]
---|---|---|---|---|---
(1) mclogit|1e+06|14.209|14.543|1282|45
(2) mclogit_fast(ser)|1e+06|2.406|1.131|463|45
(3) mclogit_fast(pll)|1e+06|3.858|0.741|437|45
(1) mclogit|1e+07|118.878|129.139|10174|457
(2) mclogit_fast(ser)|1e+07|9.734|8.174|2647|457
(3) mclogit_fast(pll)|1e+07|27.118|4.522|2303|457
(1) mclogit|1e+08|-|-|-|-
(2) mclogit_fast(ser)|1e+08|79.328|76.395|24250|4577
(3) mclogit_fast(pll)|1e+08|261.788|40.308|20817|4577

**Sparse problem**: `cbind(Y,CASE)~A:B:C:D:(M+N+O)` (108 columns, averages of 3 repeated runs)

NAME|NROW|USER TIME[s]|WALL TIME[s]|MAX RSS[MB]|DATASET[MB]
---|---|---|---|---|---
(1) mclogit|1e+06|238.225|246.739|7595|45
(2) mclogit_fast(ser)|1e+06|72.959|72.011|2034|45
(3) mclogit_fast(pll)|1e+06|176.018|13.828|2034|45
(4) mclogit_fast(pll,sparse)|1e+06|170.882|12.644|676|45
(1) mclogit|1e+07|-|-|-|-
(2) mclogit_fast(ser)|1e+07|735.317|737.995|18365|457
(3) mclogit_fast(pll)|1e+07|1764.308|130.14|18021|457
(4) mclogit_fast(pll,sparse)|1e+07|1711.869|115.897|4517|457
(1) mclogit|1e+08|-|-|-|-
(2) mclogit_fast(ser)|1e+08|-|-|-|-
(3) mclogit_fast(pll)|1e+08|-|-|-|-
(4) mclogit_fast(pll,sparse)*|1e+08|17162.256|1187.137|31273|4577

\* Added second swapfile with 16GB to catch peaks during preprocessing - htop shows about 22ǴB during estimation (with about 10GB swapped out).

Note the ridiculously bad ratio of memory usage in `mclogit_fast` is caused by the fact, that our api is compatible to `mclogit`:
In order to support the formula interface, we need to do things the R way (TM).
This includes three copies of the dataset (original data frame, model frame, model matrix) and therefore dominates memory requirements.
