# CS5220-Final

# Extract Synthetic Datasets

* Download synthetic datasets from this [link](https://www.dropbox.com/scl/fi/kzikhocdpa57u3ugytw5t/synthetic.tar.gz?rlkey=8u0hetqfzjqfg9fmsf6dgwqyt&dl=1).

* Unzip folder 

```
tar -xvzf synthetic.tar.gz
```

The datasets have one of 3 types of causal graphs: sparse (S), super-sparse (SS), dense (D). The type of graph, number of columns and rows of each dataset is given by the file name as follows:

`X_[graph type]_[# columns]_[# rows]_0.csv`

# Run Serial Code 

```Bash
>>> cd LiNGAM
>>> make
>>> ./LiNGAM -i ../synthetic/X_S_100_1024_0.csv -o X_S_100_1024.out
```

* `-i` specifies the imput file and `-o` specifies the output file for the causal order.

# Run GPU Code

``` Bash
>>> cd GPULiNGAM
>>> make
>>> ./LiNGAM -i ../synthetic/X_S_100_1024_0.csv -o X_S_100_1024.out
```

# Correctness 

To obtain the correct causal order, use the following steps:

1. Change line 36 in `LiNGAM/main.cpp` to `causal_order = model.fit(X);`

2. Run the serial code as described above.

Warning! For 500 columns or more, the runtime of this procedure is more than 1.5 hours on an AMD EPYC 7763 CPU.
