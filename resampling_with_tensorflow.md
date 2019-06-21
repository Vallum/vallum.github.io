# Under and over resampling with Tensorflow

- Imbalanced data set with multi class multi labeled images
- Not enough with rejection sampling
- With Tensorflow dataset API

## Offline Preparation Flow
- Collect sample distribution A
- Design resampled(target) distribution B
- Make a transformation (vector) T from A to B
- each column is (re)sampling probability from items in A
- If colume value p in T is less than 1, some of samples 100 * (1 - p) % are dropped.
- If columen value p is more than 1, samples are replicated p times. 

## Online Training Flow
- Loading tfrecords
- shuffle data records in memory (large shuffle)
- parsing record labels
- giving uniform probability(bet) for each record
- under resampling with map and flat_map
  - drop some of records whose probability(bet) is lower than each label's threshold given T
- over resampling with map and flat_map
  - replicate N times some of records whose probability given T(=N.xxx) is more than integer 1 (N > 1)
- shuffle data samples in memory (small shuffle)
- parsing record images
  - decode jpegs into Tensors
- make batch from shuffle buffer
  
## Disk I/O, Memory, CPU, Bus I/O, GPU parallelism
- Because of under sampling, resampling wastes much of disk I/O. 
  - Lots of samples just dumped right after loaded from disks.
  - So disk I/O is much heavier compared to the ordidary sampling.
  - And because of low latency of disk I/O, it is too long to wait to load after computation.
  - So prefetch is necessary between file reading and record parsing.
- The records in data files might be unshuffled.
  - So shuffle right after loaded to memory
- Record parsing is CPU's job.
  - for concurrent parsing, map with parallel_calls is useful.
- Undersampling before oversampling
  - undersampling is dumping away some of records.
  - it reduces the size of list.
  - it is waste of CPU resources to work with the data going to be dumped away.
  - tf.dataset.filter is not useful, because it doesn't provide parallelism.
  - map with parallel_calls is useful, but map handles the contents of record not the record itself.
  - to handle duplication or elimination of records, flat_map is necessary.
  - like this
  ```
  dataset = dataset.map(undersample_filter_fn, num_parallel_calls=num_parallel_calls) 
  dataset = dataset.flat_map(lambda x : x) 
  ```
  flat_map with the identity lambda function is just for mergning survived (and empty) records
  ![https://www.tensorflow.org/images/datasets_parallel_map.png](https://www.tensorflow.org/images/datasets_parallel_map.png)
