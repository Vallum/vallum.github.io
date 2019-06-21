# Under and over resampling with Tensorflow

- Imbalanced (very large) data set with multi class multi labeled images
  - Learning Visual Features from Large Weakly Supervised Data - Armand Joulin, Laurens van der Maaten, Allan Jabri, Nicolas Vasilache, https://arxiv.org/abs/1511.02251
  - Exploring the Limits of Weakly Supervised Pretraining - Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, Laurens van der Maaten, https://arxiv.org/abs/1805.00932
- Not enough with rejection sampling 
  - in case, some samples are too rare, so it normally does not exist in one batche or even in many batches.
  - rejection is a job of consuming too much of resources.
- With Tensorflow dataset API
  - over-and-under sampling with tensorflow : [stackoverflow](https://stackoverflow.com/questions/47236465/oversampling-functionality-in-tensorflow-dataset-api)

## Offline Preparation Flow
- Collect sample distribution A
  - crawled and/or tagged data with not too much considering of the balance between class labels  
- Design resampled(target) distribution B
  - considered for the balance of 
- Make a transformation (vector) T from A to B
- each column is (re)sampling probability from items in A
- If colume value p in T is less than 1, some of samples 100 * (1 - p) % are goint to be dropped.
- If columen value p is more than 1, samples are replicated p times. 
- let's call p as a ratio factor

## Online Training Flow
- Loading tfrecords
- shuffle data records in memory (large shuffle)
- parsing record labels
- giving uniform probability(bet) for each record
- under resampling with map and flat_map
  - drop some of records whose probability(bet) is lower than each label's threshold(=ratio factor) given T
- over resampling with map and flat_map
  - replicate some of records N times whose ratio factor p(=N.xxx) given T is more than a integer 1 (N > 1)
- shuffle data samples in memory (small shuffle)
- parsing record images
  - decode jpegs into Tensors
- make batch from shuffle buffer
- train with batches
  
## Disk I/O, Memory, CPU, Bus I/O, GPU parallelism
- Because of under sampling, resampling wastes much of disk I/O. 
  - Lots of samples are dropped right after they are loaded from disks.
  - So with undersampling, disk I/O is a much heavier burden compared to the ordidary sampling.
  - And because of low latency of disk I/O, it is too long to wait to load after computation.
    - dropping can be continued very long time undefinitely
  - So prefetch is necessary between file reading and record parsing.
- The records in data files might be unshuffled.
  - So shuffle data right after they are loaded to memory
- Record parsing is CPU's job.
  - When having multi CPUs, map with parallel_calls is useful for concurrent parsing.
- Undersampling before oversampling
  - undersampling is dumping away some of records.
  - it reduces the size of list of data sequences.
  - it is waste of CPU resources to work with the data (in short time after) going to be dropped out.
  - tf.dataset.filter is not useful, because it doesn't provide parallelism.
  - map with parallel_calls is useful, but map handles the internal contents of records, not the records set itself.
  - to handle duplication or elimination of records, flat_map is necessary.
  - like this
  ```
  dataset = dataset.map(undersample_filter_fn, num_parallel_calls=num_parallel_calls) 
  dataset = dataset.flat_map(lambda x : x) 
  ```
  flat_map with the identity lambda function is just for merging survived (and empty) records
  ```
  #parallel calls of map('A'), map('B'), and map('C')
  map('A') = 'AAAAA' # replication of A 5 times
  map('B') = ''      # B is dropped
  map('C') = 'CC'    # replication of C twice
  # merging all map results
  flat_map('AAAA,,CC') = 'AAAACC'
  ```
  ![https://www.tensorflow.org/images/datasets_parallel_map.png](https://www.tensorflow.org/images/datasets_parallel_map.png)
 - decoding compressed images like jpeg is a totally CPU bounded job
   - and can be parallelized in map with parallel calls and should be.
 - decoded image tensors 
   
