- Replicate mel-spec with esssentia
  - Train the replication and best model again
  - Warning: essentia can currently only accept numpy arrays of dtype "single". "signal" dtype is double. Precision will be automatically truncated into "single".??*

- Understand Evaluation Metrics
 - What is mini-search-validation?

- Understand the dataset
  - Train: 10k 30sec clips
  - Val: 500 tracks of length???
  - Test:
    - dummy_db: approx. 10k noise tracks. Length?
    - test-query-db: supposed to be 2000 tracks randomly cropped from ??
      - db: 500
      - query: 2000(? 500?) tracks of 30 sec. Created from Val set?? Data Augmented
      - query_fixed_SNR: 1000 tracks. What is this?

- Create a discotube dataset
  - discotube is mp3
  - Make sure that test set noise tracks contain multiple tracks from same artist. Separate val and test set
- Train on discotube
  - What should be the dataset size? Think about the training time 10k=>2days 100k=>20days?
  - Can we paralelize?

- Write your own dataset class ?
  - Will it interfere with generate.py?

- Fix the neural net's last convolutional layers

- Try 50% overlap maybe 75% is too much