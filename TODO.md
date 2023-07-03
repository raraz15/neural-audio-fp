# TODO

## Faster Train/Inference
- Try to make loading faster
  - Store all IR FFTs?
- Automatic Mixed precision

## Augmentation
### Signal Processing Understanding
  - Is not normalizing inputs and added noise a good choice?
  - Is normalizing IR good?
  - Is cutting IR good?
  - Is taking a short part of IR is good?
  - inverse fft + real?
### Improve for Training
  - Use more IR samples
  - Use more BG samples

## Code
- Write your own dataset class ?
  - Will it interfere with generate.py?
  - Benefit: Distributed GPU

## Discotube
- Create a discotube dataset
  - discotube is mp3
  - Make sure that test set noise tracks contain multiple tracks from same artist. Separate val and test set
- Train on discotube
  - What should be the dataset size? Think about the training time 10k=>2days 100k=>20days?
  - Can we paralelize?

## Architecture
- Fix the neural net's last convolutional layers

## Input Representation
- Try 50% overlap maybe 75% is too much

## Original Repo Understanding
- Understand the Baseline dataset
  - Train: 10k 30sec clips
  - Val: 500 tracks of length???
  - Test:
    - dummy_db: approx. 10k noise tracks. Length?
    - test-query-db: supposed to be 2000 tracks randomly cropped from ??
      - db: 500
      - query: 2000(? 500?) tracks of 30 sec. Created from Val set?? Data Augmented
      - query_fixed_SNR: 1000 tracks. What is this?