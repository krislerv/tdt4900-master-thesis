# tdt4501-specialization-project
Code for my master project
Based on the code from https://github.com/olesls/master_thesis

# Requirements
Python 3  
Pytorch  
Tensorflow (for Tensorboard)  
Numpy  
Pickle
Scipy
gpustat

# Usage

## Datasets
You need to download one of the datasets (Sub-reddit or Last.fm).  
They can be found here:  
  
[Sub-reddit](https://www.kaggle.com/colemaclean/subreddit-interactions)  
[Last.fm](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html)  
  
The code assumes that the datasets are located in:  
~/datasets/subreddit/  
~/datasets/lastfm/  
But you can of course store them somewhere else and change the path in the code.  
  
## Preprocessing
The sub-reddit and Last.fm dataset have a very similar structure, so the preprocessing of both these is done with `preprocess.py`. In this file you must specify which dataset you want to run preprocessing on (`dataset = reddit` or `dataset = lastfm`), and also make sure that the path to the dataset is correct.

In order to create a CET version of the Last.fm dataset, you must set the `create_lastfm_cet` variable to True.
  
Preprocessing is done in multiple stages, and the data is stored after each stage. This is to make it possible to change the preprocessing wihout having to do all the stages each time. This is especially useful for the Last.fm dataset, where it takes some time to convert the original timestamps.  
  
The first stage is to convert timestamps and ensure that the data in sub-reddit and Last.fm is in the same format. Each dataset has its own method to do this. After this, both datasets use the same methods.  
The second stage is to map items/labels to integers.  
The third stage split the data into sessions.  
The fourth stage splits the data into a training set and a test set.  
The fifth stage creates a hold-one-out version of the training/test split, for testing with BPR-MF.  
  
The data in the Instacart dataset is on a very different format than the other two, therefore it has its own preprocessing code, `preprocess_instacart.py`. The stages are similar, but Instacart does not have timestamps, and is sorted into sessions as is.  
  
In `preprocess.py`, you can specify the time delta between user interactions that is the limit deciding whether two actions belong to the same session or two different ones. This is done by specifying `SESSION_TIMEDELTA` in seconds. Additionally, you can specify the maximum number of user interactions for a session with `MAX_SESISON_LENGTH`. Longer sessions are splitted, to accomodate this limit.  
`MAX_SESSION_LENGTH_PRE_SPLIT` defines the longest session length that will be accepted. I.e. sessions longer than this limit are discarded as outliers, and sessions within this limit are kept and potentially splitted if needed.  
`PAD_VALUE` specifies the integer used when padding short sessions to the maximum length. Tensorflow requires all sessions to have the same length, which is why shorter sessions must be padded. There should be no reason to change this value, and doing so will probably cause a lot of problems.  
  
After running preprocessing on a dataset, the resulting training and testing set are stored in a pickle file, `4_train_test_split.pickle`, in the same directory as the dataset.   
  
If you want to use a different dataset, you can either do all the preprocessing yourself, or try to configure `preprocess.py` if your dataset is on a format similar to sub-reddit and Last.fm. In any case, I suggest looking at `preprocess.py` to see how the data needs to be formatted after the preprocessing.  
  
# Running the RNN models
`train_attn.py` is the file for training and testing the system with attention model. The baseline model is `train_ii_rnn.py`.   
  
You need to specify the `dataset` variable, in the code, to the dataset you want to run the model on. Preprocessing need to have been performed on that dataset. If you don't have the dataset stored in `~/datasets/...`, you must set the `dataset_path` variable accordingly instead.  

Testresults are stored in `testlog/`.  
  
### Parameters
`use_last_hidden_state` defines whether to use last hidden state or average of embeddings as session representation.  
`BATCH_SIZE` defines the number of sessions in each mini-batch.  
`INTRA_INTERNAL_SIZE` defines the number of nodes in the intra-session RNN layer (ST = Short Term)  
`INTER_INTERNAL_SIZE` defines the number of nodes in the inter-session RNN layer. These two depends on each other and needs to be the same size as each other and as the embedding size. If you want to use different sizes, you probably need to change the model as well, or at least how session representations are created.  
`LEARNING_RATE` is what you think it is.  
`DROPOUT_RATE` is the propability to drop a random node. So setting this value to 0.0 is equivalent to not using dropout.  
`MAX_SESSION_REPRESENTATIONS` defines the maximum number of recent session representations to consider.  
`MAX_EPOCHS` defines the maximum number of training epochs before the program terminates. It is no problem to manually terminate the program while training/testing the model and continue later if you have `save_best = True`. But notice that when you start the program again, it will load and continue training the last saved (best) model. Thus, if you are currently on epoch #40, and the last best model was achieved at epoch #32, then the training will start from epoch #32 again when restarting.  
`N_LAYERS` defines the number of GRU-layers used in the intra-session RNN layer.  
`TOP_K` defines the number of items the model produces in each recommendation.  
`use_hidden_state_attn` decides whether or not to use the hidden representation attention mechanism in the inter-session RNN.  
`use_delta_t_attn` decides whether or not to use the delta-t attention mechanism in the inter-session RNN.  
`use_week_time_attn` decides whether or not to use the week-time attention mechanism in the inter-session RNN.  
`use_intra_attn` decides whether or not to use the intra attention mechanism.  

  
# Visualizing attention weights
Visualization of attention weights is done in `visualizer_inter.py` and `visualizer_intra.py`.
After running the datasets through preprocessing, you should have two files called `<dataset>_map.txt` and `<dataset>_remap.txt`. These must be in the same directory as the visualizer files. While training, the system will log attention weights into files with the names `*_attn_weights-*`. Each separate logging is separated by several empty lines. You must copy one of these logging instances into a separate textfile called either `attn_weights_intra.txt` or `attn_weights_inter.txt` depending on the type of attention weight. After this is done, you should only have to run the visualizer files to see the visualization.  
At the top of the Visualizer class, there is a variable, `show_timestamp`. If set to True, it will show the time difference from each session representation to the new session. If set to False, it will show the events in the session representation instead.

# Other files
The `datahandler*.py` files are used by the `train*.py` files to manage the datasets. The utils files takes control of retrieving mini-batches, logging results, storing session representations and some other stuff.

`test_util.py` is used by all models to score recommendations. It creates a first-n score. So remember that the first-n score for the highest value of n is equal to the overall score.  
`models*.py` contains the models used by `train*.py`.

# Other questions
If anything is unclear or there are any problems, just send me a message :)