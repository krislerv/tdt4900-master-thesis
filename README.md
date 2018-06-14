# tdt4501-specialization-project
Code for my master project  

# Requirements
Python 3  
Pytorch  
Tensorflow (for Tensorboard)  
Numpy  
Pickle  
Scipy  

# Usage

## Datasets
You need to download one of the datasets (Reddit or Last.fm).  
They can be found here:  
  
[Reddit](https://www.kaggle.com/colemaclean/subreddit-interactions)  
[Last.fm](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html)  
  
The code assumes that the datasets are located in:  
`~/datasets/subreddit/`  
`~/datasets/lastfm/`  
And that the code is located in `~/<code>/`  

But you can of course store them somewhere else and change the path in the code.  
  
## Preprocessing
preprocess.py converts the data of the two datasets above to a common data structure. Several options are available when preprocessing.

If `create_lastfm_cet` is True, a dataset will be created of the lastfm data containing only users in the CET timezone. This is to be used in models where the timezone of users is important.

If `create_time_filtered_dataset` is True, a dataset will be created where only the first x months of each user's data is included. The `time_filter_months` variable decides how many months to include. Supported values are 1, 2 and 3.

If `create_user_statistic_filtered_dataset` is True, a dataset will be created that filters out user that are not within the given thresholds for average session length and session count. If `remove_above_avg_session_length` is True, all users with above average session length will be removed. If it is False, all users with below average session length will be removed. Similar for `remove_above_avg_session_count`.
  
After running preprocessing on a dataset, the resulting training and testing set are stored in a pickle file, `4_train_test_split.pickle`, in the same directory as the dataset.   
  
  
# Running the RNN models
`train_attn.py` is the file for the inter and intra attention models. The file for the hierarchical attention model is `train_attn_h.py` The baseline model is `train_inter.py`.   
  
Test results are stored in `~/tdt4501-specialization-project/testlog/`.  
  
### Parameters
`use_last_hidden_state` defines whether to use last hidden state or average of embeddings as session representation.  
`BATCH_SIZE` defines the number of sessions in each mini-batch.  
`INTRA_INTERNAL_SIZE` defines the number of nodes in the intra-session RNN layer (ST = Short Term)  
`INTER_INTERNAL_SIZE` defines the number of nodes in the inter-session RNN layer. These two depends on each other and needs to be the same size as each other and as the embedding size. If you want to use different sizes, you probably need to change the model as well, or at least how session representations are created.  
`LEARNING_RATE` is what you think it is.  
`DROPOUT_RATE` is the probability to drop a random node. So setting this value to 0.0 is equivalent to not using dropout.  
`MAX_SESSION_REPRESENTATIONS` defines the maximum number of recent session representations to consider.  
`MAX_EPOCHS` defines the maximum number of training epochs before the program terminates. Results are saved after every epoch and you can choose to save model parameters as well to continue later.  
`N_LAYERS` defines the number of GRU-layers used.  
`TOP_K` defines the number of items the model produces in each recommendation.  
`use_hidden_state_attn` decides whether or not to use the hidden representation attention mechanism in the inter-session RNN.  


#### Attention specific parameters
`use_hidden_state_attn` decides whether or not to use the hidden attention mechanism in the inter-session RNN.  
`use_delta_t_attn` decides whether or not to use the delta-t attention mechanism in the inter-session RNN.  
`use_week_time_attn` decides whether or not to use the week-time attention mechanism in the inter-session RNN.  
`use_per_user_inter_attn` decides whether or not to use per-user linear layers for computing attention weights in the inter-session RNN.  
 
`use_intra_attn` decides whether or not to use attention mechanism in the intra-session RNN.  
`intra_attn_method` decides the method of the intra attention mechanism.  
`use_per_user_intra_attn` decides whether or not to use per-user linear layers for computing attention weights in the intra-session RNN.  

`bidirectional` decides whether or not to use bidirectional RNNs.  

#### Hierarchical attention specific parameters
`method_on_the_fly` decides which method to use when creating session representations.  
`method_inter` decides which method to use when creating user representations.  
`use_delta_t_attn` decides whether or not to use delta-t attention when creating user representations.  


  
# Visualizing attention weights
Visualization of attention weights is done in `visualizer_inter.py` and `visualizer_intra.py`.
After running the datasets through preprocessing, you should have two files called `<dataset>_map.txt` and `<dataset>_remap.txt`. These must be in the same directory as the visualizer files. While training, the system will log attention weights into files with the names `*_attn_weights-*`. Each separate logging is separated by several empty lines. You must copy one of these logging instances into a separate textfile called either `attn_weights_intra.txt` or `attn_weights_inter.txt` depending on the type of attention weight. After this is done, you should only have to run the visualizer files to see the visualization.  
At the top of the Visualizer class, there is a variable, `show_timestamp`. If set to True, it will show the time difference from each session representation to the new session. If set to False, it will show the events in the session representation instead.
