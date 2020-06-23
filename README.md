# A  sequence  to  sequence(seq2seq)  model  for  Chinese  to  English  translation

## 1. Problem Statement
This repository trains an Encoder-Decoder seq2seq model with **Bidirection-GRU, Fasttext word embedding, Attention mechanism, K-Beam search** for Chinese to English Neural machine translation, and evaluate by **BLEU score**.

## 2. Data Description
The dataset is based on the **UM-Corpus** , which is a Large English-Chinese Parallel Corpus for Statistical Machine Translation.
It provides two million English-Chinese aligned corpus categorized into eight different text domains, covering several topics and text genres, including: 
Education, Laws, **Microblog** , News, Science, Spoken, Subtitles, and Thesis[1].  
  
*The UM-Corpus can be downloaded from [http://nlp2ct.cis.umac.mo/um-corpus/um-corpus-license.html](http://nlp2ct.cis.umac.mo/um-corpus/um-corpus-license.html)*  
  
**To avoid OOM error on server, here in this repository we only use the Microblog set for trainning and Twitter set for testing.** The Microblog training data file includes 5,000 Bilingual sentences, and the Twitter testing data file includes 66 Bilingual sentences. Both files are encoded in UTF-8.  

## 3. Data Pre-Processing  
### 1) Segmentation:  
Here we use a Java implementation tool - Stanford Word Segmenter to preform tokenization on the raw dataset[2](*available from [https://nlp.stanford.edu/software/segmenter.html#Download](https://nlp.stanford.edu/software/segmenter.html#Download)*).  
For English, the segmenter splits the punctuation and separates some affixes like possessives.  
For Chinese, which is standardly written without spaces between words, the segmenter splits texts into a sequence of words according to some word segmentation standard.  
### 2) Normalization:
For Chinese sentences, we covert exotic words to lowercase, then trim and remove non-letter characters.   
For English sentences, we turn the text from Unicode to plain ASCII, covert to lowercase, trim and remove non-letter characters, then as the target language we add a start and an end token to the sentence so that the model knows when to start and stop predicting.  
### 3) Tokenization:  
Turning each text into a sequence of integers, each integer being the index of a token in the dictionary. Only top num_words-1 most frequent words will be taken into account, num_words was set 160000 for Input Chinese vocabulary and 80000 for target English vocabulary by default[3][11]. (In Microblog data there are only 13756 Chinese tokens, and 11113 English tokens with zero padding.)
### 4) Zero padding:
Zero padding the sentences with max-lenght of the sentences in the dataset or a small max_length to truncat the sentence in order to reduce the memory consumption, for example 35. (91% sentences have the length less than or euqel to 35 in Mircoblog set.)
### 5) One hot encoding:
Here we use **Keras functional API** create model, so we need to specify the encoder_inputs, decoder_inputs, and decoder_outputs before start training, the decoder_outputs is the same as decoder_inputs but offset by one timestep. **Note that decoder_outputs needs to be one-hot encoded**.  

## 4. Word Embedding
In the code we use a flag to switch using pre_trained fasttext word_embedding by default, or train a fasttext model to get word embedding.
The pre_trained **fasttext word vectors**[4] trained on Common Crawl and Wikipedia using fastText, using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.   
*Download from [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html).*  
When the flag switch to 1, we train a fasttext model from gensim on the Microblog data in dimension 100, with a window of size 10, min_count of number 5, skip-gram set to True, iteration of number 20 and 10 negatives.  
No matter the flag set to 1 or 0, we need to customize the embeddings by mapping the vectors to the vocabulary based on the Microblog dataset.  

## 5. The Seq2seq Model
We use **Keras funtional API** to creat a **Bidirection-GRU with word embedding and Attention mechnism encoder-decoder Seq2seq model**[5][6][7][8], the model framework is shown below:
  
![image](https://github.com/lilanpei/NMT/blob/master/model.png)  

The total number of parameters is 24,235,213 include trainable parameters 16,774,513 and Non-trainable parameters: 7,460,700  

## 6. Training Details  
We use fit_generator() instead of the fit() method as our data is too large to fit into the memory[7]. After shuffle the Microblog dataset, we creat training and validation sets using an 80-20 split,
so there are 4000 bilingual sentences for training and 1000 for validation. We trained the network with batchsize of 32(compromised on the memory consumption and the time), adam optimizer and categorical_crossentropy  loss function for a total of 40 epochs. The training accuracy converged to 0.9996 and the val accuracy reached 0.7869.
The loss and accuracy history plots as below:  
  
![image](https://github.com/lilanpei/NMT/blob/master/Loss_history.png)  
![image](https://github.com/lilanpei/NMT/blob/master/Accuracy_history.png)  
Here shows an attention heatmap plotting example:
![image](https://github.com/lilanpei/NMT/blob/master/attention.png)  

## 7. Experiments  
The decoder will generate the predicted translation using k beam search strategy[9](k = 3 by default), we evaluate the prediction by BLEU score[10].  
The average BLEU score for random 1000 validation set is 48.10  
The average BLEU score for 66 Twitter testing set (after re-trained the model on the whole Mircoblog set) is 54.17  

## 9. Conclusion  
From the predicted translation we can see sometimes the prediction is really good, the docoder can generate the same or very closed meaning compare to the original sentence,
sometimes the prediction crashed at very begining, but consider the number of training sample is just 5000, 2.08% of the whole UM-Corpus, that is a resonable result.
In another way, compare to the 28.67 BLEU score of the Statistical Machine Translation on UM-Corpus[1], it's already a big progress.

## 10. Future Work  
1) In the UM-Corpus, we can see some sentence have a exaggerated length for example 264, which will cause problem such as memory consumption if we use it as the max-lenght for zeor-padding, low accuaracy for long sentence if truncat the sentence with a small lenght, so some long sentences should be split more.  
2) We use only Mircoblog set in UM-Corpus to reduce memory consumption and long time for training, but it will be worthwhile testing it on the full scale dataset.  
3) Using Keras functional API is easy to setup the model but we need to specify the decoder_outputs with one-hot encoding which will cause memory consumption problem as the size of target vocabulay increase, so it's better to change to another framework.  

## 11. Code Description
*Start the StanfordCoreNLPServer.ipynb* : For data segmentation.  
*NMT - Microblog - ZH to EN - Bi-GRU + Attention + Fasttext word embedding + k-Beam search + BLEU score.ipynb* : For Data Pre-Processing, model setup, training and testing.  


## Reference 
[1] Liang Tian, Derek F. Wong, Lidia S. Chao, Paulo Quaresma, Francisco Oliveira, Shuo Li, Yiming Wang, Yi Lu, "UM-Corpus: A Large English-Chinese Parallel Corpus for Statistical Machine Translation". Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC'14), Reykjavik, Iceland, 2014.  
[2] Huihsin Tseng, Pichuan Chang, Galen Andrew, Daniel Jurafsky and Christopher Manning. 2005. A Conditional Random Field Word Segmenter. In Fourth SIGHAN Workshop on Chinese Language Processing.  
[3] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in neural information processing systems. 2014: 3104-3112.  
[4] Grave E, Bojanowski P, Gupta P, et al. Learning word vectors for 157 languages[J]. arXiv preprint arXiv:1802.06893, 2018.  
[5] [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)  
[6] [TensorFlow Core - Neural machine translation with attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)  
[7] [Implementing neural machine translation using keras](https://towardsdatascience.com/implementing-neural-machine-translation-using-keras-8312e4844eb8)  
[8] [thushv89/attention_keras](https://github.com/thushv89/attention_keras)  
[9] [How to Implement a Beam Search Decoder for Natural Language Processing](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing)  
[10] [Calculate BLEU score in Python](https://stackoverflow.com/questions/32395880/calculate-bleu-score-in-python/39062009)  
[11] [Using Tokenizer with num_words #8092](https://github.com/keras-team/keras/issues/8092)  
