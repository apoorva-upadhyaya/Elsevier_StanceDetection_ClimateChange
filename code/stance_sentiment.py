import tensorflow as tf
from keras import backend as k
from keras import layers
from keras.layers.core import Lambda
import numpy as np 
from numpy import asarray,zeros, array
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,GRU,Reshape
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import optimizers,regularizers
import statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix
import torch
from transformers import AutoModel, AutoTokenizer 
from transformers import pipeline


def attentionScores(var):
    Q_t,K_t,V_t=var[0],var[1],var[2]
    scores = tf.matmul(Q_t, K_t, transpose_b=True)
    # print("first scores shape:::::",scores.shape)
    distribution = tf.nn.softmax(scores)
    scores=tf.matmul(distribution, V_t)
    # print("scores shape:::::",scores.shape)
    return scores


def create_resample(train_sequence,train_enc,train_enc_senti,train_enc_temp,train_embeddings):
    df = pd.DataFrame(list(zip(train_sequence,train_enc,train_enc_senti,train_enc_temp,train_embeddings)), columns =['text','pol', 'sent','temp','use'],index=None)
    blv=(df[df['pol'] == 0])
    print("len blv",len(blv))
    deny=(df[df['pol'] == 1])
    print("len deny",len(deny))
    upsampled1 = resample(deny,replace=True, # sample with replacement
                          n_samples=len(blv), # match number in majority class
                          random_state=27)

    upsampled = pd.concat([blv,upsampled1])
    upsampled=upsampled.sample(frac=1)
    print("After oversample train data : ",len(upsampled))
    print("After oversampling, instances of tweet act classes in oversampled data :: ",upsampled.pol.value_counts())

    train_data=upsampled
    train_sequence=[]
    train_enc=[]
    train_enc_senti=[]
    train_enc_temp=[]

    train_embeddings=[]
   
    for i in range(len(train_data)):
        train_sequence.append(np.array(train_data.text.values[i]))
        train_enc.append(train_data.pol.values[i])
        train_enc_senti.append(train_data.sent.values[i])
        train_enc_temp.append(train_data.temp.values[i])
        train_embeddings.append(np.array(train_data.use.values[i]))


    return train_sequence,train_enc,train_enc_senti,train_enc_temp,train_embeddings


data=pd.read_csv("../../data/final_data.csv", delimiter=";", na_filter= False) 
########### creating multiple lists as per the daataframe
li_text=[]
li_senti=[]
li_stance=[]
li_id=[]
li_temp=[]

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    li_id.append(data.tweetid.values[i])
    li_text.append(data.text.values[i])
    li_senti.append((data.sentiment.values[i]))
    li_temp.append((data.temporal_orientation.values[i]))
    li_stance.append(data.stance.values[i])


print("data np unique:::",np.unique(li_stance,return_counts=True))


########### converting act labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_stance
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
li_enc_stance=total_integer_encoded
total_integer_encoded=to_categorical(total_integer_encoded)
li_stance=total_integer_encoded

########### converting senti labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_senti
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
li_enc_senti=total_integer_encoded
total_integer_encoded=to_categorical(total_integer_encoded)
li_senti=total_integer_encoded

########### converting temporal labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_temp
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
li_enc_temp=total_integer_encoded
total_integer_encoded=to_categorical(total_integer_encoded)
li_temp=total_integer_encoded

########## converting text modality into sequence of vectors ############

total_text = [x.lower() for x in li_text] 

tokenizer_btw = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
model_btw = AutoModel.from_pretrained("vinai/bertweet-base")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_btw.resize_token_embeddings(len(tokenizer_btw))
pipe = pipeline('feature-extraction', model=model_btw, tokenizer=tokenizer_btw)
pipe_data = pipe(total_text)

input_sequence=[]
set_token=set()
for i in range(len(total_text)):
    sent=total_text[i]
    embed=pipe_data[i][0]
    li_token=pipe.tokenizer.encode(sent)
    input_sequence.append(li_token)

MAX_SEQ=100
MAX_LENGTH=100

padded_docs = pad_sequences(input_sequence, maxlen=MAX_SEQ, padding='post')

total_sequence=padded_docs#text
vocab_size = 62811 + 1
print("vocab_size:::",vocab_size)
print("downloading ###")
embedding_matrix= np.load("../../embedding_matrix/embed_matrix_bertweet_250.npy")
print("embedding matrix ****************",embedding_matrix.shape)

total_embeddings_use= np.load("../../embedding_matrix/embed_use_reshape_100.npy")
print("total_embeddings_use :::",total_embeddings_use.shape)

li_stance=np.array(li_stance)
li_senti=np.array(li_senti)
li_temp=np.array(li_temp)


#######data for K-fold #########

list_acc_stance,list_acc_senti,list_acc_temp=[],[],[]
list_f1_stance,list_f1_senti,list_f1_temp=[],[],[]
list_prec_stance,list_prec_senti,list_prec_temp=[],[],[]
list_rec_stance,list_rec_senti,list_rec_temp=[],[],[]


kf=StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
fold=0
results=[]
for train_index,test_index in kf.split(total_sequence,li_enc_stance):
    print("K FOLD ::::::",fold)
    fold=fold+1

    ######### stance inputs ########
    input1_stance= Input (shape = (MAX_LENGTH, ))
    input_text_stance= Embedding(vocab_size, 768, weights=[embedding_matrix], input_length=100, name='text_share_embed_stance')(input1_stance)
    lstm_stance= Bidirectional(LSTM(100, name='lstm_inp1_stance', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_stance)
    text_final_stance= lstm_stance

    input2_stance= Input (shape = (512,1))
    lstm_stance= Bidirectional(LSTM(100, name='lstm_inp2_stance', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input2_stance)
    text_use_stance= lstm_stance
    
    T1=Concatenate()([text_final_stance,text_use_stance])

    Q_t= Dense(100, activation="relu")(T1)
    Q_t= Reshape((-1,1))(Q_t)
    K_t= Dense(100, activation="relu")(T1)
    K_t= Reshape((-1,1))(K_t)
    V_t= Dense(100, activation="relu")(T1)
    V_t= Reshape((-1,1))(V_t)

    IA_T1=Lambda(attentionScores)([Q_t,K_t,V_t])

    ############## Sentiment inputs #############
    input1_senti = Input (shape = (MAX_LENGTH, ))
    input_text_senti = Embedding(vocab_size, 768, weights=[embedding_matrix], input_length=100, name='text_share_embed_senti')(input1_senti)
    lstm_senti = Bidirectional(LSTM(100, name='lstm_inp1_senti', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_senti)
    text_final_senti = lstm_senti

    input2_senti = Input (shape = (512,1))
    lstm2_senti = Bidirectional(LSTM(100, name='lstm_inp2_senti', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input2_senti)
    text_use_senti = lstm2_senti

    T2=Concatenate()([text_final_senti,text_use_senti])

    Q_e= Dense(100, activation="relu")(T2)
    Q_e= Reshape((-1,1))(Q_e)
    K_e= Dense(100, activation="relu")(T2)
    K_e= Reshape((-1,1))(K_e)
    V_e= Dense(100, activation="relu")(T2)
    V_e= Reshape((-1,1))(V_e)

    IA_T2=Lambda(attentionScores)([Q_e,K_e,V_e])

    ################ Average of 3 tensors ##############
    M=Average()([T1,T2])

    Q_m= Dense(100, activation="relu")(M)
    Q_m= Reshape((-1,1))(Q_m)
    K_m= Dense(100, activation="relu")(M)
    K_m= Reshape((-1,1))(K_m)
    V_m= Dense(100, activation="relu")(M)
    V_m= Reshape((-1,1))(V_m)

    IRAScores_T1=Lambda(attentionScores)([Q_t,K_m,V_m])
    IRAScores_T2=Lambda(attentionScores)([Q_e,K_m,V_m])
    
    T1_conc=Concatenate()([IA_T1,IRAScores_T1])
    T1_conc=Flatten()(T1_conc)
    T2_conc=Concatenate()([IA_T2,IRAScores_T2])
    T2_conc=Flatten()(T2_conc)
    
    act_specific_output=Dense(2, activation="softmax", name="task_specific_act")(T1_conc)
    senti_specific_output=Dense(3, activation="softmax", name="task_specific_senti")(T2_conc)
    
    model=Model([input1_stance,input2_stance,input1_senti,input2_senti],[act_specific_output,senti_specific_output])
    ##Compile
    model.compile(optimizer=Adam(0.0001),loss={'task_specific_act':'binary_crossentropy','task_specific_senti':'categorical_crossentropy'},
    loss_weights={'task_specific_act':1.0,'task_specific_senti':0.5}, metrics=['accuracy'])    
    print(model.summary())

    test_sequence,train_sequence=total_sequence[test_index],total_sequence[train_index]
    test_embeddings,train_embeddings=total_embeddings_use[test_index],total_embeddings_use[train_index]
    test_stance,train_stance=li_stance[test_index],li_stance[train_index]
    test_senti,train_senti=li_senti[test_index],li_senti[train_index]
    test_temp,train_temp=li_temp[test_index],li_temp[train_index]

    test_enc = np.argmax(test_stance, axis=1)
    train_enc = np.argmax(train_stance, axis=1)
    train_enc_senti = np.argmax(train_senti, axis=1)
    train_enc_temp = np.argmax(train_temp, axis=1)
    
    print("len of train",np.unique(train_enc,return_counts=True),len(train_stance))
    print("len of test",np.unique(test_enc,return_counts=True),len(test_stance))
    
    train_sequence,train_enc,train_enc_senti,train_enc_temp,train_embeddings=create_resample(train_sequence,train_enc,train_enc_senti,train_enc_temp,train_embeddings)
    train_sequence=np.array(train_sequence)
    train_embeddings=np.array(train_embeddings)

    train_stance=to_categorical(train_enc)
    train_senti=to_categorical(train_enc_senti)
    train_temp=to_categorical(train_enc_temp)

    print("final input shape:::",train_sequence.shape,train_sequence[0].shape,train_embeddings.shape,train_embeddings[0].shape)


    model.fit([train_sequence,train_embeddings,train_sequence,train_embeddings],[train_stance,train_senti], shuffle=True,validation_split=0.2,epochs=20,verbose=2)
    predicted = model.predict([test_sequence,test_embeddings,test_sequence,test_embeddings])
    print(predicted)

    test_enc = np.argmax(test_stance, axis=1)
    act_pred_specific=predicted[0]
    result_=act_pred_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_enc, p_1)
    list_acc_stance.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['believe','deny']
    class_rep=classification_report(test_enc, p_1)
    print("specific confusion matrix",confusion_matrix(test_enc, p_1))
    print(class_rep)
    class_rep=classification_report(test_enc, p_1, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    macro_prec=class_rep['macro avg']['precision']
    macro_rec=class_rep['macro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_stance.append(macro_avg)
    list_prec_stance.append(macro_prec)
    list_rec_stance.append(macro_rec)

    ########### sentiment
    test_enc_senti = np.argmax(test_senti, axis=1)

    sent_pred_specific=predicted[1]
    result_=sent_pred_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_enc_senti, p_1)
    list_acc_senti.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['negative', 'neutral', 'positive']
    class_rep=classification_report(test_enc_senti, p_1)
    print("specific confusion matrix",confusion_matrix(test_enc_senti, p_1))
    print(class_rep)
    class_rep=classification_report(test_enc_senti, p_1, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    macro_prec=class_rep['macro avg']['precision']
    macro_rec=class_rep['macro avg']['recall']
    print("macro f1 score",macro_avg)

    list_prec_senti.append(macro_prec)
    list_rec_senti.append(macro_rec)

    print("macro f1 score",macro_avg)
    list_f1_senti.append(macro_avg)


############# Stance Detection 

print("ACCURACY :::::::::::: #############")
print("Accuracy  ::: ",list_acc_stance)
print("Mean, STD DEV", statistics.mean(list_acc_stance),statistics.stdev(list_acc_stance))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_stance)
print("MTL Mean, STD DEV", statistics.mean(list_f1_stance),statistics.stdev(list_f1_stance))

print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_prec_stance),statistics.stdev(list_prec_stance))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_rec_stance),statistics.stdev(list_rec_stance))

############# sentiment 

print("Sentiment :::::::::::: #############")
print("Accuracy  ::: ",list_acc_senti)
print("Mean, STD DEV", statistics.mean(list_acc_senti),statistics.stdev(list_acc_senti))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_senti)
print("MTL Mean, STD DEV", statistics.mean(list_f1_senti),statistics.stdev(list_f1_senti))

print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_senti)
print("MTL Mean, STD DEV", statistics.mean(list_prec_senti),statistics.stdev(list_prec_senti))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_senti)
print("MTL Mean, STD DEV", statistics.mean(list_rec_senti),statistics.stdev(list_rec_senti))