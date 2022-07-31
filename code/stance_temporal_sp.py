import tensorflow as tf
from keras import backend as K
from keras import backend as k
from keras import layers
from keras.layers.core import Lambda
import numpy as np 
from numpy import asarray,zeros, array
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,GRU,Reshape
from keras.models import Model
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.engine import Layer
from keras.optimizers import Adam,RMSprop,SGD
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import json
from keras.callbacks import EarlyStopping
from keras import optimizers,regularizers
import random
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix
import torch
from transformers import AutoModel, AutoTokenizer 
import tensorflow_hub as hub


# from keras.losses import MeanAbsoluteError

def convert_tensor(var):
    a=var[0]
    Q_t=tf.convert_to_tensor(a,dtype=np.float32)
    return Q_t


# def fusion_gate(var):
#     g=var[0]
#     a=var[1]



def attentionScores(var):
    Q_t,K_t,V_t=var[0],var[1],var[2]
    scores = tf.matmul(Q_t, K_t, transpose_b=True)
    print("first scores shape:::::",scores.shape)
    distribution = tf.nn.softmax(scores)
    scores=tf.matmul(distribution, V_t)
    print("scores shape:::::",scores.shape)
    return scores

def attentionScoresIRA(var):
    Q_t,K_t,V_t,Q_e,K_e,V_e=var[0],var[1],var[2],var[3],var[4],var[5]
    score_t = tf.matmul(Q_t, K_e, transpose_b=True)
    distribution_t = tf.nn.softmax(score_t)
    scores_t=tf.matmul(distribution_t, V_t)
    
    score_e = tf.matmul(Q_e, K_t, transpose_b=True)
    distribution_e = tf.nn.softmax(score_e)
    scores_e=tf.matmul(distribution_e, V_e)

    IRAScores=Concatenate()([scores_t,scores_e])

    return IRAScores


def attentionScoresIRA1(var):
    Q_t,K_t,V_t,Q_e,K_e,V_e=var[0],var[1],var[2],var[3],var[4],var[5]
    score_t = tf.matmul(Q_t, K_e, transpose_b=True)
    distribution_t = tf.nn.softmax(score_t)
    scores_e=tf.matmul(distribution_t, V_e)
    
    score_e = tf.matmul(Q_e, K_t, transpose_b=True)
    distribution_e = tf.nn.softmax(score_e)
    scores_t=tf.matmul(distribution_e, V_t)

    IRAScores=Concatenate()([scores_t,scores_e])
    print("IRAScores1 ::::::::::::::::::::::::::::::,scores_e;",IRAScores,scores_e)

    return IRAScores



def constraint(y_true, y_pred, D_Name, S_Name):
    kdot=tf.matmul(D_Name, k.transpose(S_Name))#; 
    #kdot=k.dot(d[D], d[S])
    #kdot=k.reshape(kdot, -1); print kdot.shape
    scalar=k.sum(kdot)#tf.norm(kdot, ord="euclidean", axis=None)##this need to be adjusted
    #print kdot.shape, scalar.shape, type(kdot), type(scalar)
    return scalar+k.sum(k.abs(y_true-y_pred))

def orthoLoss(D_Name, S_Name):
    def stdLoss(y_true, y_pred):
        return constraint(y_true, y_pred, D_Name, S_Name)
    return stdLoss


def create_resample(train_sequence,train_enc,train_enc_senti,train_embeddings):
    df = pd.DataFrame(list(zip(train_sequence,train_enc,train_enc_senti,train_embeddings)), columns =['text','pol', 'temp','use'],index=None)
    blv=(df[df['pol'] == 0])
    print("len blv",len(blv))
    deny=(df[df['pol'] == 1])
    print("len deny",len(deny))
    upsampled1 = resample(blv,replace=True, # sample with replacement
                          n_samples=len(deny), # match number in majority class
                          random_state=27)

    upsampled = pd.concat([deny,upsampled1])
    upsampled=upsampled.sample(frac=1)
    print("After oversample train data : ",len(upsampled))
    print("After oversampling, instances of tweet act classes in oversampled data :: ",upsampled.pol.value_counts())

    train_data=upsampled
    train_sequence=[]
    train_enc=[]
    train_enc_senti=[]
    train_embeddings=[]
   
    for i in range(len(train_data)):
        train_sequence.append(train_data.text.values[i])
        train_enc.append(train_data.pol.values[i])
        train_enc_senti.append(train_data.temp.values[i])
        train_embeddings.append(train_data.use.values[i])


    return train_sequence,train_enc,train_enc_senti,train_embeddings


train_data=pd.read_csv("../data/train_cleanText_shuffle.csv", delimiter=";", na_filter= False) 
print("Training data :: ",len(train_data))


########### creating multiple lists as per the daataframe
train_text=[]
train_senti=[]
train_pol=[]
train_id=[]
train_topic=[]
li_text=[]
li_senti=[]
li_pol=[]
li_id=[]
li_topic=[]
for i in range(len(train_data)):
    id_=str(train_data.tid.values[i])
    # if id_!="871851443266834432" and id_!="913446598423797760" and id_!="945300622152048640" :     
    sent=str(train_data.sent.values[i])
    # if sent !="neutral" :
    train_id.append(train_data.tid.values[i])
    train_text.append(train_data.text.values[i])
    train_senti.append((train_data.temp.values[i]))
    train_pol.append(train_data.pol.values[i])

print("train_pol np unique:::",np.unique(train_pol,return_counts=True))

########### test_data
test_data=pd.read_csv("../data/test_cleanText_shuffle.csv", delimiter=";", na_filter= False) 


test_text=[]
test_senti=[]
test_pol=[]
test_id=[]
test_topic=[]

for i in range(len(test_data)):
    id_=str(test_data.tid.values[i])
    sent=str(test_data.sent.values[i])
    # if sent !="neutral" :
    test_id.append(test_data.tid.values[i])
    test_text.append(test_data.text.values[i])
    test_senti.append((test_data.temp.values[i]))
    test_pol.append(test_data.pol.values[i])


print("test_pol np unique:::",np.unique(test_pol,return_counts=True))

########### converting act labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=train_pol+test_pol
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)

train_enc_act=total_integer_encoded[0:len(train_text)]
test_enc_act=total_integer_encoded[len(train_text):]
total_integer_encoded=to_categorical(total_integer_encoded)

train_pol=total_integer_encoded[0:len(train_text)]
test_pol=total_integer_encoded[len(train_text):]


########### converting senti labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=train_senti+test_senti
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)

train_enc_senti=total_integer_encoded[0:len(train_text)]
test_enc_senti=total_integer_encoded[len(train_text):]
total_integer_encoded=to_categorical(total_integer_encoded)

train_senti=total_integer_encoded[0:len(train_text)]
print("train_senti::::",train_senti[:5])
test_senti=total_integer_encoded[len(train_text):]


########## converting text modality into sequence of vectors ############
total_text= train_text+test_text
total_text = [x.lower() for x in total_text] 
####3 need to check
MAX_SEQ=29
tokenizer = Tokenizer()
tokenizer.fit_on_texts(total_text)
total_sequence = tokenizer.texts_to_sequences(total_text)
padded_docs = pad_sequences(total_sequence, maxlen=MAX_SEQ, padding='post')

train_sequence=padded_docs[0:len(train_text)] #text
test_sequence=padded_docs[len(train_text):]
vocab_size = len(tokenizer.word_index) + 1

print("downloading ###")
embedding_matrix= np.load("../embedding_matrix/embed_matrix_shuffle.npy")
print("embedding matrix ****************",embedding_matrix.shape)
print("non zeros bert :",sum(np.all(embedding_matrix, axis=1)))


# Prepare and save USE embeddings

'''
module_url="https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  test_embeddings = session.run(embed(test_text))
  # train_embeddings= session.run(embed(train_text))

np.save("../embedding_matrix/use_test_embeddings", test_embeddings)

train_text1=train_text[:15000]
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  # test_embeddings = session.run(embed(test_text))
  train_embeddings= session.run(embed(train_text1))

np.save("../embedding_matrix/use_train_embeddings1", train_embeddings)

train_text2=train_text[15000:30000]
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  # test_embeddings = session.run(embed(test_text))
  train_embeddings= session.run(embed(train_text2))

np.save("../embedding_matrix/use_train_embeddings2", train_embeddings)


train_text3=train_text[30000:45000]
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  # test_embeddings = session.run(embed(test_text))
  train_embeddings= session.run(embed(train_text3))

np.save("../embedding_matrix/use_train_embedding3", train_embeddings)

train_text4=train_text[45000:]
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  # test_embeddings = session.run(embed(test_text))
  train_embeddings= session.run(embed(train_text4))

np.save("../embedding_matrix/use_train_embedding4", train_embeddings)
'''
train_embeddings1= np.load("../embedding_matrix/use_train_embeddings1.npy")
train_embeddings2= np.load("../embedding_matrix/use_train_embeddings2.npy")
train_embeddings3= np.load("../embedding_matrix/use_train_embedding3.npy")
train_embeddings4= np.load("../embedding_matrix/use_train_embedding4.npy")
test_embeddings= np.load("../embedding_matrix/use_test_embeddings.npy")

train_embeddings12=np.vstack((train_embeddings1,train_embeddings2))
train_embeddings34=np.vstack((train_embeddings3,train_embeddings4))
train_embeddings=np.vstack((train_embeddings12,train_embeddings34))
print("shape train embedding :::",train_embeddings.shape)

train_ids=set(train_id)
test_ids=set(test_id)
print("train_ids**********",train_ids.intersection(test_ids))

train_pol=np.array(train_pol)
test_pol=np.array(test_pol)

train_senti=np.array(train_senti)
test_senti=np.array(test_senti)

MAX_LENGTH=29



########## glove embedding: Investigating the impact of emotion on temporal orientation in a deep multitask setting,Misinformation detection using multitask learning with mutual learning for novelty detection and emotion recognition
#######data for K-fold #########

########## model2 ##############
total_labels_pol= np.vstack((train_pol,test_pol))
print("len of total_labels_pol",len(total_labels_pol))
total_labels_senti= np.vstack((train_senti,test_senti))
total_sequence=np.vstack((train_sequence,test_sequence))
print("len of total_sequence",len(total_sequence))

total_bert=np.vstack((train_embeddings,test_embeddings))
print("len of total_bert",len(total_bert))

total_labels_pol_enc = np.argmax(total_labels_pol, axis=1)
list_acc_pol,list_acc_senti=[],[]
list_f1_pol,list_f1_senti=[],[]
list_prec_pol=[]
list_rec_pol=[]
list_prec_senti=[]
list_rec_senti=[]
kf=StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
fold=0
results=[]
for train_index,test_index in kf.split(total_sequence,total_labels_pol_enc):
    print("K FOLD ::::::",fold)
    fold=fold+1

    #### model2 fit ############
    test_sequence,train_sequence=total_sequence[test_index],total_sequence[train_index]
    test_embeddings,train_embeddings=total_bert[test_index],total_bert[train_index]

    test_pol,train_pol=total_labels_pol[test_index],total_labels_pol[train_index]
    test_senti,train_senti=total_labels_senti[test_index],total_labels_senti[train_index]
    test_enc = np.argmax(test_pol, axis=1)
    train_enc = np.argmax(train_pol, axis=1)
    train_enc_senti = np.argmax(train_senti, axis=1)
    
    print("len of train",np.unique(train_enc,return_counts=True),len(train_pol))
    print("len of test",np.unique(test_enc,return_counts=True),len(test_pol))
    print(train_index,test_index)
    # oversample = SMOTE()
    # X, y = oversample.fit_resample(train_sequence, train_enc)
    # X1, y1 = oversample.fit_resample(train_sequence_topic, train_enc_senti)
    train_sequence,train_enc,train_enc_senti,train_embeddings=create_resample(train_sequence,train_enc,train_enc_senti,train_embeddings)

    train_sequence=np.array(train_sequence)
    train_embeddings=np.array(train_embeddings)

    train_pol=to_categorical(train_enc)
    train_senti=to_categorical(train_enc_senti)



    input1_pol = Input (shape = (MAX_LENGTH, ))
    input_text_pol = Embedding(vocab_size, 768, weights=[embedding_matrix], input_length=29, name='text_share_embed_pol')(input1_pol)
    lstm_pol = Bidirectional(GRU(100, name='lstm_inp1_pol', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_pol)
    text_final_pol = Dense(100, activation="relu")(lstm_pol)

    input2_pol = Input (shape = (512, ))
    input_pol = Reshape((512,1))(input2_pol)
    lstm_pol = Bidirectional(GRU(100, name='lstm_inp2_pol', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_pol)
    text_use_pol = Dense(100, activation="relu")(lstm_pol)
    
    T1=Concatenate()([text_final_pol,text_use_pol])

    Q_t= Dense(100, activation="relu")(T1)
    K_t= Dense(100, activation="relu")(T1)
    V_t= Dense(100, activation="relu")(T1)

    IA_T1=Lambda(attentionScores)([Q_t,K_t,V_t])

    

    ############## Temporal inputs #############

    input1_temp = Input (shape = (MAX_LENGTH, ))
    input_text_temp = Embedding(vocab_size, 768, weights=[embedding_matrix], input_length=29, name='text_share_embed_temp')(input1_temp)
    lstm_temp = Bidirectional(LSTM(100, name='lstm_inp1_temp', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_temp)
    text_final_temp = Dense(100, activation="relu")(lstm_temp)

    input2_temp = Input (shape = (512, ))
    input_temp = Reshape((512,1))(input2_temp)
    lstm2_temp = Bidirectional(LSTM(100, name='lstm_inp2_temp', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_temp)
    text_use_temp = Dense(100, activation="relu")(lstm2_temp)

    T2=Concatenate()([text_final_temp,text_use_temp])

    Q_e= Dense(100, activation="relu")(T2)
    K_e= Dense(100, activation="relu")(T2)
    V_e= Dense(100, activation="relu")(T2)

    IA_T2=Lambda(attentionScores)([Q_e,K_e,V_e])

    

    ################ Average of 2 tensors ##############

    M=Average()([T1,T2])

    print("M::::",M)

    Q_m= Dense(100, activation="relu")(M)
    K_m= Dense(100, activation="relu")(M)
    V_m= Dense(100, activation="relu")(M)


    IRAScores_T1=Lambda(attentionScores)([Q_t,K_m,V_m])
    IRAScores_T2=Lambda(attentionScores)([Q_e,K_m,V_m])

    T1_conc=Concatenate()([IA_T1,IRAScores_T1])
    T2_conc=Concatenate()([IA_T2,IRAScores_T2])


    act_specific_output=Dense(2, activation="softmax", name="task_specific_act")(T1_conc)

    temp_specific_output=Dense(3, activation="softmax", name="task_specific_temp")(T2_conc)


    model=Model([input1_pol,input2_pol,input1_senti,input2_senti],[act_specific_output,temp_specific_output])


    ##Compile
    model.compile(optimizer=Adam(0.0001),loss={'task_specific_act':'binary_crossentropy','task_specific_temp':'categorical_crossentropy'},
    loss_weights={'task_specific_act':1.0,'task_specific_temp':0.3}, metrics=['accuracy'])    
    print(model.summary())


    model.fit([train_sequence,train_embeddings,train_sequence,train_embeddings],[train_pol,train_senti], shuffle=True,validation_split=0.2,epochs=7,verbose=2)
    predicted = model.predict([test_sequence,test_embeddings,test_sequence,test_embeddings])
    print(predicted)

    test_enc = np.argmax(test_pol, axis=1)

    act_pred_specific=predicted[0]
    result_=act_pred_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_enc, p_1)
    list_acc_pol.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['blv','deny']
    class_rep=classification_report(test_enc, p_1)
    print("specific confusion matrix",confusion_matrix(test_enc, p_1))
    print(class_rep)
    class_rep=classification_report(test_enc, p_1, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    macro_prec=class_rep['macro avg']['precision']
    macro_rec=class_rep['macro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_pol.append(macro_avg)
    list_prec_pol.append(macro_prec)
    list_rec_pol.append(macro_rec)

    ########### sentiment
    test_enc_senti = np.argmax(test_senti, axis=1)

    sent_pred_specific=predicted[1]
    result_=sent_pred_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_enc_senti, p_1)
    list_acc_senti.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['future', 'past', 'present']
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
print("Accuracy  ::: ",list_acc_pol)
print("Mean, STD DEV", statistics.mean(list_acc_pol),statistics.stdev(list_acc_pol))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_pol)
print("MTL Mean, STD DEV", statistics.mean(list_f1_pol),statistics.stdev(list_f1_pol))


print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_pol)
print("MTL Mean, STD DEV", statistics.mean(list_prec_pol),statistics.stdev(list_prec_pol))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_pol)
print("MTL Mean, STD DEV", statistics.mean(list_rec_pol),statistics.stdev(list_rec_pol))

############# Temporal 

print("Temporal :::::::::::: #############")
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
