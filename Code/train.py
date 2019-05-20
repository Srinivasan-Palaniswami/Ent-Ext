import numpy as np
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from preprocess import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding,append_dict
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import pickle

epochs = 75

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

trainSentences = readfile("C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Dataset/CEMP/train.txt")
#devSentences = readfile("E:/Srinivasan/8KM/MS Azure/Entity Extraction/Word_to_Vec/ALL_MODELS/Dataset/twitter/IOB/Final_single_file_IOB/valid.txt")
testSentences = readfile("C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Dataset/CEMP/valid.txt")

trainSentences = addCharInformatioin(trainSentences)
#devSentences = addCharInformatioin(devSentences)
testSentences = addCharInformatioin(testSentences)

labelSet = set()
words = {}

for dataset in [trainSentences, testSentences]:
    for sentence in dataset:
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("C:/Users/Test-Custom-Ent/Desktop/Word_Embeddings/glove.6B/glove.6B.300d.txt", encoding="utf-8")
# fEmbeddings = open("E:/Srinivasan/8KM/MS Azure/Entity Extraction/Word_to_Vec/NER_Chkpt/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/embeddings/glove.twitter.27B.200d.txt", encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)

#char2Idx = {"PADDING":0, "UNKNOWN":1}
char2Idx_1 = {"PADDING":0, "UNKNOWN":1, "…":2, "’":3, "—":4, "£":5, "“":6, "”":7, "–":8, "‘":9, "■":10, "速":11, "報":12, "米":13, "版":14, "：":15, "«":16, "\u200b":17, "š":18, "ě":19, "á":20, "í":21, "ř":22, "é":23, "【":24, "】":25, 'ِ':26, '＊':27, '□':28, '・':29, '·':30, '→':31, '（':32, '一':33, '）':34, '●':35, 'Ì':36, 'ö':37, 'Û':38, 'ò':39, '˙':40, 'à':41, '￥':42, 'の':43, '\ufeff':44, 'µ':45, '»':46, '‹':47, 'Œ':48, 'ó':49, 'Ä':50, '¿':51, 'ν':52, 'λ':53, '®':54, 'β':55, 'α':56,'ß':57, '\xad':58, '°':59, '←':60, '≡':61, '×':62, 'ι':63}
text_voc = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]}{><!?:;#'\"/\\%$`&=*+@^~|"

char2Idx_2 = append_dict(char2Idx_1, trainSentences, text_voc)
char2Idx = append_dict(char2Idx_2, testSentences, text_voc)

# print(char2Idx)

for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]}{><!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

word2Idx_file = "C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/word2Idx.pkl"
char2Idx_file = "C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/char2Idx.pkl"
label2Idx_file = "C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/label2Idx.pkl"
case2Idx_file = "C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/case2Idx.pkl"
pickle.dump(word2Idx, open(word2Idx_file, "wb"))
pickle.dump(char2Idx, open(char2Idx_file, "wb"))
pickle.dump(label2Idx, open(label2Idx_file, "wb"))
pickle.dump(case2Idx, open(case2Idx_file, "wb"))

train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx))
#dev_set = padding(createMatrices(devSentences,word2Idx, label2Idx, case2Idx,char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}

train_batch,train_batch_len = createBatches(train_set)
#dev_batch,dev_batch_len = createBatches(dev_set)
test_batch,test_batch_len = createBatches(test_set)

words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
character_input=Input(shape=(None,52,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
dropout= Dropout(0.5)(embed_char_out)
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, casing,char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()
# plot_model(model, to_file='model.png')

for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens, casing,char = batch       
        model.train_on_batch([tokens, casing,char], labels)
        a.update(i)
    print(' ')
model.save("C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/model.h5")
#   Performance on dev dataset        
#predLabels, correctLabels = tag_dataset(dev_batch)        
#pre_dev, rec_dev, f1_dev, label_pred, label_correct = compute_f1(predLabels, correctLabels, idx2Label)
#print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
#   Performance on test dataset       
predLabels, correctLabels = tag_dataset(test_batch)        
pre_test, rec_test, f1_test, label_pred, label_correct = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
###ETA: 0sTest-Data: Prec: 0.667, Rec: 0.723, F1: 0.694