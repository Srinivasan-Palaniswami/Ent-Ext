import numpy as np 
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from preprocess import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding,append_dict
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import pickle
from keras.models import load_model
import csv
import pandas as pd
import os

print("Enter Input Filename :")
input_filename=input()
print("Enter Output Filename :")
output_filename=input()

def initilize_df (key_lst,wrd_lst,predicted_lst):
            df1 = pd.DataFrame(np.column_stack([key_lst,wrd_lst,predicted_lst]), columns=['File_ID', 'Word', 'Entity'])
            #df1 = df1[df1.Entity != 'O']
            #print(df1)
            #df1.to_csv('E:/Srinivasan/8KM/MS Azure/Entity Extraction/Word_to_Vec/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs-master/result/Output_Correct.csv', mode='w', columns=['File_ID', 'Word', 'Entity'], index=False)
            return df1

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    tok = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tok.append(tokens)
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

word2Idx_file = open('C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/word2Idx.pkl', 'rb')   # 'rb' for reading binary file
word2Idx = pickle.load(word2Idx_file)
word2Idx_file.close()
char2Idx_file = open('C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/char2Idx.pkl', 'rb')   # 'rb' for reading binary file
char2Idx_1 = pickle.load(char2Idx_file)     
char2Idx_file.close()
label2Idx_file = open('C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/label2Idx.pkl', 'rb')   # 'rb' for reading binary file
label2Idx = pickle.load(label2Idx_file)     
label2Idx_file.close()
case2Idx_file = open('C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/case2Idx.pkl', 'rb')   # 'rb' for reading binary file
case2Idx = pickle.load(case2Idx_file)
case2Idx_file.close()

model = load_model("C:/Users/Test-Custom-Ent/Desktop/NER/NER_Domain/Models/model.h5")

all_files=os.listdir(input_filename)
for files in all_files:
        print(files)
        ip_file = files[:-4]
        testSentences = readfile(input_filename+"\\"+files)
        testSentences = addCharInformatioin(testSentences)
        text_voc = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]}{><!?:;#'\"/\\%$`&=*+@^~|"

        char2Idx = append_dict(char2Idx_1, testSentences, text_voc)

        test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

        idx2Label = {v: k for k, v in label2Idx.items()}
        wordlabel = {v: k for k, v in word2Idx.items()}

        labelSet = set()
        words = {}
        wrd_lst = []
        key_lst = []
        count = 1
        for dataset in [testSentences]:
                for sentence in dataset:
                        for token,char,label in sentence:
                                labelSet.add(label)
                                words[token.lower()] = True
                                wrd_lst.append(token)
                                key = ip_file
                                key_lst.append(key)

        test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

        idx2Label = {v: k for k, v in label2Idx.items()}
        wordlabel = {v: k for k, v in word2Idx.items()}

        #   Performance on test dataset       
        predLabels, correctLabels = tag_dataset(test_set)
        pre_test, rec_test, f1_test, label_pred, label_correct = compute_f1(predLabels, correctLabels, idx2Label)
        count = 0

        predicted_lst = []
        for pred_sent in label_pred:
            for pred_wrd in pred_sent:
                pred_wrd = pred_wrd[:-1]
                predicted_lst.append(pred_wrd)
        df1 = initilize_df(key_lst,wrd_lst,predicted_lst)
        print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

        ID = df1.File_ID.tolist()
        Entity = df1.Word.tolist()
        Type = df1.Entity.tolist()
        
        N_ID = []
        N_Entity = []
        N_Type = []
        N_count = 0
        count = 0
        Type_position_count = 0
        for B in Type:
            if B[0] == "I" and Type_position_count != 0:
                if B[2:] != N_Type[N_count-1][2:]:
                        N_ID.append(ID[count])
                        N_Type.append(Type[count])
                        N_Entity.append(Entity[count])
                        N_count = N_count+1
                else:
                        pre_count = count - 1
                        if  ID[count] == ID[pre_count]:
                                N_Entity[N_count-1] = N_Entity[N_count-1]+" "+Entity[count]
            else:
                N_ID.append(ID[count])
                N_Type.append(Type[count])
                N_Entity.append(Entity[count])
                N_count = N_count+1
            count = count+1
            Type_position_count = Type_position_count+1
        df1 = pd.DataFrame(np.column_stack([N_ID,N_Entity,N_Type]), columns=['ID', 'Entity', 'Type'])
        df1 = df1[df1.Type != 'O']
        N_ID = df1.ID.tolist()
        N_Entity = df1.Entity.tolist()
        N_Type = df1.Type.tolist()
        t_len = len(N_Type)
        for R in range(t_len):
            N_Type[R] = N_Type[R][2:]
        df1 = pd.DataFrame(np.column_stack([N_ID,N_Entity,N_Type]), columns=['ID', 'Entity', 'Type'])
        df1.to_csv(output_filename+'\\'+ip_file+".csv", mode='w', columns=['ID', 'Entity', 'Type'], index=False, header=None)
        #print(df1)
        print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

print("Enter Final Output CSV Path :")
final_csv=input()
with open(final_csv+'\\'+'Output.csv', 'ab') as op_file:
        op_file.write(bytes("ID,Entity,Type", 'utf-8'))
        op_file.write(bytes("\n", 'utf-8'))
        all_files=os.listdir(output_filename)
        for files in all_files:
                print(files)
                size = os.path.getsize(output_filename+'\\'+files)
                if size != 0:
                        with open(output_filename+'\\'+files, 'rb') as read_file:
                                for line in read_file:
                                        op_file.write(line)
                else:
                        op_file.write(bytes(files[:-4]+",NO_Entity,NO_Type", 'utf-8'))
                        op_file.write(bytes("\n", 'utf-8'))
