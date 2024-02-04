from email import header
import os
import h5py
from collections import Counter
from sklearn.model_selection  import train_test_split
import pandas as pd
import numpy as np

from utils import read_h5

version = 1
maxIntancesPerClass = 45

class DataReader():

    def __init__(self, datasets, kpModel, output_path):

        self.classes = []
        self.videoName = []
        self.data = []
        self.output_path = os.path.normpath(output_path)

        for dataset in datasets:

            path = os.path.normpath(f"output/{dataset}--{kpModel}.hdf5")
            classes, videoName, data = read_h5(path)
            
            classes = [_class.upper() for _class in classes]

            self.classes = self.classes + classes
            self.videoName = self.videoName + videoName
            self.data = self.data + data
    
    def deleteSelectedVideosToBan(self):

        df_selectedBanned = pd.read_csv("./dataCleaningFunctions/banned_selected_videos.csv", header=None)
        selectedBanned = [banned.replace('\\','/') for banned in df_selectedBanned[0]]

        # We go through the inverse of the list to use "pop" to delete the banned words
        for pos in range(len(self.videoName)-1, -1, -1):

            if self.videoName[pos] in selectedBanned:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)

    def deleteBannedWords(self):

        df_bannedWords = pd.read_csv("./bannedList.csv",encoding='latin1', header=None)
        bannedList = list(df_bannedWords[0])

        bannedList = bannedList + [ban.upper() for ban in bannedList] + ['él','tú','','G-R']#+ ['lugar', 'qué?', 'sí', 'manejar', 'tú', 'ahí', 'dormir', 'cuatro', 'él', 'NNN'] #["hummm"]

        for pos in range(len(self.classes)-1, -1, -1):
            if self.classes[pos] in bannedList:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)

    def limitIntancesPerClass(self):
        
        dict_class = {_class:[] for _class in set(self.classes)}

        for pos, _class in enumerate(self.classes):
            dict_class[_class].append(pos)

        for _class in set(self.classes):
            dict_class[_class] = dict_class[_class][:maxIntancesPerClass]

        ind_list = [ind for values in dict_class.values() for ind in values]

        for pos in range(len(self.classes)-1, -1, -1):
            if pos not in ind_list:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)

    def generate_meaning_dict(self, words_dict):

        meaning = {v:k for (k,v) in words_dict.items()}
        self.labels = [meaning[_class] for _class in self.classes]

    def fixClasses(self):

        self.classes = list(map(lambda x: x.replace('amigos', 'amigo'), self.classes))

        _before = len(self.classes)
        self.deleteSelectedVideosToBan()

        print(f"About {_before - len(self.classes)} instances has been deleted by the ban list 'selectedVideos'")
        
        _before = len(self.classes)
        self.deleteBannedWords()

        print(f"About {_before - len(self.classes)} instances has been deleted by the ban list 'banned words'")

    def selectClasses(self, selected):
    
        for pos in range(len(self.classes)-1, -1, -1):
            if self.classes[pos] not in selected:
                self.classes.pop(pos)
                self.videoName.pop(pos)
                self.data.pop(pos)


    def saveData(self, indexOrder,save_path, train=True):

        #reorder data
        class_tmp = [self.classes[pos] for pos in indexOrder]
        videoName_tmp = [self.videoName[pos] for pos in indexOrder]
        data_tmp = [self.data[pos] for pos in indexOrder]
        labels_tmp = [self.labels[pos] for pos in indexOrder]

        counter = Counter(class_tmp)
        #print(counter)
        print("counter:",len(counter))
        #print(set(class_tmp))

        if train:
            print("Train:", len(indexOrder))
            path = f"{save_path}-Train.hdf5"
        else:
            print("Val:", len(indexOrder))
            path = f"{save_path}-Val.hdf5"

        # Save H5 
        h5_file = h5py.File(path, 'w')

        for pos, (c, v, d, l) in enumerate(zip(class_tmp, videoName_tmp, data_tmp, labels_tmp)):
            grupo_name = f"{pos}"
            h5_file.create_group(grupo_name)
            h5_file[grupo_name]['video_name'] = v # video name (str)
            h5_file[grupo_name]['label'] = c # classes (str)
            h5_file[grupo_name]['data'] = d # data (Matrix)
            h5_file[grupo_name]['class_number'] = l #label (int)

        h5_file.close()

    def splitDataset(self,n_folds=5,random_state=42,use_split=False):
        df_words = pd.read_csv(f"./incrementalList.csv",encoding='utf-8', header=None)
        print(df_words[0])
        words = list(df_words[0])
        print(len(words),len(words),len(words),len(words),len(words))

        print('#'*40)
    
        # Filter the data to have selected instances
        self.selectClasses(words)

        self.limitIntancesPerClass()

        # generate classes number to use it in stratified option
        self.generate_meaning_dict(df_words.to_dict()[0])

        # split the data into Train and Val (but use list position as X to reorder)
        x_pos = range(len(self.labels))
        n_unique_classes = len(set((self.classes)))
        print("Number of classes:",  n_unique_classes)

        # set the path
        save_path = os.path.normpath(f"split/{self.output_path.split(os.sep)[1]}")
        save_path = save_path.replace('$',str(n_unique_classes))
        save_path_base = save_path.split('.')[0]

        self.labels = np.array(self.labels)
        x_pos = np.array(x_pos)
        if use_split:

            #from sklearn.model_selection import StratifiedShuffleSplit
            from sklearn.model_selection import StratifiedKFold


            #sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.2, random_state=random_state)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

            fold = 0
            #for train_index, val_index in sss.split(x_pos, self.labels):
            for train_index, val_index in skf.split(x_pos, self.labels):
                fold +=1
                #print(x_pos)
                X_train, X_val = x_pos[train_index], x_pos[val_index] 
                print("X_train:",len(X_train),"X_val:",len(X_val))
    
                save_path = save_path_base+"_n_folds_"+str(n_folds)+"_seed_"+str(random_state)+"_klod_"+str(fold)
                self.saveData(X_train,save_path,train=True)
                self.saveData(X_val,save_path, train=False)
                
        else:
            pos_train, pos_val, y_train, y_val = train_test_split(x_pos, self.labels, train_size=0.8 , random_state=1, stratify=self.labels)

            self.saveData(pos_train,save_path_base,train=True)
            self.saveData(pos_val,save_path_base, train=False)

import argparse

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_folds", type=int, default=5, help="")
    parser.add_argument("--random_state", type=int, default=42, help="")
    parser.add_argument("--use_split", type=int, default=0, help="")

    return parser
if __name__ == '__main__':
    #python unifyAndsplitOutputs_byCSV.py --n_folds=3 --random_state=123 --use_split=1
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    kpModel = "mediapipe"
    datasets = ["PUCP_PSL_DGI305", "AEC"] #["AEC", "PUCP_PSL_DGI156", "PUCP_PSL_DGI305", "WLASL", "AUTSL"]

    dataset_out_name = [dataset if len(dataset)<6 else dataset[-6:] for dataset in datasets]
    dataset_out_name = '-'.join(dataset_out_name)

    print(f"procesing {datasets} - using {kpModel} ...")

    output_path = f"output/{dataset_out_name}--$--incremental--{kpModel}.hdf5"

    dataReader = DataReader(datasets, kpModel, output_path)
    dataReader.fixClasses()
    dataReader.splitDataset(n_folds=args.n_folds,random_state=args.random_state,use_split=bool(args.use_split))
    #splitDataset(path)

