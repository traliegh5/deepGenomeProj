from comet_ml import Experiment
import numpy as np
import tensorflow as tf
# import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
import scipy.stats 
import time

experiment = Experiment(
    project_name='midterm',
    
)
# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_data = None

# Load data
train_data = np.load('/home/tjenkin3/course/Genomics/cs1850-midterm/data/cs1850-spring2021-midterm/train.npz')
eval_data = np.load('/home/tjenkin3/course/Genomics/cs1850-midterm/data/cs1850-spring2021-midterm/eval.npz')

def get_data():
    # Combine Train Data to use information from all cells
    train_inputs = [] # Input histone mark data
    train_outputs = [] # Correct expression value
    for cell in train_cells:
        cell_data = train_data[cell]
        hm_data = cell_data[:,:,1:6]
        exp_values = cell_data[:,0,6]
        train_inputs.append(hm_data)
        train_outputs.append(exp_values)

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)

    # Prepare Eval inputs in similar way
    eval_inputs = []
    eval_labels=[]
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data = cell_data[:,:,1:6]
        exp_values = cell_data[:,0,6]
        eval_inputs.append(hm_data)
        eval_labels.append(exp_values)

    eval_inputs = np.concatenate(eval_inputs, axis=0)
    eval_labels=np.concatenate(eval_labels,axis=0)
    print(eval_labels[0],eval_inputs[0])

    return train_inputs, train_outputs, eval_inputs,eval_labels


def train(train_x, train_y):
    hyperparams={"filters":50,"kernel_size":10,"pool_size":5,"dense_1":625,"dense_2":125,"ler":5e-4}
    
    experiment.log_parameters(hyperparams)
    ler=5e-4
    optimizer=Adam(lr=hyperparams["ler"])
    model = Sequential()
    model.add(Conv1D(filters=hyperparams["filters"], kernel_size=hyperparams["kernel_size"], activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    
    model.add(MaxPooling1D(pool_size=hyperparams["pool_size"]))
    model.add(Dropout(0.1))
   
   
    model.add(Flatten())
    model.add(Dense(hyperparams["dense_1"],activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(hyperparams["dense_2"],activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    with experiment.train():
        history=model.fit(train_x, train_y, validation_split=0.2, epochs=20, batch_size=500, verbose=1)
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])  # RAISE ERROR
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss']) #RAISE ERROR
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show() print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])  # RAISE ERROR
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss']) #RAISE ERROR
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    return model

def predict(model, test_x,test_y):
    mse = tf.keras.losses.MeanSquaredError()
    with experiment.test():
        score=model.evaluate(x=test_x,y=test_y, batch_size=200, verbose=1)
        print(score)
        experiment.log_metric("mse",score)
        experiment.log_metric("idk",score)
    return score

# def get_csv(test_x, prediction):
#     cell_list = []
#     gene_list = []

#     for cell in eval_cells:
#         cell_data = eval_data[cell]
#         cell_list.extend([cell]*len(cell_data))
#         genes = cell_data[:,0,0].astype('int32')
#         gene_list.extend(genes)

#     id_column = [] # ID is {cell type}_{gene id}
#     for idx in range(len(test_x)):
#         id_column.append(f'{cell_list[idx]}_{gene_list[idx]}')

#     df_data = {'id': id_column, 'expression' : prediction.flatten()}
#     submit_df = pd.DataFrame(data=df_data)

#     submit_df.to_csv('submission-{}.csv'.format(int(time.time())), header=True, index=False, index_label=False)

def main():
    train_x, train_y, test_x,test_y = get_data()

    model = train(train_x, train_y)

    # prediction = predict(model, test_x,test_y)

    #get_csv(test_x, prediction)

if __name__ == '__main__':
    main()