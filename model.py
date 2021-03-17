import numpy as np
import pandas as pd
from comet_ml import Experiment # Must be imported before tensorflow for some reason
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, Dropout, MaxPooling1D, Dense, ReLU, LeakyReLU
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import time

size = 13
rcParams['axes.labelsize'] = size + 2
rcParams['xtick.labelsize'] = size
rcParams["legend.fontsize"] = size
rcParams['ytick.labelsize'] = size
rcParams['axes.titlesize'] = size + 3

############ comet_ml portion ############

exp = Experiment(
  
    api_key="WqRtWjJU3OT8bsCw8rqnonR21", 
    project_name="midterm",
    
   
)

# Create an experiment with your api key
#experiment = Experiment(
#    api_key="Tig1xygDevBpKU65KlLE345np",
#    project_name="general",
#    workspace="bfoulon",
#)

##########################################

# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

train_eval_diff_cells = ['E096', 'E113', 'E084', 'E112', 'E071', 'E087'] # These are in eval but NOT in train

# Load data
train_data = np.load('./train.npz')
eval_data = np.load('./eval.npz')

def get_data():
    # Keys to npzfile of train & eval
    #train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
    #'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

    #eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
    #'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

    # Load data
    #train_data = np.load('data/train.npz')
    #eval_data = np.load('data/eval.npz')

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
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data = cell_data[:,:,1:6]
        eval_inputs.append(hm_data)

    eval_inputs = np.concatenate(eval_inputs, axis=0)

    return train_inputs, train_outputs, eval_inputs
        
def train(train_x, train_y, valid_x, valid_y):
    #print(train_x.shape)
    #print(train_y.shape)
    #print(valid_x.shape)
    #print(valid_y.shape)
    # Best models so far:
    # 1) Baseline with
    # 2) 

    # DeepChrome Parameters
    batch_size = 100
    dropouts = [0.5] # 0.5 is best so far. 0.1 did better on test data though. Will that bring test performance down though?
    #conv_filters = [50] # [20, 50, 100]
    # Having more than one convolution layer doesn't seem to help at all
    conv_filters = [50] # 50 is best so far. Going to 75 didn't improve things that much
    conv_kernels = [10] # [10, 5]
    pool_sizes = [5] # 5
    #hidden_layer_sizes = [1000]
    #hidden_layer_sizes = [960, 320]
    #hidden_layer_sizes = [500, 100]
    #hidden_layer_sizes = [800, 300]
    #hidden_layer_sizes = [625, 125]
    hidden_layer_sizes = [64, 32]
    #hidden_layer_sizes = [1000, 200] # This seems to do a bit better than the [625,125] one. Worth it?
    #hidden_layer_sizes = [625, 375, 125] # 950 didn't improve things. Nor did adding extra layer

    # Can try adding more conv layers too

    # Narratives to talk about
    # 1) Where to place dropout layer

    # Takeaways so far
    # 1) More than one convolution layer is not helpful
    # 2) Two MLP hidden layers = good

    # To try
    # 1) Should you just try to get rid of the dropout layers altogether?

    # Validation MSEs
    # baseline params with 0.1 dropout: [3.098770214474201, 3.0717750911295414, 3.092356529891491, 3.0978253623366356, 3.123860747367144]
    # 

    # Maybe learning rate should be increased??
    l_rate = 5e-4 # DeepChrome used 1e-3, but with normal SGD (I think? Unclear which kind of SGD)
    # 1e-2 doesn't seem that great...
    # 5e-3 didn't help
    #  5e-4 better than 7.5e-4 better than 1e-3
    num_epochs = 20

    params = {"conv_filters": conv_filters,
          "conv_kernels": conv_kernels,
          "pool_sizes": pool_sizes,
          "hidden_layer_sizes": hidden_layer_sizes,
          "dropouts": dropouts,
          "batch_size": batch_size
          }
    #exp.log_parameters(params)
    print(params)

    #filters1 = 32
    #kernel1 = 4 # Tried 4
    #hu1 = 10

    """

    # Validate model using cross validation
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=0)
    cv_losses = []
    count = 1
    lowest_loss = 100
    for train_idx, test_idx in kfold.split(train_x, train_y):
        print('Fold ' + str(count))
        model = tf.keras.Sequential()
        model.add(Conv1D(filters=conv_filters[0], kernel_size=conv_kernels[0], activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(MaxPooling1D(pool_size=pool_sizes[0]))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(hidden_layer_sizes[0], activation='relu'))
        model.add(Dense(hidden_layer_sizes[1], activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=l_rate), metrics=['mean_squared_error'])
        model.fit(train_x[train_idx], train_y[train_idx], batch_size=batch_size, epochs=10, verbose=0)
        MSE, _ = model.evaluate(train_x[test_idx], train_y[test_idx], verbose=0)
        if MSE < lowest_loss:
            best_model = model
            lowest_loss = MSE
        cv_losses.append(MSE)
        count += 1
    
    print('Cross Validation Results')
    print('Losses (MSE):')
    print(cv_losses)
    """


    print('Full model training begins')
    model = Sequential()
    model.add(Conv1D(filters=conv_filters[0], kernel_size=conv_kernels[0], activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(MaxPooling1D(pool_size=pool_sizes[0]))
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Conv1D(filters=conv_filters[1], kernel_size=conv_kernels[1], activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    #model.add(MaxPooling1D(pool_size=pool_sizes[1]))
    model.add(Dropout(dropouts[0]))
    model.add(Flatten())
    model.add(Dense(hidden_layer_sizes[0], activation='relu'))
    #model.add(Dropout(dropouts[1]))
    model.add(Dense(hidden_layer_sizes[1], activation='relu'))
    #model.add(Dense(hidden_layer_sizes[2], activation='relu'))
    model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=l_rate))
    #model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size)
    with experiment.train():
        history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(valid_x, valid_y))
    #"""
    # Plot training and validation loss
#     plt.plot(np.arange(1,num_epochs+1,1),history.history['loss'])
#     plt.plot(np.arange(1,num_epochs+1,1),history.history['val_loss'])
#     plt.title('Model Loss')
#     #plt.title(params)
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='best')
#     #plt.tight_layout()
#     #plt.show()
#     now = datetime.now()
#     time_id = now.strftime("%m.%d-%H%M%p")
#     print(time_id)
#     plt.savefig('graphs/%s.png' % time_id)
#     #"""
#     print(model.summary())

    # Save trained model afterwards

    return model

def get_csv(test_x, prediction):
    cell_list = []
    gene_list = []
    
    for cell in eval_cells:
        cell_data = eval_data[cell]
        cell_list.extend([cell]*len(cell_data))
        genes = cell_data[:,0,0].astype('int32')
        gene_list.extend(genes)
    
    id_column = [] # ID is {cell type}_{gene id}
    for idx in range(len(test_x)):
        id_column.append(f'{cell_list[idx]}_{gene_list[idx]}')
    
    df_data = {'id': id_column, 'expression' : prediction.flatten()}
    submit_df = pd.DataFrame(data=df_data)
    submit_df.to_csv('submission-{}.csv'.format(int(time.time())), header=True, index=False, index_label=False)

def evaluate():
    pass

def print_csv():
    pass

def visualize():
    pass

def main():
    tf.random.set_seed(0)

    Train_x, Train_y, Test_x = get_data()
    #print(train_x.shape) # (800000, 100, 5)
    #print(train_y.shape) # (800000,)
    #print(test_x.shape) # (177032, 100, 5)

    X_train, X_valid, y_train, y_valid = train_test_split(Train_x, Train_y, test_size=0.2)
    #print(X_train.shape)
    #print(X_valid.shape)
    #print(y_train.shape)
    #print(y_valid.shape)

    #model = train(train_x, train_y)
    model = train(X_train, y_train, X_valid, y_valid)
#     pred_y_test = model.predict(Test_x)
#     get_csv(Test_x, pred_y_test)

if __name__ == '__main__':
    main()
