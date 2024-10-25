import torch
import pandas as pd
from utils import LSTM, parameters, preparedata, eval, plot, train, rmse, r2
import torch.nn as nn
import numpy as np
import os

if __name__ == '__main__':
    if not os.path.exists('Results'):
        os.makedirs('Results')

    if not os.path.exists('Trained_Models/normal_intra'):
        os.makedirs('Trained_Models/normal_intra')   
    params = parameters()
    device = params.device
    #df = pd.DataFrame(params.list_of_excersices)
    #print("params.list_of_exercises: ",df)
    best_model = None
    best_r2 = float(0.0)  # 初始化最好的R2值
    for randSeed in range(10): # runing 10 times with different seed values
        params.randomseed = randSeed
        RANDOM_SEED = params.randomseed
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        for hand in range(1):  # 0: for left hand 1: for right hand

            # Intra-subject Cross Validation
            for subjectID in range(1):#select subject here
                data_trial = []
                for trialID in range(5):
                    data_trial.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params))
                for trialID in range(5):
                    model = LSTM(input_size=params.number_of_input, hidden_layer_size=params.number_of_hidden_layer,
                                 output_size=params.number_of_output, lstm_layer=params.lstm_layer)
                    model = model.to(device)
                    loss_function = nn.SmoothL1Loss(beta=0.5) #一個兼具L1與L2 loss function 優勢的版本
                    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

                    RMSE_list = []
                    R2_list = []
                    init_flag = 0 #標記訓練集還未放入資料
                    #把一組資料獨立出來做為validation set，剩下4組資料做為training data, trialID是指被拿來當validation的那組
                    for trainingID in range(5):
                        if trainingID != trialID:
                            if init_flag == 0:
                                training_data = data_trial[trainingID]
                                init_flag = 1
                            else:
                                training_data = np.concatenate((training_data, data_trial[trainingID]), axis=0)
                    ######################

                    print("Training intra-subject CV model_" + str(trialID) + "_subject_" + str(subjectID) + "_hand_" + str(
                        hand)+ "_randSeed_" + str(randSeed))
                    model = train(training_data, model, device, params, optimizer, loss_function, data_trial[trialID],
                                  params.patience)
                    actual, predicted = eval(data_trial[trialID], model, device,
                                             "intra-subject CV model_" + str(trialID) + "_subject_" + str(
                                                 subjectID) + "_hand_" + str(hand) + "_randSeed_" + str(randSeed))
                    if params.plot == True:
                        plot(actual, predicted, params, 'scatter',
                             "Results/Subject" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                        plot(actual, predicted, params, 'time', "Results/Subject" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                    
                    current_rmse = rmse(actual, predicted)
                    current_r2 = r2(actual, predicted)
                    RMSE_list.append(current_rmse)
                    R2_list.append(current_r2)

                    # results
                    df_r2 = pd.DataFrame(R2_list)
                    df_r2.to_csv("Results/r2_intra_subject" + str(subjectID) + "_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")
                    df_r2.describe().to_csv(
                        "Results/r2_intra_subject" + str(subjectID) + "_summary_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")

                    df_rmse = pd.DataFrame(RMSE_list)
                    df_rmse.to_csv("Results/rmse_intra_subject" + str(subjectID) + "_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")
                    df_rmse.describe().to_csv(
                        "Results/rmse_intra_subject" + str(subjectID) + "_summary_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")

                    if current_r2 > best_r2:
                        best_r2 = current_r2
                        best_model = model
                        best_model_info = f"model_{trialID}_subject_{subjectID}_hand_{hand}_randSeed_{randSeed}"

    if best_model is not None:
        # 儲存整個模型
        torch.save(best_model, f"Trained_Models/normal_intra/best_model_{best_model_info}.pth")
        print(f"Best model saved as Trained_Models/normal_intra/best_model_{best_model_info}.pth with R2: {best_r2}")
