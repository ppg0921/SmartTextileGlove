import torch
import pandas as pd
from utils import LSTM, parameters, preparedata, eval, plot, train, rmse, r2
import torch.nn as nn
import numpy as np
import os

if __name__ == '__main__':
    if not os.path.exists('Results'):
        os.makedirs('Results')

    if not os.path.exists('Trained_Models/normal_inter'):
        os.makedirs('Trained_Models/normal_inter') 
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

            # Inter-subject Cross Validation
            user_data = []
            for subjectID in range(5):
                for trialID in range(1):# 5個subject各拿一次trial的資料來放入user_data
                    if trialID == 0:
                        data_trial = preparedata(params.list_of_excersices[hand][subjectID][trialID], params)
                    else:
                        data_trial = np.concatenate(
                            (data_trial, preparedata(params.list_of_excersices[hand][subjectID][trialID], params)), axis=0)
                user_data.append(data_trial)

            RMSE_list = []
            R2_list = []

            for subjectID in range(5):
                model = LSTM(input_size=params.number_of_input, hidden_layer_size=params.number_of_hidden_layer,
                             output_size=params.number_of_output, lstm_layer=params.lstm_layer)
                model = model.to(device)
                loss_function = nn.SmoothL1Loss(beta=0.5)
                optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

                init_flag = 0
                for trainingID in range(5): # 5個subject中找一個出來當validation set
                    if trainingID != subjectID:
                        if init_flag == 0:
                            training_data = user_data[trainingID]
                            init_flag = 1
                        else:
                            training_data = np.concatenate((training_data, user_data[trainingID]), axis=0)
                print("Training inter-subject CV model subject_" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                model = train(training_data, model, device, params, optimizer, loss_function, user_data[subjectID],
                              params.patience)
                actual, predicted = eval(user_data[subjectID], model, device,
                                         "inter-subject CV model_" + "subject_" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                if params.plot == True:
                    plot(actual, predicted, params, 'scatter', "Results/Subject" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                    plot(actual, predicted, params, 'time', "Results/Subject" + str(subjectID) + "_hand_" + str(hand)+ "_randSeed_" + str(randSeed))
                
                
                current_rmse = rmse(actual, predicted)
                current_r2 = r2(actual, predicted)
                RMSE_list.append(current_rmse)
                R2_list.append(current_r2)

                # Saving inter-subject cross validation resutls
                df_r2 = pd.DataFrame(R2_list)
                df_r2.to_csv("Results/r2_inter_subject_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")
                df_r2.describe().to_csv("Results/r2_inter_subject_summary_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")

                df_rmse = pd.DataFrame(RMSE_list)
                df_rmse.to_csv("Results/rmse_inter_subject_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")
                df_rmse.describe().to_csv("Results/rmse_inter_subject_summary_hand_" + str(hand) + "_randSeed_" + str(randSeed) + ".csv")
                
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_model = model
                    best_model_info = f"model_inter_subject_{subjectID}_as_valid_hand_{hand}_randSeed_{randSeed}"

    if best_model is not None:
        # 儲存整個模型
        torch.save(best_model, f"Trained_Models/normal_inter/best_model_{best_model_info}.pth")
        print(f"Best model saved as Trained_Models/normal_inter/best_model_{best_model_info}.pth with R2: {best_r2}")
