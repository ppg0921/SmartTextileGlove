import torch
import pandas as pd
from utils import LSTM, parameters, preparedata, eval, plot, train, rmse, r2
import torch.nn as nn
import numpy as np
import os

if __name__ == '__main__':
    if not os.path.exists('Results'):
        os.makedirs('Results')

    if not os.path.exists('Trained_Models/augmented_intra'):
        os.makedirs('Trained_Models/augmented_intra')
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
        # Self-supervised pretraining
        for hand in range(1): #select which hand to train
            for subjectID in range(1): #select which subject to train
                data_trial_normal = []
                data_trial_augmented = []
                data_trial_masked = []
                data_trial_scaled = []
                data_trial_noisy = []

                print("Loading datasets")
                for trialID in range(2): #dataset0當訓練集;dataset1當testing set
                    print("Trial"+str(trialID))
                    print("Normal data")
                    data_trial_normal.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params))
                    print("Augmented data")
                    data_trial_augmented.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params, augmented = True))
                    print("Masked data")
                    data_trial_masked.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params, masked = True))
                    print("Scaled data")
                    data_trial_scaled.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params, scaled = True))
                    print("Noisy data")
                    data_trial_noisy.append(preparedata(params.list_of_excersices[hand][subjectID][trialID], params, noisy=True))

                model_normal = LSTM(input_size=params.number_of_input, hidden_layer_size=params.number_of_hidden_layer, output_size=params.number_of_output, lstm_layer=params.lstm_layer)
                model_normal = model_normal.to(device)

                model_augmented = LSTM(input_size=params.number_of_input, hidden_layer_size=params.number_of_hidden_layer, output_size=params.number_of_output, lstm_layer=params.lstm_layer)
                model_augmented = model_augmented.to(device)

                loss_function_normal = nn.SmoothL1Loss(beta=0.5)
                optimizer_normal = torch.optim.Adam(model_normal.parameters(), lr=params.learning_rate)

                loss_function_augmented = nn.SmoothL1Loss(beta=0.5)
                optimizer_augmented = torch.optim.Adam(model_augmented.parameters(), lr=params.learning_rate)

                print("Training normal model")
                model_normal = train(data_trial_normal[0], model_normal, device, params, optimizer_normal, loss_function_normal, data_trial_normal[1], params.patience)

                print("Training augmented model")
                model_augmented = train(data_trial_augmented[0], model_augmented, device, params, optimizer_augmented, loss_function_augmented, data_trial_augmented[1], params.patience)

                print("Fine-tuning augmented model on normal dataset")
                model_augmented = train(data_trial_normal[0], model_augmented, device, params, optimizer_augmented, loss_function_augmented, data_trial_augmented[1], params.patience)
                
                R2_normal = []
                RMSE_normal = []

                #evaluate normal model
                actual, predicted = eval(data_trial_normal[1], model_normal, device, "normal_model_normal_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_normal.append(rmse(actual, predicted))
                R2_normal.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_masked[1], model_normal, device, "normal_model_masked_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_normal.append(rmse(actual, predicted))
                R2_normal.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_scaled[1], model_normal, device, "normal_model_scaled_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_normal.append(rmse(actual, predicted))
                R2_normal.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_noisy[1], model_normal, device, "normal_model_noisy_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_normal.append(rmse(actual, predicted))
                R2_normal.append(r2(actual, predicted))

                R2_augmented = []
                RMSE_augmented = []
                # evaluate augmented model
                actual, predicted = eval(data_trial_normal[1], model_augmented, device, "augmented_model_normal_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_augmented.append(rmse(actual, predicted))
                R2_augmented.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_masked[1], model_augmented, device, "augmented_model_masked_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_augmented.append(rmse(actual, predicted))
                R2_augmented.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_scaled[1], model_augmented, device, "augmented_model_scaled_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_augmented.append(rmse(actual, predicted))
                R2_augmented.append(r2(actual, predicted))

                actual, predicted = eval(data_trial_noisy[1], model_augmented, device, "augmented_model_noisy_data_subject"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed))
                RMSE_augmented.append(rmse(actual, predicted))
                R2_augmented.append(r2(actual, predicted))

                df_r2_normal = pd.DataFrame(R2_normal)
                df_rmse_normal = pd.DataFrame(RMSE_normal)
                df_r2_augmented = pd.DataFrame(R2_augmented)
                df_rmse_augmented = pd.DataFrame(RMSE_augmented)

                df_r2_normal.to_csv("Results/r2_normal_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")
                df_r2_normal.describe().to_csv("Results/r2_normal_summary_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")

                df_rmse_normal.to_csv("Results/rmse_normal_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")
                df_rmse_normal.describe().to_csv("Results/rmse_normal_summary_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")

                df_r2_augmented.to_csv("Results/r2_augmented_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")
                df_r2_augmented.describe().to_csv("Results/r2_augmented_summary_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")

                df_rmse_augmented.to_csv("Results/rmse_augmented_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")
                df_rmse_augmented.describe().to_csv("Results/rmse_augmented_summary_subject_"+str(subjectID)+"_hand_"+str(hand)+"_seed_"+str(randSeed)+".csv")

                