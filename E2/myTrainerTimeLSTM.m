% My trainer LSTM


lasertrain = load('Files\lasertrain.dat');
laserpred = load('Files\laserpred.dat');
data = [lasertrain' laserpred'];

numTimeStepsTrain = 1000; % 999 ?

dataTrain = lasertrain';
dataTest = laserpred';

mu = mean(dataTrain);
sig = std(dataTrain); %standard deviation

dataTrainStandardized = (dataTrain - mu) / sig;
dataTestStandardized = (dataTest - mu) / sig;

% Predictors and responses

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
XTest = dataTestStandardized(1:end-1);

% Define LSTM

numFeatures = 1;
numResponses = 1;

%% Iterations
numHiddenUnits = 300;%100:50:300;%200;
n = 3;
i = 1;
rmse = zeros(length(numHiddenUnits),1);
rmse_bis = zeros(length(numHiddenUnits),1);
rmse_it = zeros(n,1);
rmse_update_it = zeros(n,1);

for neur = numHiddenUnits

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(neur)
        fullyConnectedLayer(numResponses)
        regressionLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs',150, ... %250 % 500
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    for j = 1:n

        net = trainNetwork(XTrain,YTrain,layers,options);

        net = predictAndUpdateState(net,XTrain);
        [net,YPred] = predictAndUpdateState(net,YTrain(end));

        numTimeStepsTest = numel(XTest);
        for k = 2:numTimeStepsTest
            [net,YPred(:,k)] = predictAndUpdateState(net,YPred(:,k-1),'ExecutionEnvironment','cpu');
        end

        YPred = sig*YPred + mu;

        YTest = dataTest(2:end);
        rmse_it(j) = sqrt(mean((YPred-YTest).^2));

        % with update

        net = resetState(net);
        net = predictAndUpdateState(net,XTrain);

        YPred = [];
        numTimeStepsTest = numel(XTest);
        for k = 1:numTimeStepsTest
            [net,YPred(:,k)] = predictAndUpdateState(net,XTest(:,k),'ExecutionEnvironment','cpu');
        end

        YPred = sig*YPred + mu;

        rmse_update_it(j) = sqrt(mean((YPred-YTest).^2));
        
    end
    
    rmse(i) = mean(rmse_it);
    rmse_bis(i) = mean(rmse_update_it);
    i = i+1;
end

%%

figure
plot(numHiddenUnits,rmse_bis,'r')
hold on
plot(numHiddenUnits,rmse,'b') %noUpdate
title('RMS error in function of LSTM hidden units');
legend('update','no update');
xlabel('hidden units [nbr of neurons]')
ylabel('rms error [/]')
hold off

%%
figure
loglog(numHiddenUnits,rmse_bis,'r')
hold on
loglog(numHiddenUnits,rmse,'b') %noUpdate
title('RMS error in function of LSTM hidden units');
legend('update','no update');
xlabel('hidden units [nbr of neurons]')
ylabel('rms error [/]')
grid on
hold off



