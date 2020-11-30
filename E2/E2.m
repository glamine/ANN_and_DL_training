% Exercices session 2 ANN



% stable points being the vectors from T.

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);

% Single step iteration
Y = net([],[],Ai);
% Multiple step iteration
Y = net({num_step},{},Ai);

% DEMOS :

% demohop1 A two neuron Hopfield network
% demohop2 A Hopfield network with unstable equilibrium
% demohop3 A three neuron Hopfield network
% demohop4 Spurious stable points

%% 1. 4 Hopdigit

% noise represents the level of noise that will corrupt the digits and is a number between 0 and 1
% numiter is the number of iterations the Hopfield network (having as input the noisy digits) will run.

noise = 1;
numiter = 5;
hopdigit_v2(noise,numiter)

%% 2 Time series data

lasertrain = load('Files\lasertrain.dat');
laserpred = load('Files\laserpred.dat');

%% 2.2 Neural network approach

lag = 10; % paam 1
numTimeStepsTrain = 1000;
numTimeStepsTest = 100;

dataTrain = lasertrain;
dataTest = laserpred;
dataFull = [lasertrain ; laserpred];

mu = mean(dataTrain);
sig = std(dataTrain); %standard deviation

dataTrainStandardized = (dataTrain - mu) / sig;

[TrainData, TrainTarget] = getTimeSeriesTrainData(dataTrainStandardized, lag);

% network definition

numLayers = 50; %param 2
net1=feedforwardnet(numLayers,'trainlm');
net1.trainParam.epochs=1000;

p = con2seq(TrainData); t = con2seq(TrainTarget); 

%net1=train(net1,p,t); % p t = x, y
net1=train(net1,TrainData,TrainTarget); % p t = x, y
%a11=sim(net1,p);



dataTestStandardized = (dataTest - mu) / sig;
dataFullStandardized = (dataFull - mu) / sig;

dataTestStandardized = [dataTrainStandardized(end - lag + 1 : end) ; dataTestStandardized];

[TestData, TestTarget] = getTimeSeriesTrainData(dataTestStandardized, lag);

pTest = con2seq(TestData); %t = con2seq(TrainTarget); 

%%  Prediction with update on observations :

YPred = sim(net1,pTest);
YPred = cell2mat(YPred);
YPred = sig*YPred' + mu;

YTest = dataTest(1:end);
rmse = sqrt(mean((YPred - YTest).^2));

figure
plot(dataTrain(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,YPred,'.-')
hold off
xlabel("Discrete time k")
ylabel("laser value")
title("Forecast with Updates")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("laser value")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("discrete time k")
ylabel("Error")
title("RMSE = " + rmse)

%% Prediction with no update :

pInit = con2seq(dataTrainStandardized(end - lag + 1 : end));
YPred = [];
YPred(1) = cell2mat(sim(net1,pInit));

for i = 2:numTimeStepsTest
    State = [dataTrainStandardized(end - lag + 1 : end); YPred'];
    pCurrent = con2seq(State(end - lag + 1 : end,:));
    YPred(:,i) = cell2mat(sim(net1,pCurrent));
end

%YPred = cell2mat(YPred);

YPred = sig*YPred' + mu;

YTest = dataTest(1:end);
rmse = sqrt(mean((YPred - YTest).^2));

figure
plot(dataTrain(1:end))
hold on
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,YPred,'.-')
hold off
xlabel("Discrete time k")
ylabel("laser value")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("laser value")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("discrete time k")
ylabel("Error")
title("RMSE = " + rmse)

%% 2.3 LSTM

data = [lasertrain' laserpred'];

% figure
% plot(lasertrain)
% xlabel("Discrete time k")
% ylabel("laser value")
% title("Laser Santa Fe train")
% 
% figure
% plot(laserpred)
% xlabel("Discrete time k")
% ylabel("laser value")
% title("Laser Santa Fe test")

% standardize data :

numTimeStepsTrain = 1000; % 999 ?

dataTrain = lasertrain';
dataTest = laserpred';

mu = mean(dataTrain);
sig = std(dataTrain); %standard deviation

dataTrainStandardized = (dataTrain - mu) / sig;

% Predictors and responses

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% Define LSTM

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 300;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ... %250
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Training

net = trainNetwork(XTrain,YTrain,layers,options);

% Forecast future time steps

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Discrete time k")
ylabel("laser value")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("laser value")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("discrete time k")
ylabel("Error")
title("RMSE = " + rmse)

% Update Network State with Observed Values

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2));

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("laser value")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("discrete time k")
ylabel("Error")
title("RMSE = " + rmse)