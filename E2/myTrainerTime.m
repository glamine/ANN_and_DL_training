% My time series trainer :

%% 2 Time series data

lasertrain = load('Files\lasertrain.dat');
laserpred = load('Files\laserpred.dat');

%% 2.2 Neural network approach

lag = 11; % 10 % param 1
numLayers = 40; %50; %param 2

numTimeStepsTrain = 1000;
numTimeStepsTest = 100;

dataTrain = lasertrain;
dataTest = laserpred;
dataFull = [lasertrain ; laserpred];

mu = mean(dataTrain);
sig = std(dataTrain); %standard deviation

dataTrainStandardized = (dataTrain - mu) / sig;
dataTestStandardized = (dataTest - mu) / sig;
dataFullStandardized = (dataFull - mu) / sig;
% dataTestStandardized = [dataTrainStandardized(end - lag + 1 : end) ; dataTestStandardized];
% 
% [TrainData, TrainTarget] = getTimeSeriesTrainData(dataTrainStandardized, lag);
% [TestData, TestTarget] = getTimeSeriesTrainData(dataTestStandardized, lag);

%p = con2seq(TrainData); t = con2seq(TrainTarget);
%pTest = con2seq(TestData); %t = con2seq(TrainTarget);

%%

i = 1;
n = 10;%5
myLag = 10:10:100;%1:2:20;
myNeurons = 10:10:100;
neur = 40;

rmse = zeros(length(myLag),1);
rmse_bis = zeros(length(myLag),1);
rmse_it = zeros(n,1);
rmse_it_bis = zeros(n,1);

for lag = myLag
%for neur = myNeurons
    
    dataTestStandardized_it = [];
    dataTestStandardized_it = [dataTrainStandardized(end - lag + 1 : end) ; dataTestStandardized];

    [TrainData, TrainTarget] = getTimeSeriesTrainData(dataTrainStandardized, lag);
    [TestData, TestTarget] = getTimeSeriesTrainData(dataTestStandardized_it, lag);
    pInit = con2seq(dataTrainStandardized(end - lag + 1 : end));
    pTest = con2seq(TestData); %t = con2seq(TrainTarget);
    
    for j = 1:n

        % network definition
        %net1=feedforwardnet(numLayers,'trainlm');
        net1=feedforwardnet(neur,'trainlm');
        net1.trainParam.epochs=1000; 

        %net1=train(net1,p,t); % p t = x, y
        net1 = train(net1,TrainData,TrainTarget); % p t = x, y

        YPred = sim(net1,pTest);
        YPred = cell2mat(YPred);
        YPred = sig*YPred' + mu;

        YTest = dataTest(1:end);
        rmse_it(j) = sqrt(mean((YPred - YTest).^2));
        
        % noUpdate
        
        YPredBis = [];
        YPredBis(1) = cell2mat(sim(net1,pInit));

        for k = 2:numTimeStepsTest
            State = [dataTrainStandardized(end - lag + 1 : end); YPredBis'];
            pCurrent = con2seq(State(end - lag + 1 : end,:));
            YPredBis(:,k) = cell2mat(sim(net1,pCurrent));
        end

        YPredBis = sig*YPredBis' + mu;

        YTest = dataTest(1:end);
        rmse_it_bis(j) = sqrt(mean((YPredBis - YTest).^2));
        
    end
    rmse(i) = mean(rmse_it);
    rmse_bis(i) = mean(rmse_it_bis);
    i = i+1;
end

%%

figure
plot(myLag,rmse,'r')
hold on
plot(myLag,rmse_bis,'b') %noUpdate
title('RMS error in function of lag');
legend('update','no update');
xlabel('lag [nbr of neurons]')
ylabel('rms error [/]')
%%
figure
loglog(myLag,rmse,'r')
hold on
loglog(myLag,rmse_bis,'b') %noUpdate
title('RMS error in function of lag');
legend('update','no update');
xlabel('lag [nbr of neurons]')
ylabel('rms error [/]')
grid on

%%

% figure
% plot(myNeurons,rmse,'r')
% hold on
% plot(myNeurons,rmse_bis,'b') %noUpdate
% title('RMS error in function of number of neurons in layer');
% legend('update','no update');
% xlabel('hidden layer [nbr of neurons]')
% ylabel('rms error [/]')
% %%
% figure
% loglog(myNeurons,rmse,'r')
% hold on
% loglog(myNeurons,rmse_bis,'b') %noUpdate
% title('RMS error in function of the number of neurons in layer');
% legend('update','no update');
% xlabel('hidden layer [nbr of neurons]')
% ylabel('rms error [/]')
% grid on