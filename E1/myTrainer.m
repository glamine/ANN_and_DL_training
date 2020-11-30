%% My trainer

load('Files\Data_Problem1_regression.mat') 

Tnew = (7*T1 + 7*T2 + 6*T3 + 3*T4 + 3*T5)/(7 + 7 + 6 + 3 + 3);
data = [X1 X2 Tnew];
k = 3000;

opti = 'trainlm';
numLayers = [25 25]; %50; %[10 10 10 10 10];
iter = 1000;

%% Training



Sample = datasample(data,k,'Replace',false);

TestData = Sample(2001:3000,:); %Test used only ONCE

xTest = TestData(:,1);
yTest = TestData(:,2);
zTest = TestData(:,3);

pTest = con2seq([xTest' ; yTest']);

i = 1;
myIter = [1 5 10 50 100 500 1000];
n = length(myIter);
n1 = 10;%10
rmseValid = zeros(n,2);
rmseTrain = zeros(n,2);
RMSE_valid_it = zeros(n1,1);
RMSE_train_it = zeros(n1,1);

for algID = 1:2
    
    if algID == 1
        opti = 'trainlm';
    else
        opti = 'traingd';
    end
    
    for iter = myIter
        
        for j = 1:n1
            Sample = datasample(data,k,'Replace',false);

            TrainData = Sample(1:1000,:);
            ValidData = Sample(1001:2000,:);

            x = TrainData(:,1);
            y = TrainData(:,2);
            z = TrainData(:,3);

            xValid = ValidData(:,1);
            yValid = ValidData(:,2);
            zValid = ValidData(:,3);

            pValid = con2seq([xValid' ; yValid']);
            p = con2seq([x' ; y']); t = con2seq(z');

             % param number neurons
            net1 = feedforwardnet(numLayers,opti); % algo opti
            net1.trainParam.epochs = iter; % nbre epochs
            net1 = train(net1,p,t); % p t = x, y
            a11 = sim(net1,p);
            RMSE_train_it(j) = sqrt(mean((z - transpose(cell2mat(a11))).^2,1));

            % Validation

            zLatentValid = sim(net1,pValid);

            RMSE_valid_it(j) = sqrt(mean((zValid - transpose(cell2mat(zLatentValid))).^2,1));  
            
        end
        rmseTrain(i,algID) = mean(RMSE_train_it);
        rmseValid(i,algID) = mean(RMSE_valid_it); 
        i = i+1;
    
    end
    
end

%%

figure(1)
plot(log(myIter),log(rmseValid(1:7,1)),'r');
hold on
plot(log(myIter),log(rmseTrain(1:7,1)),'r--');
plot(log(myIter),log(rmseValid(8:14,2)),'g');
plot(log(myIter),log(rmseTrain(8:14,2)),'g--');
hold off

figure(2)
plot((myIter'),rmseValid(1:7,1),'r');
hold on
plot((myIter'),rmseTrain(1:7,1),'r--');
plot((myIter'),rmseValid(8:14,2),'g');
plot((myIter'),rmseTrain(8:14,2),'g--');
hold off

figure(3)
plot(log(myIter'),log(rmseValid(1:7,1) - rmseValid(8:14,2)),'r');
hold on
plot(log(myIter'),log(rmseTrain(1:7,1) - rmseTrain(8:14,2)),'r--');
hold off

figure(4)
loglog((myIter'),rmseValid(1:7,1),'r');
hold on
loglog((myIter'),rmseTrain(1:7,1),'r--');
loglog((myIter'),rmseValid(8:14,2),'g');
loglog((myIter'),rmseTrain(8:14,2),'g--');
title('LM versus GD rms error in function of number of epochs');
legend('trainlm valid','trainlm train','traingd valid','traingd train','Location','SouthWest');
grid on
hold off
%% Testing

% zLatentTest = sim(net1,pTest);
% 
% rmseTest = sqrt(mean((zTest' - zLatent).^2,1));