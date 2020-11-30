% Exercices session 1 ANN

% Ex1 : 

%% 1. Perceptron 

net = newp(P,T,TF,LF);
net = init(net);

% net.IW{1,1} Returns the weights of the neuron(s) in the first layer
% net.b{1,1} Returns the bias of the neuron(s) in the first layer
% net.IW{1,1} = rand(1,2); Assigns random weights in [0,1]
% net.b{1,1} = rands(1); Assigns random bias in [-1,1]

[net,tr_descr] = train(net,P,T);
net.trainParam.epochs = 20;
sim(net,Pnew)

% DEMOS : 
% nnd4db decision boundary (2d input)
% nnd4pr perceptron learning rule (2d input)
% demop1 classification with 2d input perceptron
% demop4 classification with outlier (2d input)
% demop5 classification with outlier using normalized perceptron learning rule (2d input)
% demop6 linearly non-separable input vectors (2d input)

% Functions and commands : 

% newp(P,T,TF,LF) Creates a perceptron with the right number of neurons, based on input values P,
% target vector T, and with transfer function TF and learning function LF.
% init(net) Initializes the weights and biases of the perceptron.
% adapt(net,P,T) Trains the network using inputs P, targets T and some online learning algorithm.
% train(net,P,T) Trains the network using inputs P, targets T and some batch learning algorithm.
% sim(net,Ptest) Simulates the perceptron using inputs Ptest.
% learnp, learnpn Perceptron and normalized perceptron learning rules
% hardlim Transfer function

%% 2. MLP and backprop

net = feedforwardnet(numN,trainAlg);
net = train(net,P,T);
sim(net,P); % OR Y = net(P);

% Training algorithms : 

% traingd gradient descent
% traingda gradient descent with adaptive learning rate
% traincgf Fletcher-Reeves conjugate gradient algorithm
% traincgp Polak-Ribiere conjugate gradient algorithm
% trainbfg BFGS quasi Newton algorithm (quasi Newton)
% trainlm Levenberg-Marquardt algorithm (adaptive mixture of Newton and steepest descent algorithms)

% postreg : 

a=sim(net,P);
[m,b,r]=postreg(a,T);

% DEMOS :

% nnd11nf network function
% nnd11bc backpropagation calculation
% nnd11fa function approximation
% nnd12sd1 steepest descent backpropagation
% nnd12sd2 steepest descent backpropagation with various learning rates
% nnd12mo steepest descent with momentum
% nnd12vl steepest descent with variable learning rate
% nnd12cg conjugate gradient backpropagation
% nnd9mc comparison between steepest descent and conjugate gradient


%% 2.1

x = 0:0.01:3*pi; % 0.05
y = sin(x.*x);

figure(21)
plot(x,y)

hiddenSizes = 10;
trainFcn = 'trainlm';
myNet = feedforwardnet(hiddenSizes,trainFcn);
net = train(myNet,y,x);
view(myNet)
out = myNet(x);
perf = perform(myNet,out,x)

%% 2.2 personnal regression

Tnew = (7*T1 + 7*T2 + 6*T3 + 3*T4 + 3*T5)/(7 + 7 + 6 + 3 + 3);
data = [X1 X2 Tnew];
k = 3000;
Sample = datasample(data,k,'Replace',false);
%%
TrainData = Sample(1:1000,:);
ValidData = Sample(1001:2000,:);
TestData = Sample(2001:3000,:);

% scatter3(TrainData(:,1),TrainData(:,2),TrainData(:,3))

% x=data(:,1);
% y=data(:,2);
% z=data(:,3);

x=TrainData(:,1);
y=TrainData(:,2);
z=TrainData(:,3);


SamplePerDim=500;
X=linspace(min(x),max(x),SamplePerDim);
Y=linspace(min(y),max(y),SamplePerDim);
[X,Y]=ndgrid(X,Y);
F=scatteredInterpolant(x,y,z,'linear','none');
Z=F(X,Y);
figure(21),clf(21)
surf(X,Y,Z,'EdgeColor','none')
hold on
scatter3(TrainData(:,1),TrainData(:,2),TrainData(:,3))
hold off
%view(-160,80)

% ValidData = datasample(data,k)
% TestData = datasample(data,k)

%%

numLayers = [10 10 10 10 10];
net1 = feedforwardnet(numLayers,'trainlm');
net1.trainParam.epochs=1000;

p = con2seq([x' ; y']); t = con2seq(z'); % needed to avoid split of train/test/valid

%%
net1=train(net1,p,t); % p t = x, y
a11=sim(net1,p);
%%


SamplePerDim=500;
X=linspace(min(x),max(x),SamplePerDim);
Y=linspace(min(y),max(y),SamplePerDim);
[X,Y]=ndgrid(X,Y);
F=scatteredInterpolant(x,y,cell2mat(a11)','linear','none');
Zapprox=F(X,Y);

figure(22),clf(22)
surf(X,Y,Zapprox,'EdgeColor','none')
hold on
scatter3(x,y,z)
hold off

% figure(22)
% plot(x,z,'bx',x,cell2mat(a11),'r');
% title('1000 epochs');
% legend('target','traingd','Location','north');

%% Test set perf


xTest = TestData(:,1);
yTest = TestData(:,2);
zTest = TestData(:,3);

p = con2seq([xTest' ; yTest']);
testOut = sim(net1,p);

SamplePerDim=500;
Xtest=linspace(min(xTest),max(xTest),SamplePerDim);
Ytest=linspace(min(yTest),max(yTest),SamplePerDim);
[Xtest,Ytest]=ndgrid(Xtest,Ytest);
F=scatteredInterpolant(xTest,yTest,cell2mat(testOut)','linear','none');
Ztest=F(Xtest,Ytest);

figure(24),clf(24)
surf(Xtest,Ytest,Ztest,'EdgeColor','none')
hold on
scatter3(xTest,yTest,zTest)
hold off
%% 3. Bayesian Inference (for Net hyperparam)


