clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt

% traingd gradient descent
% traingda gradient descent with adaptive learning rate
% traincgf Fletcher-Reeves conjugate gradient algorithm
% traincgp Polak-Ribiere conjugate gradient algorithm
% trainbfg BFGS quasi Newton algorithm (quasi Newton)
% trainlm Levenberg-Marquardt algorithm (adaptive mixture of Newton and steepest descent algorithms)
%%%%%%%%%%%

%% generation of examples and targets
x=0:0.01:3*pi; y = sin(x.^2) + 0.5*randn(1,943); %0.05 incr %0.1*randn



figure(1)
plot(x,y)
%%
p=con2seq(x); t=con2seq(y); % convert the data to a useful format
%p=x; t=y;

%%
%creation of networks
net1=feedforwardnet(50,'trainbr');
net2=feedforwardnet(50,'trainlm');
net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training 
net2.trainParam.epochs=1;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
a11=sim(net1,p); a21=sim(net2,p);  % simulate the networks with the input vector p

net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net1=train(net1,p,t);
net2=train(net2,p,t);
a12=sim(net1,p); a22=sim(net2,p);

net1.trainParam.epochs=985;%5000;%
net2.trainParam.epochs=985;%5000;%
net1=train(net1,p,t);
net2=train(net2,p,t);
a13=sim(net1,p); a23=sim(net2,p);

%%
%plots
% figure(2)
% subplot(3,3,1);
% plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
% title('1 epoch');
% legend('target','traingd','trainbr','Location','north');
% subplot(3,3,2);
% postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(a21),y);
% %
% subplot(3,3,4);
% plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
% title('15 epochs');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,3,5);
% postregm(cell2mat(a12),y);
% subplot(3,3,6);
% postregm(cell2mat(a22),y);
% %
% subplot(3,3,7);
% plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
% title('1000 epochs');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(a13),y);
% subplot(3,3,9);
% postregm(cell2mat(a23),y);

figure(3)
plot(x,y,'bx',x,cell2mat(a11),'k:',x,cell2mat(a21),'k--','LineWidth',2); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','traingd','trainlm','Location','SouthWest');

figure(4)
plot(x,y,'bx',x,cell2mat(a12),'k:',x,cell2mat(a22),'k--','LineWidth',2);
title('15 epochs');
legend('target','traingd','trainlm','Location','SouthWest');

figure(5)
plot(x,y,'bx',x,cell2mat(a13),'k:',x,cell2mat(a23),'k--','LineWidth',2);
title('1000 epochs');
legend('target','traingd','trainlm','Location','SouthWest');

figure(6)
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
figure(7)
postregm(cell2mat(a21),y);

figure(8)
postregm(cell2mat(a12),y); % perform a linear regression analysis and plot the result
figure(9)
postregm(cell2mat(a22),y);

figure(10)
postregm(cell2mat(a13),y); % perform a linear regression analysis and plot the result
figure(11)
postregm(cell2mat(a23),y);

%% 

%%
%plots
% figure(2)
% subplot(3,3,1);
% plot(x,y,'bx',x,(a11),'r',x,(a21),'g'); % plot the sine function and the output of the networks
% title('1 epoch');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,3,2);
% postregm((a11),y); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm((a21),y);
% %
% subplot(3,3,4);
% plot(x,y,'bx',x,(a12),'r',x,(a22),'g');
% title('15 epochs');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,3,5);
% postregm((a12),y);
% subplot(3,3,6);
% postregm((a22),y);
% %
% subplot(3,3,7);
% plot(x,y,'bx',x,(a13),'r',x,(a23),'g');
% title('1000 epochs');
% legend('target','trainlm','trainbr','Location','north');
% subplot(3,3,8);
% postregm((a13),y);
% subplot(3,3,9);
% postregm((a23),y);
% 
% figure(3)
% plot(x,y,'bx',x,(a11),'k:',x,(a21),'k--','LineWidth',2); % plot the sine function and the output of the networks
% title('1 epoch');
% legend('target','traingd','traingda','Location','north');
% 
% figure(4)
% plot(x,y,'bx',x,(a12),'k:',x,(a22),'k--','LineWidth',2);
% title('15 epochs');
% legend('target','traingd','traingda','Location','SouthWest');
% 
% figure(5)
% plot(x,y,'bx',x,(a13),'k:',x,(a23),'k--','LineWidth',2);
% title('1000 epochs');
% legend('target','traingd','traingda','Location','north');
% 
% figure(6)
% postregm((a11),y); % perform a linear regression analysis and plot the result
% figure(7)
% postregm((a21),y);
% 
% figure(8)
% postregm((a12),y); % perform a linear regression analysis and plot the result
% figure(9)
% postregm((a22),y);
% 
% figure(10)
% postregm((a13),y); % perform a linear regression analysis and plot the result
% figure(11)
% postregm((a23),y);