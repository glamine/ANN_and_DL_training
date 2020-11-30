% Exercice session 3 ANN

%% 1.2.1.0. Load data
load choles_all;

load('Files\threes.mat','-ascii');

myNoise = randn(50,500); % 500 data points of DIM 50

data1 = threes';%p;%myNoise;%



%% 1.2.1.1. zero-mean data
dim = 2;% long de colonnes (variables) 
myMean = mean(data1,dim);
data = data1 - myMean;

i = 10;

%colormap('gray')
% figure(1);
% imagesc(reshape(threes(i,:),16,16),[0,1])
% figure(2);
% imagesc(reshape(myMean,16,16),[0,1])

%% 1.2.1.2. Dim covariance

kinit = 1;
kmax = 256;%256;%21;%50%
rmse = zeros(kmax,1);
rmse = zeros(kmax,1);

    for k = kinit:kmax
    % calculate eigenvectors (loadings) W, and eigenvalues of the covariance matrix
        %[W, EvalueMatrix] = eig(cov(data'));
        %k = 21;
        [v,d] = eigs(cov(data'),k);
        %Evalues = diag(EvalueMatrix);
        Evalues = diag(d);
        
        if k == kmax
            SumEigen = cumsum(Evalues);
            figure(17);
            plot(Evalues);
            title('Eigenvalues of the threes dataset')
            ylabel('Value')
            xlabel('index of the eigenvalue')
        end

    % order by largest eigenvalue
        %Evalues = Evalues(end:-1:1);
        %W = W(:,end:-1:1); W=W';

        Et = transpose(v);

    % generate PCA component space (PCA scores)
        %pc = W * data;
        reducedData = Et * data; %(data + myMean); % is it good? mean ?
        recoveredData = v * reducedData + myMean; % Or that way ?

    % plot PCA space of the first two PCs: PC1 and PC2
        id = 20 + kmax-kinit;
%         figure;
%         recovInter = transpose(recoveredData);
%         imagesc(reshape(recovInter(10,:),16,16),[0,1])
        %plot(pc(1,:),pc(2,:),'.')
        %plot(reducedData(1,:),reducedData(2,:),'.')

    % Evaluate error :

        rmse(k) = sqrt(mean((data1 - recoveredData).^2,'All'));
    end

 %%
 
figure(14)
plot(rmse,'.-')
hold on 
plot(SumEigen*0.01,'*-')
xlabel("Number of eigenvectors used k")
ylabel("Value [RMS error - 0.01 * eigenvalue]")
%title("RMS error in function of dimension reduction [threes dataset]")
title("Comparison RMS error to cumulative sum of eigenvalues")
hold off

figure(15)
plot(Evalues,'r*');
title('Eigenvalues of the threes dataset')
ylabel('Value')
xlabel('index of the eigenvalue')

%% Tools : ? not able to make it work

    i = 1;
    rmse1 = zeros(501,1);
    for maxfrac = 0:0.001:0.5
        [mappedData,PS1] = mapstd(data1); % x1_again = mapstd('reverse',y1,PS)
        %maxfrac = 0.001;
        [reducedData,PS2] = processpca(mappedData,maxfrac); % X = processpca('reverse',Y,PS)

        recoveredData = processpca('reverse',reducedData,PS2);
        p_approx = mapstd('reverse',recoveredData,PS1);
        rmse1(i) = sqrt(mean((data1 - recoveredData).^2,'All'));
        i = i+1;
    end
    
% figure(16)
% plot(rmse1,'.-')
% xlabel("MaxFrac value")
% ylabel("RMS error")
% title("RMS error in function of maxfrac factor")

%% 1.2.2 PCA for handwritten images digits

% load('Files\threes.mat','-ascii')
% 
% i = 1;
% figure
% colormap('gray')
% imagesc(reshape(threes(i,:),16,16),[0,1]);
% 
% data1 = threes';
% 
% dim = 2;% long de colonnes (variables) 
% myMean = mean(data1,dim);
% 
% figure
% colormap('gray')
% imagesc(reshape(myMean',16,16),[0,1]);
% data = data1 - myMean;

%% 
