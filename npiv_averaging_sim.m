% NPIV Monte-Carlo simulation.
% The DGP is the dataset provided by Horowitz(2011a)
% I want to illustrate the vulnerability of NPIV in the finite sample setting.
% And plan to show that the model averaging can make better performance under
% finite sample.

data = importdata('Fig2.xlsx');
dataset = data.data;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Population g(X), beta_iv, beta_ols %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_p = 1552;
population = dataset;
Y = population(:,1);
X = population(:,2);
Z = population(:,3);
K = 3;

% Transform X and Z to Unit Interval
minx = min(X);
rangeX = max(X) - minx;
minz = min(Z);
rangeZ = max(Z) - minz;
X = (X - minx) ./ rangeX;
Z = (Z - minz) ./ rangeZ;

% define a Legendre polynomial basis for L2[0,1]

MX = zeros(n_p,K);
for i=1:K
    if i == 1
        MX(:,i) = ones(n_p,1);
    elseif mod(i,2) == 0
        MX(:,i) = sqrt(2)*sin(floor((i)/2)*(X.*2*pi));
    else
        MX(:,i) = sqrt(2)*cos(floor((i)/2)*(X.*2*pi));
    end
end

MZ = zeros(n_p,K);
for i=1:K
    if i == 1
        MZ(:,i) = ones(n_p,1);
    elseif mod(i,2) == 0
        MZ(:,i) = sqrt(2)*sin(floor((i)/2)*(Z.*2*pi));
    else
        MZ(:,i) = sqrt(2)*cos(floor((i)/2)*(Z.*2*pi));
    end
end

G = (MZ'*MX)\(MZ'*Y);   % Fourier coefficients
g_true = MX*G;          % true g(X)

Xmat = [ones(n_p,1),X];
Zmat = [ones(n_p,1),Z];
betaIV0 = (Zmat'*Xmat)\(Zmat'*Y);
Y_IV0 = Xmat*betaIV0;
beta0 = (Xmat'*Xmat)\(Xmat'*Y);
Y_OLS0 = Xmat*beta0;

population = [population,g_true,Y_IV0];     % population matrix

figure(1)
plot(population(:,2),g_true,'.',population(:,2),Y_IV0,'.',population(:,2),Y_OLS0,'.',population(:,2),Y,'.')
legend 'NPIV' 'IV' 'OLS' 'obs'
ylim([-0.2,0.6])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Monte-Carlo Simulation : n = 200 %%%%%%%%%
%%%%% Comparing finite sample performances %%%%%
%%%%% of NPIV and IV %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng("default")
rep = 100;    % number of repetition
n = 200;    % number of obs.

K = 2;  % truncation parameter - should be dependent on the data. For now I just fix it.

beta_iv_dist = zeros(rep,2);
G_hat_dist = zeros(rep,K);
rmse_npiv = zeros(rep,1);
rmse_iv = zeros(rep,2);

for r = 1:rep
    
    D = datasample(dataset,n,1,"Replace",true);     % Simulation sample
    Y = D(:,1);
    X = D(:,2);
    Z = D(:,3);
    
    % Transform X and Z to Unit Interval
    minx = min(X);
    rangeX = max(X) - minx;
    minz = min(Z);
    rangeZ = max(Z) - minz;
    X = (X - minx) ./ rangeX;
    Z = (Z - minz) ./ rangeZ;
    
    % define a Legendre polynomial basis for L2[0,1]
    
    MX = zeros(n,K);
    for i=1:K
        if i == 1
            MX(:,i) = ones(n,1);
        elseif mod(i,2) == 0
            MX(:,i) = sqrt(2)*sin(floor((i)/2)*(X.*2*pi));
        else
            MX(:,i) = sqrt(2)*cos(floor((i)/2)*(X.*2*pi));
        end
    end
    
    MZ = zeros(n,K);
    for i=1:K
        if i == 1
            MZ(:,i) = ones(n,1);
        elseif mod(i,2) == 0
            MZ(:,i) = sqrt(2)*sin(floor((i)/2)*(Z.*2*pi));
        else
            MZ(:,i) = sqrt(2)*cos(floor((i)/2)*(Z.*2*pi));
        end
    end
    
    % NPIV estimate
    G_hat = (MZ'*MX)\(MZ'*Y);
    G_hat_dist(r,:) = G_hat';
    g_fit = MX*G_hat;

    % RMSE of NPIV
    index = zeros(n,1);      % index of population matrix that stores value of i-th obs
    for i = 1:n
        pop_index =  find(population(:,2)==D(i,2));
        index(i,1) = pop_index(1,1);
    end
    error = g_fit - population(index,4);
    rmse_npiv(r,1) = sqrt(mean(error.^2));
    
    Xmat = [ones(n,1),X];
    Zmat = [ones(n,1),Z];
    betaIV = (Zmat'*Xmat)\(Zmat'*Y);
    beta_iv_dist(r,:) = betaIV;
    Y_IV = Xmat*betaIV;

    % RMSE of IV
    error = Y_IV - population(index,4);
    rmse_iv(r,1) = sqrt(mean(error.^2));    % rmse wrt true g(X)
    error = Y_IV - population(index,5);
    rmse_iv(r,2) = sqrt(mean(error.^2));    % rmse wrt X'betaIV0

    % beta = (Xmat'*Xmat)\(Xmat'*Y);
    % Y_OLS = Xmat*beta;
    % 
    % figure
    % plot(X,g_fit,'.',X,Y_IV,'.',X,Y,'.')
    % legend 'NPIV' 'IV' 'OLS' 'obs'
    % ylim([-0.2,0.6])


end

results = zeros(4,3);
results(1,1:2) = mean(rmse_iv);
results(2,1) = mean(rmse_npiv);
results(3,1:2) = std(beta_iv_dist);
results(4,1:2) = std(G_hat_dist);




beta = (Xmat'*Xmat)\(Xmat'*Y);
Y_OLS = Xmat*beta;

figure
plot(D(:,2),g_fit,'.',D(:,2),Y_IV,'.',D(:,2),Y_OLS,'.',D(:,2),Y,'.')
legend 'NPIV' 'IV' 'OLS' 'obs'
ylim([-0.2,0.6])


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Averaging Estimator %%%%%%%%%%%%%%%%%
%%%%% Different Truncation parameters %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rng("default")  % fix seed for replicability
rep = 5000;    % number of repetition
n = 20000;    % number of obs.

% pre-allocate matrices for saving the estimates at each rep
rmse_npiv = zeros(rep,3);   % rmse for npiv's
rmse_aver = zeros(rep,1);   % rmse for averaging 3 estimators
rmse_aver1 = zeros(rep,1);  % rsme for averaging 2 estimators
G_hat_cell = cell(3,1); % The Fourier coef's
G_hat_cell{1,1} = zeros(rep,2);
G_hat_cell{2,1} = zeros(rep,3);
G_hat_cell{3,1} = zeros(rep,4);
J_dist = zeros(rep,3);  % J-statistics

for r = 1:rep

    % Randomly draw the Simulation samples with replacement
    D = datasample(dataset,n,1,"Replace",true);     
    Y = D(:,1);
    X = D(:,2);
    Z = D(:,3);
    
    % Transform X and Z to Unit Interval
    minx = min(X);
    rangeX = max(X) - minx;
    minz = min(Z);
    rangeZ = max(Z) - minz;
    X = (X - minx) ./ rangeX;
    Z = (Z - minz) ./ rangeZ;
    
    % index of population matrix that stores value of i-th obs.
    index = zeros(n,1);      
    for i = 1:n
        pop_index =  find(population(:,2)==D(i,2));
        index(i,1) = pop_index(1,1);
    end

    % repeat for each truncation parameter K
    g_fit_mat = zeros(n,3); % To store the fitted valued of each K
    Jk = zeros(1,3);        % To store the J-stat of each K
    for K = 2:4

        % define a Legendre polynomial basis for L2[0,1]       
        MX = zeros(n,K);
        for i=1:K
            if i == 1
                MX(:,i) = ones(n,1);
            elseif mod(i,2) == 0
                MX(:,i) = sqrt(2)*sin(floor((i)/2)*(X.*2*pi));
            else
                MX(:,i) = sqrt(2)*cos(floor((i)/2)*(X.*2*pi));
            end
        end
        
        % To make the estimation problem over-identified, we add one more dimension to MZ
        MZ = zeros(n,K+1);  
        for i=1:K+1
            if i == 1
                MZ(:,i) = ones(n,1);
            elseif mod(i,2) == 0
                MZ(:,i) = sqrt(2)*sin(floor((i)/2)*(Z.*2*pi));
            else
                MZ(:,i) = sqrt(2)*cos(floor((i)/2)*(Z.*2*pi));
            end
        end

        % NPIV estimate
        G_hat = (((MX'*MZ)/(MZ'*MZ))*(MZ'*MX))\(((MX'*MZ)/(MZ'*MZ))*(MZ'*Y));   % the 2SLS version
        G_hat_cell{K-1,1}(r,:) = G_hat;  % store G_hat's for each K
        g_fit = MX*G_hat;
        g_fit_mat(:,K-1) = g_fit;   % store the fitted values

        % RMSE of NPIV
        error = g_fit - population(index,4);
        rmse_npiv(r,K-1) = sqrt(mean(error.^2));

        Mgn = (MZ'*(Y-g_fit))/n;    % sample analog of the (misspecified) moment condition.
        Jn = (Mgn'/((MZ'*MZ)/n))*Mgn; % J-statistic
        J_dist(r,K-1) = Jn;
        Jk(1,K-1) = Jn;
    end

    % Now, we compute the average estimator.
    %Jk = Jk.^(5);
    % Compute the weights
    omega1 = (1-(Jk(1,1)/sum(Jk,"all")))/2;
    omega2 = (1-(Jk(1,2)/sum(Jk,"all")))/2;
    omega3 = (1-(Jk(1,3)/sum(Jk,"all")))/2;
    % Compute the averaging estimator.
    g_aver = omega1*g_fit_mat(:,1)+omega2*g_fit_mat(:,2)+omega3*g_fit_mat(:,3);

    % RMSE of the Averaging NPIV
    error = g_aver - population(index,4);
    rmse_aver(r,1) = sqrt(mean(error.^2));

    % 2. Averaging 2 estimators
    omega5 = (1-Jk(1,1)/(Jk(1,1)+Jk(1,2)));
    omega6 = (1-Jk(1,2)/(Jk(1,1)+Jk(1,2)));
    g_aver1 = omega5*g_fit_mat(:,1)+omega6*g_fit_mat(:,2);

    error = g_aver1 - population(index,4);
    rmse_aver1(r,1) = sqrt(mean(error.^2));

%     figure(8), hold on
%     plot(D(:,2),g_aver1,'b.', ...
%         population(:,2),g_true,'.r')
%     ylim([-0.1,0.6])
end


RMSEs = zeros(4,1);
RMSEs(1:3,1) = mean(rmse_npiv,1);
RMSEs(4,1) = mean(rmse_aver,1);

RMSEs1 = zeros(3,1);
RMSEs1(1:2,1) = mean(rmse_npiv(:,1:2),1);
RMSEs1(3,1) = mean(rmse_aver1,1);