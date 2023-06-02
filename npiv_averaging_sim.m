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

population = [population,g_true,Y_IV0];   % population matrix

figure
plot(X,g_true,'.',X,Y_IV0,'.',X,Y_OLS0,'.',X,Y,'.')
legend 'NPIV' 'IV' 'OLS' 'obs'
ylim([-0.2,0.6])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Monte-Carlo Simulation : n = 200 %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng("default")
rep = 5000;    % number of repetition
n = 5000;    % number of obs.

K = 3;  % truncation parameter - should be dependent on the data. For now I just fix it.

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

mean(rmse_iv)
mean(rmse_npiv)
std(beta_iv_dist)
std(G_hat_dist)




beta = (Xmat'*Xmat)\(Xmat'*Y);
Y_OLS = Xmat*beta;

figure
plot(X,g_fit,'.',X,Y_IV,'.',X,Y_OLS,'.',X,Y,'.')
legend 'NPIV' 'IV' 'OLS' 'obs'
ylim([-0.2,0.6])



