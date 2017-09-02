clear;
clc;
close all;

addpath(genpath('gco-v3.0'));
load('girl.mat');
[h,w,~,n] = size(input(:,:,:,:));

%% parameters setting
r = 1;                                % low rankness
param.maxiter = 40;       % maximal iteration number
param.k = 3;                    % the number of MoG components
param.NumIter = 100;     % maximum iteration number
param.tol = 1.0e-2;         % threshold can be tuned smaller for better result
param.lambda = 5;         % control the balance of regulariation terms
param.weight = 1;          % weight of time direction of 3dtv, larger if obvious streaks are seen in the mask
param.rho = 0.9;             % control the size of mask, larger mask if smaller rho
par.patsize = 2;              % patch size
par.Pstep = 1;                 % patch passing step


%% run RSRV
derain = zeros(size(input));
for i = 1:3  %RGB channels
    D = reshape(uint8(input(:,:,i,:)),[h,w,n]);
    D = double(D);
    D = mat2gray(D);
    [U,S,V] = svd(Unfold(D,size(D),3)','econ');
    U = U*S.^.5;                % initialized factorized matrice U
    V = V*S.^.5;                 % initialized factorized matrice V
    param.InU = U(:,1:r);
    param.InV = V(:,1:r);
    
    [label,model,OutU,OutV,Mask,llh] = PatchMoG(D,r,param,par);
    tempC = OutU*OutV';
    C = Fold(tempC',size(D),3);
    derain(:,:,i,:) = double(1-Mask).*C+double(Mask).*D;
end

implay(input);  
implay(derain);