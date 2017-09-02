function [label,model,OutU,OutV,Mask,llh] = PatchMoG(D,r,param,par)
sizeVideo = size(D);
InX  = Unfold(D,sizeVideo,3)';
p  = par.patsize;                                                 % patch size
step = par.Pstep;                                               % patch step
N = (floor((sizeVideo(1)-p)/step)+1)*(floor((sizeVideo(2)-p)/step)+1)*sizeVideo(3);      %Total Patch Number

if (~isfield(param,'maxiter'))
    maxiter = 100;
else
    maxiter = param.maxiter;
end

if (~isfield(param,'InU'))
    s = median(abs(InX(:)));
    s = sqrt(s/r);
    if min(InX(:)) >= 0
        InU = rand(size(InX,1),r)*s;
    else
        InU = rand(size(InX,1),r)*s*2-s;
    end
else
    InU = param.InU;
end

if (~isfield(param,'InV'))
    if min(InX(:)) >= 0
        InV = rand(size(InX,2),r)*s;
    else
        InV = rand(size(InX,2),r)*s*2-s;
    end
else
    InV = param.InV;
end

if (~isfield(param,'k'))
    k = 3;
else
    k = param.k;
end

if (~isfield(param,'tol'))
    tol = 1.0e-7;
else
    tol = param.tol;
end

if (~isfield(param,'lambda'))
    param.lambda = 5;
end       % alpha = param.lambda * beta;

if (~isfield(param,'sigma'))
    param.sigma = [];
end

if (~isfield(param,'weight'))
    param.weight = 1;
end

if (~isfield(param,'rho'))
    param.rho = 0.3;
end

WPat = ones(p*p,N/sizeVideo(3));
W = WIm(InX,par,WPat,sizeVideo);
Y = Video2Patch(D,par);          % Y = f(D)

%% Initialize model parameters
R = initialization(Y,k);
[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label));
model.mu = zeros(p,p,k);
Diag = zeros(p*p,p*p,k);
U = zeros(p*p,p*p,k);
for i = 1:k
    Diag(:,:,i) = diag(rand(p*p,1));
    U(:,:,i) = orth(rand(p*p,p*p));
    model.Sigma(:,:,i) = U(:,:,i)'*Diag(:,:,i)*U(:,:,i);
end
nk = sum(R,1);
model.weight = nk/size(R,1);

% graph cuts initialization
% GCO toolbox is called
Omega = true(size(InX));                    % background support
ObjArea = sum(~Omega(:));
OmegaOut = false(size(InX));
minObjArea = numel(InX(:,1))/1e4;    % minimum number of outliers
sigma = param.sigma;
lambda = param.lambda;
beta = 0.5*(std(InX(:,1)))^2;                
minbeta = 0.5*(3*std(InX(:,1))/20)^2; 
weight = param.weight;
rho = param.rho;                                  %control the size the mask

hMRF = GCO_Create(sizeVideo(1)*sizeVideo(2)*sizeVideo(3),2);
GCO_SetSmoothCost( hMRF, [0 1;1 0] );
AdjMatrix = getAdj([sizeVideo(1),sizeVideo(2),sizeVideo(3)],weight);
amplify = 10 * lambda;
GCO_SetNeighbors( hMRF, amplify * AdjMatrix );
energy_cut = 0;
energy_old = inf;
converged = false;
TempU = InU;
TempV = InV;
TempX = TempU*TempV';
TempError = Fold( (double(Omega).*(InX-TempX))',size(D),3);
Error = Video2Patch(TempError,par);

mu = 0.05;
Lambda = zeros(p*p,N);

C = double(Omega).*(TempU*TempV')+double(~Omega).*InX;
tempC = Fold(C',sizeVideo,3);
temp = Video2Patch(tempC, par);

t = 1;
%%%%%%%%%%%%%%%%Initialized E Step %%%%%%%%%%%%%%%%%%%
[R, llh(t)] = expectation(Error,model,par);
%%%%%%%%%%%%%%%%Initialized E Step %%%%%%%%%%%%%%%%%%%

while ~converged && t < maxiter
    t = t+1;
    fprintf('-----------This is %d iteration.-----------\n',t-1);
    
    %%%%%%%%%%%%%%%% M Step 1 %%%%%%%%%%%%%%%%%%%
    disp('*** Update MoG parameters *** ');
    [model] = maximizationModel(Error,R,par);
    %%%%%%%%%%%%%%%% M Step 1 %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% E Step %%%%%%%%%%%%%%%%%%%
    disp('*** Update Gaussian components responsibilties *** ');
    [R, llh(t)] = expectation(Error,model,par);
    L1 = llh(t);
    fprintf('Likelihood equals to %f \n', L1)
    %%%%%%%%%%%%%%%% E Step %%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%% M Step 2 %%%%%%%%%%%%%%%%%%%
    %% Update L
    disp('*** Update variable L *** ');
    L = UpdateL(Y,temp,Lambda,model.Sigma,R,k,mu);
    L_Im = LIm(par,L,Lambda,mu,sizeVideo);
    %% Update U,V
    disp('*** Update low-rank matrices U, V *** ');
    TempV=subpro(((L_Im+double(~Omega).*(TempU*TempV'-InX))./W),W,TempU,size(C,2));
    TempU=subpro(((L_Im+double(~Omega).*(TempU*TempV'-InX))./W)',(W)',TempV,size(C,1));
    C = double(Omega).*(TempU*TempV')+double(~Omega).*InX;
    tempC = Fold(C',sizeVideo,3);
    temp = Video2Patch(tempC, par);
    Lambda = Lambda + mu*(L-temp);
    mu = mu*1.05;
    TempX = TempU*TempV';
    E = InX - TempX;
    
    %% estimate sigma
    disp('*** Update foreground support *** ');
    if isempty(sigma)
        sigma_old = sigma;
        residue = sort(E(Omega(:)));
        truncate = 0.005;
        idx1 = round(truncate*length(residue))+1;
        idx2 = round((1-truncate)*length(residue));
        sigma = std(residue(idx1:idx2));
        if abs(sigma_old-sigma)/abs(sigma_old) < 0.01
            sigma = sigma_old;
        end
    end
    % update beta
    if ObjArea < minObjArea
        beta = beta/2;
    else
        beta = min(max([beta/2,3*(rho*sigma)^2 minbeta]),beta);
    end
    alpha = lambda * beta;
    
    % estimate S = ~Omega;
    % comment these part if there is no moving object
    disp('*** Estimate Outlier Support *** ');
    if lambda > 0
        % call GCO to run graph cuts
        GCO_SetDataCost( hMRF, (amplify/alpha)*[ 0.7*(E(:)).^2, ~OmegaOut(:)*beta + OmegaOut(:)*0.5*max(E(:)).^2]' );
        GCO_Expansion(hMRF);
        Omega(:) = ( GCO_GetLabeling(hMRF) == 1 )';
        energy_cut = energy_cut + double( GCO_ComputeEnergy(hMRF) );
        ObjArea = sum(Omega(:)==0);
        energy_cut = (alpha/amplify) * energy_cut;
    else
        % direct hard thresholding if no smoothness
        Omega = 0.5*E.^2 < beta;
        ObjArea = sum(Omega(:)==0);
        energy_cut =  beta*ObjArea;
    end
    
    TempX = TempU*TempV';
    TempError = Fold( (double(Omega).*(InX-TempX))',sizeVideo,3);
    Error = Video2Patch(TempError,par);
    
    
    %%%%%%%%%%%%%%%% M Step 2 %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% E Step %%%%%%%%%%%%%%%%%%%
    [R, llh(t)] = expectation(Error,model,par);
    L2 = llh(t);
    fprintf('Likelihood equals to %f \n', L2)
    energy = energy_cut - L2;
    model.R = R;
    %%%%%%%%%%%%%%%% E Step %%%%%%%%%%%%%%%%%%%
    
    %% check termination condition
    if ObjArea > minObjArea && abs(energy_old-energy)/energy < tol
        break;
    end
    energy_old = energy;
    
end
OutU = TempU;
OutV = TempV;
Mask = reshape(~Omega,size(D));
    disp(['There are ',num2str(k),' Gaussian noises mixed in data']);
    disp(['with weights ',num2str(model.weight),'.']);
if abs(energy_old-energy)/energy < tol
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end
end


%% function to get the adjcent matirx of the graph
function W = getAdj(sizeData,weight)
numSites = prod(sizeData);
id1 = [1:numSites, 1:numSites, 1:numSites];
id2 = [ 1+1:numSites+1,...
    1+sizeData(1):numSites+sizeData(1),...
    1+sizeData(1)*sizeData(2):numSites+sizeData(1)*sizeData(2)];
value = [weight*ones(1,2*numSites),1*ones(1,numSites)];
W = sparse(id1,id2,value);
W = W(1:numSites,1:numSites);
end
%% function to calculate the times each pixel covered by the running patch
function W = WIm(InX,par,WPat,sizeVideo)
p = par.patsize;
step = par.Pstep;
Y = Fold(InX,sizeVideo,3);
[d,n,~] = size(Y);
TempR        =   floor((d-p)/step)+1;
TempC        =   floor((n-p)/step)+1;
TempOffsetR  =   [1:step:(TempR-1)*step+1];
TempOffsetC  =   [1:step:(TempC-1)*step+1];
k = 0;
tempW = zeros(d,n);
for i  = 1:p
    for j  = 1:p
        k    =  k+1;
        tempW(TempOffsetR-1+i,TempOffsetC-1+j)      =  tempW(TempOffsetR-1+i,TempOffsetC-1+j) + reshape(WPat(k,:)',  [TempR TempC]);
    end
end
W = reshape(tempW,d*n,1);
W = repmat(W,1,sizeVideo(3));
W = W+ double(W==0);
end

%% function to initialize MoG paramters
function R = initialization(X, init, par)
[d,n] = size(X);
if isstruct(init)  % initialize with a model
    R  = expectation(X,init,par);
elseif length(init) == 1  % random initialization
    k = init;
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
    while k ~= length(u)
        idx = randsample(n,k);
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        [u,~,label] = unique(label);
    end
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == d  %initialize with only centers
    k = size(init,2);
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end
end

%% E step
function [R, llh] = expectation(X, model, par)
mu = model.mu;
Sigma = model.Sigma;
w = model.weight;
n = size(X,2);
k = size(mu,3);
p = par.patsize;
mu1 = zeros(p*p,1,k);
for i = 1:k
    mu1(:,:,i)= mu1(:,:,i) + reshape(mu(:,:,i),p*p,1);
end
[~,N] = size(X);
A = zeros(N,k);

for i=1:k
    A(:,i) = loggausspdf(X,mu1(:,:,i),Sigma(:,:,i));
end

logRho = bsxfun(@plus,A,log(w));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
end

function y = loggausspdf(X, mu, Sigma)
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;
end

function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
end

%% M step to update PatchMoG parameters
function [model] = maximizationModel(X,R,par)

p = par.patsize;
[~,TotalPatNum] = size(X);
X0 = reshape(X, p*p, 1,TotalPatNum);
X1 = reshape(X0, p, p, TotalPatNum);

k = size(R,2);
nk = sum(R,1);
mu = zeros(p,p,k);
w = nk/size(R,1);
Sigma = zeros(p*p , p*p , k);
sqrtR = sqrt(R);

for i=1:k
    X2=bsxfun(@minus,X1,mu(:,:,i));
    X3=reshape(X2,p*p,1,TotalPatNum);
    X4=reshape(X3,p*p, TotalPatNum);
    X5=bsxfun(@times,X4,sqrtR(:,i)');
    Sigma(:,:,i)=X5*X5'/nk(i);
    Sigma(:,:,i)=Sigma(:,:,i)+(1e-6);
end

model.mu = mu;
model.Sigma = Sigma;
model.weight = w;
end
%% function to update L
function L = UpdateL(Pa,PaUV,lambda,Sigma,Gamma,k,rho)
[d,n]     = size(Pa);
tempZ     = rho*PaUV-lambda;
tempSig   = zeros(d,d+1,n);
tempGam   = zeros(1,1,n);
for i = 1:k
    tempGam(1,1,:) = Gamma(:,i);
    invSig = inv(Sigma(:,:,i));
    tempSig(:,1:d,:) = bsxfun(@times,invSig,tempGam)+tempSig(:,1:d,:);
    temp = bsxfun(@times,Gamma(:,i),Pa')';
    tempZ = tempZ + invSig*temp;
end
tempSig(:,1:d,:) = tempSig(:,1:d,:) + rho*repmat(eye(d),[1,1,n]);
tempSig(:,d+1,:)   = permute(tempZ,[1 3 2]);
if d<=9
    for i = 1:d
        tempSig(i,:,:)     = bsxfun(@rdivide,tempSig(i,:,:),tempSig(i,i,:));
        for j = i+1:d
            tempSig(j,:,:) = tempSig(j,:,:)- bsxfun(@times,tempSig(i,:,:),tempSig(j,i,:));
        end
    end
    for i = d:-1:2
        for j = 1:i-1
            tempSig(j,:,:) = tempSig(j,:,:)-bsxfun(@times, tempSig(i,:,:),tempSig(j,i,:));
        end
    end
    L = permute(tempSig(:,d+1,:),[3,1,2])';
else
    L=zeros(d,1,n);
    for i=1:n
        L(:,:,i)=inv(tempSig(:,1:d,i))*tempSig(:,d+1,i);
    end
    L=permute(L,[1,3,2]);
end
end

function L_Im = LIm(par,L,Lambda,mu,sizeVideo)
p = par.patsize;
step = par.Pstep;
PatchNum = size(L,2)/sizeVideo(3);      %Total Patch Number

TempR        =   floor((sizeVideo(1)-p)/step)+1;
TempC        =   floor((sizeVideo(2)-p)/step)+1;
TempOffsetR  =   [1:step:(TempR-1)*step+1];
TempOffsetC  =   [1:step:(TempC-1)*step+1];
k = 0;
L_Im = zeros(sizeVideo);
for i  = 1:p
    for j  = 1:p
        k    =  k+1;
        for t = 1:sizeVideo(3)
            L_Im(TempOffsetR-1+i,TempOffsetC-1+j,t)  =  L_Im(TempOffsetR-1+i,TempOffsetC-1+j,t)...
                + reshape(L(k,(t-1)*PatchNum+1:t*PatchNum)', [TempR TempC])...
                + reshape(Lambda(k,(t-1)*PatchNum+1:t*PatchNum)',[TempR TempC])/mu;
            %                 -reshape(temp(k,(t-1)*PatchNum+1:t*PatchNum)', [TempR TempC]);
        end
    end
end

L_Im = reshape(L_Im,sizeVideo(1)*sizeVideo(2),sizeVideo(3));
end
%% Update U,V
function V=subpro(X,H,U,n)
H1   = permute(sqrt(H),[1 3 2]);
U1   = repmat(U,[1 1 n]);
temp1 = bsxfun(@times,permute(H,[1 3 2]),U1);
temp1 = permute(temp1,[2 1 3]);
U1   = bsxfun(@times,H1,U1);
U2   = permute(U1,[2 1 3]);
temp = TensorTimes(U2,U1);
r    = size(temp,1);
H1   = permute(H,[1 3 2]);
X    = permute(X,[3 1 2]);
temp1 = sum(bsxfun(@times,temp1,X),2);
temp = [temp,temp1];
for i = 1:r
    temp(i,:,:)     = bsxfun(@rdivide,temp(i,:,:),temp(i,i,:));
    for j = i+1:r
        temp(j,:,:) = temp(j,:,:)- bsxfun(@times,temp(i,:,:),temp(j,i,:));
    end
end
for i = r:-1:2
    for j = 1:i-1
        temp(j,:,:) = temp(j,:,:)-bsxfun(@times, temp(i,:,:),temp(j,i,:));
    end
end
V = permute(temp(:,r+1,:),[3,1,2]);
end
function C = TensorTimes(A,B)
A = permute(A,[2,4,3,1]);
C = bsxfun(@times, A,B);
C = sum(C,1);
C = permute(C,[4,2,3,1]);
end
%% function f
function Y = Video2Patch(Video,par)
if isfield(par,'Pstep')
    step   = par.Pstep;
else
    step   = 1;
end
patsize = par.patsize;
TotalPatNum = (floor((size(Video,1)-patsize)/step)+1)*(floor((size(Video,2)-patsize)/step)+1);                 %Total Patch Number in the image
sizeVideo = size(Video);
for k = 1:sizeVideo(3)
    Y(:,(k-1)*TotalPatNum+1:k*TotalPatNum) = Im2Patch(Video(:,:,k),par);
end
end

function  [Y]  =  Im2Patch( E_Img, par )
if isfield(par,'Pstep')
    step   = par.Pstep;
else
    step   = 1;
end
patsize = par.patsize;
TotalPatNum = (floor((size(E_Img,1)-patsize)/step)+1)*(floor((size(E_Img,2)-patsize)/step)+1);                 %Total Patch Number in the image
Y           =   zeros(par.patsize*par.patsize, TotalPatNum);                      %Current Patches                   %Patches in the original noisy image
k           =   0;

for i  = 1:par.patsize
    for j  = 1:par.patsize
        k           =  k+1;
        E_patch     =  E_Img(i:step:end-patsize+i,j:step:end-patsize+j);
        Y(k,:)      =  E_patch(:)';
    end
end         %Estimated Local Noise Level
end