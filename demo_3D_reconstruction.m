clear all; close all; clc;

% Load rotations, imagedata and Calibration.
% The datasets are available online at http://www.maths.lth.se/matematiklth/personal/calle/
load .\dataset\Cathedral\rotations.mat
load .\dataset\Cathedral\Calib_Results.mat KK;

%remove points that are not visible in two cameras or more
vis = zeros(1,size(u{1},2));
for i=1:length(u)
    vis = vis + isfinite(u{i}(1,:));
end
vis = vis >= 2;
for i = 1:length(u)
    u{i} = u{i}(1:2,vis);
end

%Solve known rotation problem with slack variables
tol = 5/KK(1,1); % error tolerance roughly 5 pixels

% depth constraint
depth_min = 0.1;
depth_max = 100;


% L1 algorithm of "Olsson, Eriksson, Hartley, Outlier Removal using Duality, CVPR2010"
% "krot_slack.m" is a copy from http://www.maths.lth.se/matematiklth/personal/calle/
tic; [U,P,slack(1,:)] = krot_slack(u,A,tol,depth_min,depth_max);toc
[~,~,res(1),nr(1)] = remout_bundle(U,P,u,slack(1,:),1e-7,KK);


% Algorithm 1: L1 algorithm with reduced dimension
tic; [U,P,slack(2,:)] = krot_slack_reduced(u,A,tol,depth_min,depth_max);toc
[~,~,res(2),nr(2)] = remout_bundle(U,P,u,slack(2,:),1e-7,KK);


% Algorithm 2: Iteratively reweighted algorithm with reduced dimension
tic; [U,P,slack(3,:)] = krot_irw_reduced(u,A,tol,depth_min,depth_max);toc
[U,~,res(3),nr(3)] = remout_bundle(U,P,u,slack(3,:),1e-7,KK);

RO = sum((slack>1e-7)');

disp(['Removed outliers (RO):  ', num2str(RO)])
disp(['Remaining inliers (RI): ', num2str(nr)])
disp(['RMSE (pixels):          ', num2str(res,4)])

plot3(U(1,:),U(2,:),U(3,:),'.','MarkerSize',3);
xlabel('x');ylabel('y');
