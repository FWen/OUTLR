function [U,P,slack] = krot_irw_reduced(u,A,tol,min_depth,max_depth)
% Solves the known rotation problem with slack variables using a
% dimension reduced formualtion by iteratively reweighted L1-min, details see:
% Wen, Zou, Liu, Efficient Global Outlier Removal for Large Scale Multiview
% Reconstruction, 2018
%
% This code is modified from that of Carl Olsson at
%  http://www.maths.lth.se/matematiklth/personal/calle/
% Olsson, Eriksson, Hartley, Outlier Removal using Duality, CVPR2010
%
% inputs -  u: 1xD cell with image data.
%           u{i} is of size 3xN, where N is the number of observed points.
%           If point j is not observed in image then u{i}(:,j) = NaN.
%
%        -  A: 1xD cell with estimated orientation matrices.
%
%        -  tol: maximal inlier tolerance.
%
%        -  min_depth: minimal allowed depth.
%
%        -  max_depth: maximal allowed depth.
%
% outputs - U: 3xD cell with 3D points
%
%         - P: 1xD cell with camera matrices
%
%         - slack: slackvariables. slack(i) > 0 indicates that U(:,i) might
%           be an outlier.
%

[a,a0,b,b0,c,c0] = gen_krot(u,A);

[Linfsol,slack] = LinfSolverfeas(a,a0,b,b0,c,c0,tol,min_depth,max_depth);

[U,P] = form_str_mot(u,A,Linfsol);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U,P] = form_str_mot(u,A,sol)
numpts = size(u{1},2);
numcams = length(A);

U = reshape(sol(1:(3*numpts)), [3 numpts]);

tpart = sol(3*numpts+1:end);
P = cell(size(A));
P{1} = [A{1} [0 0 0]'];
for i=2:length(A)
    P{i} = [A{i} [tpart([(i-2)*3+1:(i-1)*3])]];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y,s] = LinfSolverfeas(a,a0,b,b0,c,c0,tol,min_depth,max_depth)

%Set up the problem in SeDuMi form.
[ma, na] = size(a);
n_elc = 6; % number of equivalent linear constraints

A1 = sparse(ma*n_elc,na);
A1(1:6:end,:) = -a-tol*c;
A1(2:6:end,:) =  a-tol*c;
A1(3:6:end,:) = -b-tol*c;
A1(4:6:end,:) =  b-tol*c;
A1(5:6:end,:) =  -c;
A1(6:6:end,:) =  c;

B1 = sparse(ma*n_elc,1);
B1(1:6:end,:) =  a0+tol*c0;
B1(2:6:end,:) = -a0+tol*c0;
B1(3:6:end,:) =  b0+tol*c0;
B1(4:6:end,:) = -b0+tol*c0;
B1(5:6:end,:) = c0-min_depth;
B1(6:6:end,:) = max_depth-c0;

%Add slack variables
C = [B1; sparse(ma,1)]; 
B = [sparse(na,1); ones(ma,1)];  
J = kron(speye(ma),ones(n_elc,1));
A = [A1, -J; sparse(ma,na), -speye(ma)];

K.l = size(A,1);
pars.eps = 1e-10;
pars.maxiter = 1000;
pars.fid = 0;

[~,y,~] = sedumi(A,-B,C,K,pars);
s = y(na+1:end);
indx = find(s>0);
w = (s(indx)+1e-6).^(-0.9);

C = [B1; sparse(length(w),1)]; 
A = [A1, -J(:,indx); sparse(length(w),na), -speye(length(w))];
B = [sparse(na,1); w];
K.l = size(A,1);
[~,y,info] = sedumi(A,-B,C,K,pars);
s(indx) = y(na+1:end);
Y = y(1:na);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,a0,b,b0,c,c0] = gen_krot(u,A)
numvar  = 3*size(u{1},2)+3*length(A);
numpts  = size(u{1},2);
numcams = length(A);
% Reformulate the problem in matrix form
% Error residuals = max((a'*x+a0),(b'*x+b0))/(c'*x+c0);
a = [];
b = [];
c = [];
for i = 1:numcams;
    R = A{i};
    p = [u{i}; ones(1,size(u{i},2))];
    visible_points = isfinite(p(1,:));
    numres = sum(visible_points);
    
    
    %---First term---------------------------------------------------------
    %Compute the coeficients infront of the 3D point in each residual
    ptind = find(visible_points');
    pointcoeff = p(1,visible_points)'*R(3,:)-ones(numres,1)*R(1,:); 
    %Compute the position in the a-matrix
    pointcol = [(ptind-1)*3+1, (ptind-1)*3+2, ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Compute the coefficients infront of the translation part of each
    %residual
    tcoeff = [-ones(numres,1), zeros(numres,1), p(1,visible_points)']; 
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;
    
    %Create a new a-matrix and fill with data
    data = [pointcoeff(:); tcoeff(:)];
    row  = [pointrow(:);  trow(:)];
    col  = [pointcol(:); tcol(:)];
    newa = sparse(row,col,data,numres,numvar);

    
    %---Second term--------------------------------------------------------
    %Compute the coeficients infront of the 3D point in each residual
    ptind = find(visible_points');
    pointcoeff = p(2,visible_points)'*R(3,:)-ones(numres,1)*R(2,:); 
    %Compute the position in the b-matrix
    pointcol = [(ptind-1)*3+1 (ptind-1)*3+2 ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Compute the coeffisients infront of the translation part of each
    %residual
    tcoeff = [zeros(numres,1), -ones(numres,1), p(2,visible_points)']; 
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;

    %Create a new b-matrix and fill with data
    data = [pointcoeff(:); tcoeff(:)];
    row = [pointrow(:); trow(:)];
    col = [pointcol(:); tcol(:)];
    newb = sparse(row,col,data,numres,numvar);

    %---The denominator----------------------------------------------------
    %Compute the coeficients infront of the 3D point in each residual
    ptind = find(visible_points');
    pointcoeff = ones(numres,1)*R(3,:); % w: R(3,:)
    %Compute the position in the c-matrix
    pointcol = [(ptind-1)*3+1, (ptind-1)*3+2, ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Compute the coeffisients infront of the translation part of each
    %residual
    tcoeff = [zeros(numres,1), zeros(numres,1), ones(numres,1)];
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;

    %Create a new b-matrix and fill with data
    data = [pointcoeff(:); tcoeff(:)];
    row  = [pointrow(:);  trow(:)];
    col  = [pointcol(:);  tcol(:)];
    newc = sparse(row,col,data,numres,numvar);

    %Add to the old matrices
    a = [a; newa];
    b = [b; newb];
    c = [c; newc];
end

%Choose a coordinate system such that
%first cameracenter = (0,0,0)
%(The last dof is handled by the depth constraints.)

a = a(:,[1:numpts*3 (numpts*3+4):end]);
b = b(:,[1:numpts*3 (numpts*3+4):end]);
c = c(:,[1:numpts*3 (numpts*3+4):end]);
a0 = zeros(size(a,1),1);
b0 = zeros(size(b,1),1);
c0 = zeros(size(c,1),1);
