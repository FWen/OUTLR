function [U,P,slack] = krot_slack(u,A,tol,min_depth,max_depth)
% Solves the known rotation problem with slack variables as stated in
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
% (C) 2010 Carl Olsson (calle@maths.lth.se, carl.a.c.olsson@gmail.com)

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

%Error residuals
A1 = [-a-tol*c; a-tol*c; -b-tol*c; b-tol*c];
B1 = [a0+tol*c0; -a0+tol*c0; b0+tol*c0; -b0+tol*c0];

%Depth restrictions 
A2 = [-c; c];
B2 = [c0-min_depth; max_depth-c0];

%Add slack variables
A = [A1; A2]; 
C = [B1; B2; zeros(size(A,1),1)]; 
B = [sparse(size(A1,2),1); ones(size(A,1),1)]; 
A = [A -speye(size(A,1)); 
    sparse(size(A,1),size(A,2)), -speye(size(A,1))];

K.l = size(A,1);
pars.eps = 1e-10;
pars.maxiter = 1000;
pars.fid = 0;

[X,Y,info] = sedumi(A,-B,C,K,pars);
s = Y(size(A1,2)+1:end);
Y = Y(1:size(A1,2));
s = s(1:size(a,1)) + s(size(a,1)+1:2*size(a,1)) + s(2*size(a,1)+1:3*size(a,1)) + s(3*size(a,1)+1:4*size(a,1))+...
    s(4*size(a,1)+1:5*size(a,1)) + s(5*size(a,1)+1:end);

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
