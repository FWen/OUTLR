function [U,P,res,nv] = remout_bundle(U,P,u,s,thresh,KK);
% This code is modified from that of Carl Olsson at
%  http://www.maths.lth.se/matematiklth/personal/calle/

for i = 1:length(P);
    vis = find(isfinite(u{i}(1,:)));
    res = length(vis);
    indx = find(s(1:res) > thresh);
    u{i}(:,vis(indx)) = NaN;
    s = s(res+1:end);
end

%Refine using bundle ajustment
[U,P] = bundle(U,P,u,20);

Uh = [U; ones(1,size(U,2))];
res = 0;
nv = 0;
for i = 1:length(P);
    vis = isfinite(u{i}(1,:));
    nv  = nv + sum(vis);
    res = res + ...
        sum( (KK(1,1)*((P{i}(1,:)*Uh(:,vis))./(P{i}(3,:)*Uh(:,vis)) - u{i}(1,vis))).^2 ) + ...
        sum( (KK(2,2)*((P{i}(2,:)*Uh(:,vis))./(P{i}(3,:)*Uh(:,vis)) - u{i}(2,vis))).^2 );
end

res = sqrt(res/2/nv);
end