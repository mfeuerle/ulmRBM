function Z_ht = zfullmatrix(t,int)
% returns some auxilary matrices needed for the hierarchical structure of
% of the basis functions
% Input: 
% t ... vector with time steps
% int ... integer, either 0,1,20,21 for the corresponding matrix (see readme.pdf)

% Author: Richard Löscher, 17.11.2022


ht = t(2:end)-t(1:end-1);
Nt = length(ht);

switch int
    %% first order auxiliary matrices
    case 1
        Z_hta=spdiags(-1./ht',0,Nt+1,Nt); % set diagonal entries
        Z_htb=spdiags(1./ht',-1,Nt+1,Nt); % set off diagonal entries
        Z_ht = Z_hta+Z_htb; 
    case 0
        Z_hta=spdiags((t(2:end)./ht)',0,Nt+1,Nt); % set diagonal entries
        Z_htb=spdiags((-t(1:end-1)./ht)',-1,Nt+1,Nt); % set off diagonal entries
        Z_ht = Z_hta+Z_htb;
    %% second order auxiliary matrices for full second order space
    case 21
        ht_p = ht(1:end-1)+ht(2:end); %sum up h
        hti = [ht(1),ht_p,ht(end)]; % length = Nt+1
        Z_hta=spdiags(-1./hti',0,Nt+2,Nt+1); % set diagonal entries
        Z_htb=spdiags(1./hti',-1,Nt+2,Nt+1); % set off diagonal entries
        Z_ht = Z_hta+Z_htb; 
    case 20
        ht_p = ht(1:end-1)+ht(2:end); %sum up h
        hti = [ht(1),ht_p,ht(end)]; % length = Nt+1
        Z_hta=spdiags([t(2:end),t(end)]'./hti',0,Nt+2,Nt+1); % set diagonal entries
        Z_htb=spdiags(-[t(1),t(1:end-1)]'./hti',-1,Nt+2,Nt+1); % set off diagonal entries
        Z_ht = Z_hta+Z_htb; 
    case 22
        ht_p = ht(1:end-1)+ht(2:end); %sum up h
        ht_h1 = ht_p.*ht(1:end-1); ht_h2 = ht_p.*ht(2:end); 
        Z_hta = spdiags(2./([ht(1)^2,ht_h2]'),0,Nt+2,Nt);
        Z_htb = spdiags(-2*(1./[ht(1)^2,ht_h2]'+1./[ht_h1,ht(end)^2]'),-1,Nt+2,Nt);
        Z_htc = spdiags(2./[ht_h1,ht(end)^2]',-2,Nt+2,Nt);
        Z_ht = Z_hta+Z_htb+Z_htc; 
    otherwise
        disp('int has to be chosen as 1 or 0 in zmatrix!!')
end

end
