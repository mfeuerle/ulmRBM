function A_ht=arqmatrix(grid,r,q,rho)
% returns the auxiliary matrices A_ht^{r,q} for the computation of the
% Hilberttransformation using the Legendre chi function
% Input: 
% grid ... struct with time step parameters  
% r,q ... integers with degree for matrix
% rho ... index of truncation of the series expansion

% Author: Richard Löscher, 17.11.2022

T = grid.t(end);
[zeta,dGammaGamma] = zeta_Gamma_values(); %precomputed values of needed functions for chi function
A_ht = zeros(grid.N,grid.N); % A_ht is dense anyway
switch q
    case 0
        switch r
            case 0 % q=0, r=0
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0);
                    end
                end
            case 1 % q=0, r=1
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(1,0);
                    end
                end
            case 2 %q=0, r=2
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(1,0)+summand(2,0);
                    end
                end
            otherwise
                disp('a higher degree for r in arqmatrix is not yet available')
        end
    case 1
        switch r
            case 0 % q=1, r=0
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1);
                    end
                end
            case 1 % q=1, r=1
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1)...
                                        +summand(1,0)+summand(1,1);
                    end
                end
            case 2 % q=1, r=2
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1)...
                                        +summand(1,0)+summand(1,1)...
                                        +summand(2,0)+summand(2,1);
                    end
                end
            otherwise
                disp('q=1 and r>2 not yet available in arqmatrix')
        end
    case 2
        switch r
            case 0 %q=2,r=0
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1)+summand(0,2);
                    end
                end
            case 1 %q=2,r=1
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1)+summand(0,2)...
                                        +summand(1,0)+summand(1,1)+summand(1,2);
                    end
                end
            case 2 %q=2,r=2
                for k=2:grid.N+1
                    for l = 2:grid.N+1
                        A_ht(k-1,l-1) = summand(0,0)+summand(0,1)+summand(0,2)...
                                        +summand(1,0)+summand(1,1)+summand(1,2)...
                                        +summand(2,0)+summand(2,1)+summand(2,2);
                    end
                end 
            otherwise
                disp('q=2 and r>2 not yet available in arqmatrix')
        end
    otherwise
        disp('a higher degree for q in arqmatrix is not yet available')
end

    function res = chi(nu,beta)
       %returns the values of the Legandre chi function of order nu in exp(i*pi/(2T)*beta)
       % for even nu the imaginary part is returned Im(chi(nu,beta)) for odd nu the real part Re(chi(nu,beta)) is returned
       % Input:
       % nu ... order of the Legandre chi function (also defines imaginary or real part)
       % beta ... argument of the exp(i*.) function 
       res =0;
       beta = beta*pi/(2*T);
       sign = 1; 
       if beta>=pi/2 % use symmetry of functions to get higher convergence of sum
           beta =pi-beta;
           if mod(nu,2)==1; sign=-1; end
       elseif beta<=-pi/2
           beta = -(pi+beta);
           if mod(nu,2)==1; sign=-1; end
       end
       eta = [0:rho];
       switch nu
           case 2
                res = res - sum( (1-2.^((-2).*eta-1)).*zeta(2.*eta+2)./((2.*eta+2).*(2.*eta+3)).*(beta^3/pi^2).*(beta/pi).^(2.*eta) );
                if abs(beta)>eps % in case beta=0 this is zero
                    res = res -beta/2*( dGammaGamma(1)-dGammaGamma(2)+log(2*abs(beta)));
                end
%                 res = imag(1/2*(polylog(2,exp(1i*beta))-polylog(2,-exp(1i*beta))));
           case 3
               res = res + 7/8*zeta(3);
               res = res + sum( (1-2.^((-2).*eta-1)).*zeta(2.*eta+2)./((2.*eta+2).*(2.*eta+3).*(2.*eta+4)).*(beta^4/pi^2).*(beta/pi).^(2.*eta) );
               if abs(beta)>eps % in case beta=0 this is zero
                    res = res + (beta^2)/4*( dGammaGamma(1)-dGammaGamma(3)+log(2*abs(beta)) );
               end
               res = sign*res; 
%                res = real(1/2*(polylog(3,exp(1i*beta))-polylog(3,-exp(1i*beta))));
           case 4
               res = res + 7/8*zeta(3)*beta;
               res = res + sum( (1-2.^((-2).*eta-1)).*zeta(2.*eta+2)./((2.*eta+2).*(2.*eta+3).*(2.*eta+4).*(2.*eta+5)).*(beta^5/pi^2).*(beta/pi).^(2.*eta) );
               if abs(beta)>eps % in case beta=0 this is zero
                    res = res + beta^3/12*( dGammaGamma(1)-dGammaGamma(4)+log(2*abs(beta)) );
               end
%                res = imag(1/2*(polylog(4,exp(1i*beta))-polylog(4,-exp(1i*beta))));
           case 5
               res = res - 7/16*zeta(3)*beta^2+31/32*zeta(5);
               res = res - sum( (1-2.^((-2).*eta-1)).*zeta(2.*eta+2)./((2.*eta+2).*(2.*eta+3).*(2.*eta+4).*(2.*eta+5).*(2.*eta+6)).*(beta^6/pi^2).*(beta/pi).^(2.*eta) );
               if abs(beta)>eps % in case beta=0 this is zero
                    res = res - beta^4/48*( dGammaGamma(1)-dGammaGamma(5)+log(2*abs(beta)) );
               end
               res = sign*res;
           case 6
               res = res - 7/48*zeta(3)*beta^3+31/32*zeta(5)*beta;
               res = res - sum( (1-2.^((-2).*eta-1)).*zeta(2.*eta+2)./((2.*eta+2).*(2.*eta+3).*(2.*eta+4).*(2.*eta+5).*(2.*eta+6).*(2.*eta+7)).*(beta^7/pi^2).*(beta/pi).^(2.*eta) );
               if abs(beta)>eps % in case beta=0 this is zero
                    res = res - beta^5/240*( dGammaGamma(1)-dGammaGamma(6)+log(2*abs(beta)) );
               end
       end
    end

    function [zeta,dGammaGamma] = zeta_Gamma_values()
    %returns the values needed for the evaluation of the Legandre chi function 
    % Output: 
    % zeta ... vector of values of the zeta function zet, i.e. zeta(i)=zet(i),
    %          i=2,3,4,6,8,10,...,42
    % dGammaGamma ... vector of values for Gamma'(i)/Gamma(i) for i=0.5,2,3

    zeta = [0,pi^2/6, ... % i=2
            1.2020569031595942853997381615114499907649862923404988817922715553, ... % i=3
            pi^4/90, ... %i=4
            1.0369277551433699263313654864570341680570809195019128119741926779,...  %i=5
            pi^6/945, ... %i=6
            0,pi^8/9450, ... %i=8
            0,pi^10/93555, ... %i=10
            0,(691*pi^12)/638512875, ... %i=12
            0,(2*pi^14)/18243225, ... %i=14
            0,(3617*pi^16)/325641566250, ... %i=16
            0,(43867*pi^18)/38979295480125, ... %i=18
            0,(174611*pi^20)/1531329465290625, ... %i=20
            0,(155366*pi^22)/13447856940643125, ... %i=22
            0,(236364091*pi^24)/201919571963756521875, ... %i=24
            0,(1315862*pi^26)/11094481976030578125, ... %i=26
            0,(6785560294*pi^28)/564653660170076273671875, ... %i=28
            0,(6892673020804*pi^30)/5660878804669082674070015625, ... %i=30
            0,(7709321041217*pi^32)/62490220571022341207266406250, ... %i=32
            0,(151628697551*pi^34)/12130454581433748587292890625, ... %i=34
            0,(26315271553053477373*pi^36)/20777977561866588586487628662044921875, ... %i=36
            0,(308420411983322*pi^38)/2403467618492375776343276883984375, ... %i=38
            0,(261082718496449122051*pi^40)/20080431172289638826798401128390556640625, ... %i=40
            0,(3040195287836141605382*pi^42)/2307789189818960127712594427864667427734375]; %i=42

    dGammaGamma = [-1.963510026021423479440976332998755567193159604660434107047127253, ... %i=1/2
                   0.4227843350984671393934879099175975689578406640600764011942327651, ... %i=2
                   0.9227843350984671393934879099175975689578406640600764011942327651, ... % i=3
                   1.2561176684318004727268212432509309022911739973934097345275660984, ... %i=4
                   1.5061176684318004727268212432509309022911739973934097345275660984, ... %i=5
                   1.7061176684318004727268212432509309022911739973934097345275660984 ]; %i=6

    end

    function res = summand(n,m)
    % returns the summand for n and m of the auxiliary matrix A_ht^{r,q}
    % computaiton
    % Input:
    % n ... Integer of the summand n
    % m ... Integer of the summand m
    t = grid.t; 
    fac = [1,1,2,6,24,120,720,5040,40320]; % fac(n+1)=n!
        res = (2*T/pi)^(n+m+2)*((-1)/T)*(1i^(mod(n+m,2))/1i^(n+m))*fac(r+1)/fac(r-n+1)*fac(q+1)/fac(q-m+1)*...
                (  t(k)^(r-n)*t(l)^(q-m)*( (-1)^n*chi(n+m+2,t(l)-t(k))+chi(n+m+2,t(l)+t(k)) )...
                  -t(k-1)^(r-n)*t(l)^(q-m)*( (-1)^n*chi(n+m+2,t(l)-t(k-1))+chi(n+m+2,t(l)+t(k-1)) )...
                  -t(k)^(r-n)*t(l-1)^(q-m)*( (-1)^n*chi(n+m+2,t(l-1)-t(k))+chi(n+m+2,t(l-1)+t(k)) ) ...
                  +t(k-1)^(r-n)*t(l-1)^(q-m)*( (-1)^n*chi(n+m+2,t(l-1)-t(k-1))+chi(n+m+2,t(l-1)+t(k-1)) ) ) ;
    
    end
end