% --- Bidiagonalization performed according to the Golub-Van Loan scheme

% --- This is Algorithm 5.4.2 page 252 of Golun and Van Loan III Ed.

% --- Assumption: m <= n
% --- A = UB * B * VB.'

function [B, UB, VB] = Bidiagonalization(A)

m   = size(A, 1);
n   = size(A, 2);
UB  = eye(m);
VB  = eye(n);

if (m == n) 
    maxIter = n - 1;
else
    maxIter = n;
end

for j = 1 : maxIter
    [v, beta] = house(A(j : m, j));
    L               = eye(m - j + 1) - beta * v * v.';          % --- Saving eye(m - j + 1) - beta * v * v.' into L is necessary for left-hand side accumulation
    A(j : m, j : n) = L * A(j : m, j : n);
    
    % --- Left-hand side accumulation
    N   = zeros(j - 1, m - j + 1);
    P   = [eye(j - 1) N; N.' L];
    UB  = UB * P;        
    
    if j <= n - 2
        [v, beta] = house(A(j, j + 1 : n).');
        R                   = eye(n - j) - beta * v * v.';      % --- Saving eye(n - j) - beta * v * v.' into R is necessary for right-hand side accumulation
        A(j : m, j + 1 : n) = A(j : m, j + 1 : n) * R;
        
        % --- Right-hand side accumulation
        N1 = zeros(j, n - j);
        P1 = [eye(j) N1; N1.' R];
        VB = VB * P1;
    end 
end 

B = A;
end 
