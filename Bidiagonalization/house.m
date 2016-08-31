% --- Householder vector computation

% --- This is algorithm 5.1.1 page 210 of Golub and Van Loan III Ed.

function [v, beta] = house(x)

n = length(x);          
essential = x(2 : n);                           % --- "Essential" part of v   
sigma     = essential.' * essential;            % --- norm(x(2:n))
v         = [1; x(2 : n)];
if sigma == 0
    beta = 0;
else 
    u = sqrt((x(1))^2 + sigma);                 % --- ||x||
    if x(1) <= 0
       v(1) = x(1) - u;
    else 
       v(1) = -sigma / (x(1) + u);              % --- Implement Parlett formula 
end
beta = 2 * v(1)^2 / (sigma + v(1)^2);
v    = v / v(1);
end
