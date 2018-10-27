function [answer] = classifier_Gaussian(x, u1, omiga1, u2, omiga2, p1, p2)
% there are two class y, x follows Gaussian distrubition give y 
% answer is the class x most likely belongs to
% 1 indicate fronter, 0 indicate background
d = length(x);
pyx1 = p1*exp(-0.5*(x-u1)*omiga1*(x-u1)')*det(omiga1)^(d/2)/(2*pi)^(d/2);
pyx2 = p2*exp(-0.5*(x-u2)*omiga2*(x-u2)')*det(omiga2)^(d/2)/(2*pi)^(d/2);
if pyx1 > pyx2
    answer = 1;
else
    if pyx1 < pyx2
        answer = 0;
    else
        %answer = round(rand(1));
        answer = 0;
    end
end   
end

