function [u, omiga] = ML_MGaussian(data)
%ML_MGaussian 
%u and omiga is parameter of MultiGaussian distribution 
u = sum(data,1)/length(data);
omiga = 0;
for i = 1:length(data)
    omiga = omiga + data(i,:)'*data(i,:);
end
omiga = omiga/length(data);
end

