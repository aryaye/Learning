function [classifier_table] = bayes_classifier()
%Given the number x, out put whether it belongs to object or background
% the prior is given by two matrix
% class=0 indicate object object, class=1 indicate background
load TrainingSamplesDCT_8.mat *
indexBG = reduce_dimension(TrainsampleDCT_BG);
indexFG = reduce_dimension(TrainsampleDCT_FG);
probBGY = tabulate(indexBG(:));     %prob(X\Y)
probFGY = tabulate(indexFG(:)); 
sizeBG = length(indexBG);
sizeFG = length(indexFG);
probBG = sizeBG/(sizeBG + sizeFG);    %prob(Y)
probFG = sizeFG/(sizeBG + sizeFG);
classifier_table = zeros(1,64);
for x = 1:64
    pbg = 0;
    pfg = 0;
    for i = 1:length(probBGY)
        if probBGY(i,1) == x
            pbg = probBGY(i,3)*probBG/100;
        end
    end
    
    for i = 1:length(probFGY)
        if probFGY(i,1) == x
            pfg = probFGY(i,3)*probFG/100;
        end
    end
    
    if pbg > pfg
        classifier_table(x) = 0;
    end
    if pbg < pfg
        classifier_table(x) = 1;
    end
    if pbg == pfg
        classifier_table(x) = 0;
    end
end
end


