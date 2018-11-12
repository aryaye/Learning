load('TrainingSamplesDCT_8_new.mat')
% ML estimate for conditional Prob of background case and front case
[u_BG,omiga_BG] = ML_MGaussian(abs(TrainsampleDCT_BG));
[u_FG,omiga_FG] = ML_MGaussian(abs(TrainsampleDCT_FG));
%chose 8 features from 64 degree vectors
%{
x = linspace(-0.2,0.2);
for i = 1:64
    if mod(i,4)==1
        figure(ceil(i/4))       
    end
    subplot(2,2,i-ceil(i/4)*4+4)
%    histogram(TrainsampleDCT_BG(:,i),'Normalization','probability')
    plot(x,normpdf(x,u_BG(i),omiga_BG(i,i)),'-.r')
    hold on
    plot(x,normpdf(x,u_FG(i),omiga_FG(i,i)),'-.g')
 %   histogram(TrainsampleDCT_FG(:,i),'normalization','probability')
end
%}
%features = [1,7,12,14,15,17,18,19];
features = [1,7,12,14,15,17,18,16];
% we chose x1,x7,x12, x14,x15,x17,x18,x19 as training feature
FG_data = abs(TrainsampleDCT_FG(:,features));
BG_data = abs(TrainsampleDCT_BG(:,features));

img = im2double(imread('cheetah.bmp'));
img_mask = im2double(imread('cheetah_mask.bmp'));
[m,n] = size(img);
mask_all = zeros(m,n);
mask_8 = zeros(m,n);
img(m+7,n+7)=0;
[u_BG_8,omiga_BG_8] = ML_MGaussian(BG_data);
[u_FG_8,omiga_FG_8] = ML_MGaussian(FG_data);
iomiga_bg_8 = inv(omiga_BG_8);
iomiga_fg_8 = inv(omiga_FG_8);
iomiga_bg = inv(omiga_BG);
iomiga_fg = inv(omiga_FG);
p1 = length(FG_data)/(length(BG_data)+length(FG_data));
p2 = length(BG_data)/(length(BG_data)+length(FG_data));

for i = 1:m
    for j = 1:n
        blocks = img(i:i+7,j:j+7);
        DCT = abs(dct2(blocks));
        x = zigzag_scan(DCT);
        x_8 = x(features);
        mask_all(i,j) = classifier_Gaussian(x, u_FG, iomiga_fg, u_BG, iomiga_bg, p1, p2);
        mask_8(i,j) = classifier_Gaussian(x_8, u_FG_8, iomiga_fg_8, u_BG_8, iomiga_bg_8, p1, p2);
    end
end
subplot(1,2,1)
imshow(mask_all)
subplot(1,2,2)
imshow(mask_8)

sumf = 0;
sumb = 0;
errf = 0;
errb = 0;
for i = 1:m
    for j = 1:n
        if img_mask(i,j) == 0
            sumf = sumf + 1;
            if mask_8(i,j) == 1
                errf = errf + 1;
            end
        end
        if img_mask(i,j) == 1
            sumb = sumb + 1;
            if mask_8(i,j) == 0
                errb = errb + 1;
            end
        end 
    end
end
    
p2*errf/sumf + p1*errb/sumb
