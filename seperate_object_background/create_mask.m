img = im2double(imread('cheetah.bmp'));
[m,n] = size(img);
mask = zeros(m,n);
img(m+7,n+7)=0;
class_table = bayes_classifier();
M = zeros(m,n);
img_mask = im2double(imread('cheetah_mask.bmp'));
pfsum = 0;
pbsum = 0;
fsum = 0;
bsum = 0;
for i = 1:m
    for j = 1:n
        blocks = img(i:i+7,j:j+7);
        DCT = abs(dct2(blocks));
        x = zigzag_scan(DCT);
        [~, index] = sort(x, 'descend');
        M(i,j) = index(2);
        mask(i,j) = class_table(index(2));
        if img_mask(i,j) == 1
            if mask(i,j) == 1
                pfsum = pfsum + 1;
            end
            fsum = fsum + 1;
        end
        if img_mask(i,j) == 0
            if mask(i,j) == 0
                pbsum = pbsum + 1;
            end
            bsum = bsum + 1;
        end
    end
end
subplot(1,3,3)
imshow(mask)
PoE = (1-pbsum/bsum)*0.8081 + (1-pfsum/fsum)*0.1919;


