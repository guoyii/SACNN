close all;  
clear all;
clc; 
data_path = 'V:/users/gy/MyProject/WGAN_SACNN_AE/results/v7/image.mat';
image = load(data_path);
full_image = permute(image.('full_image'),[2,3,1]);
quarter_image = permute(image.('quarter_image'),[2,3,1]);
pred_image = permute(image.('pred_image'),[2,3,1]);

imtool(full_image(:,:,1), [0, 2500]);
imtool(quarter_image(:,:,1), [0, 2500]);
imtool(pred_image(:,:,1), [0, 2500]);

%imtool(full_image(:,:,1)-quarter_image(:,:,1), [0, 100]);
%imtool(full_image(:,:,1)-pred_image(:,:,1), [0, 100]);

