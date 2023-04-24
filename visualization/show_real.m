%% plot color pics
clear; clc;
close all
load('F:\code2\MST-main\MST-main\visualization\real_results\results\meas28\9stg\59\0.mat');
% load('scene01.mat');
x_result_1 = flip(flip(res,1),2);
% x_result_2 = flip(flip(squeeze(res(2, :, :, :)),1),2);
% x_result_3 = flip(flip(squeeze(res(3, :, :, :)),1),2);
% x_result_4 = flip(flip(squeeze(res(4, :, :, :)),1),2);
% x_result_5 = flip(flip(squeeze(res(5, :, :, :)),1),2);
% x_result_1 = flip(flip(squeeze(x(1, :, :, :)),1),2);
% x_result_2 = flip(flip(squeeze(x(2, :, :, :)),1),2);
% x_result_3 = flip(flip(squeeze(pred(3, :, :, :)),1),2);
% x_result_4 = flip(flip(squeeze(pred(4, :, :, :)),1),2);
% x_result_5 = flip(flip(squeeze(pred(5, :, :, :)),1),2);

save_file = 'real_results\rgb_results\meas28\mlp9stg\100_2\';
mkdir(save_file);

frame = 1;
for recon = {x_result_1}
    recon = cell2mat(recon);
    intensity = 2;
    h=figure;
    for channel=1:28
        img_nb = [channel];  % channel number
        row_num = 1; col_num = 1;
        lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0 516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0 614.5 625.0 636.5 648.0];
%         lam28 = [476.5 492.5 584.5 614.5];
        recon(find(recon>1))=1;
        name = [save_file 'frame' num2str(frame) 'channel' num2str(channel)];
        dispCubeAshwin(h,recon(:,:,img_nb),intensity,lam28(img_nb), [] ,col_num,row_num,0,1,name);
    end
    frame = frame+1;
end
close all;