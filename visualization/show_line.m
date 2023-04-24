%% plot color pics
clear; clc;

load(['simulation_results\results\','lambda_net','.mat']);
pred_block_lambda = pred;

load(['simulation_results\results\','cst_plus','.mat']);
pred_block_cst_plus = pred;

load(['simulation_results\results\','dau_9stg','.mat']);
pred_block_dauhst_9stg = pred;

load(['simulation_results\results\','mst_l','.mat']);
pred_block_mst_l = pred;

load(['simulation_results\results\','tsanet','.mat']);
pred_block_tsanet = pred;
load(['simulation_results\results\','dgsmp','.mat']);
pred_block_dgsmp = pred;
load(['simulation_results\results\','mfmlp_9stg','.mat']);
pred_block_mfmlp9stg = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

truth(find(truth>0.7))=0.7;
pred_block_lambda(find(pred_block_lambda>0.7))=0.7;
pred_block_cst_plus(find(pred_block_cst_plus>0.7))=0.7;
pred_block_dauhst_9stg(find(pred_block_dauhst_9stg>0.7))=0.7;
pred_block_mst_l(find(pred_block_mst_l>0.7))=0.7;
pred_block_tsanet(find(pred_block_tsanet>0.7))=0.7;
pred_block_dgsmp(find(pred_block_dgsmp>0.7))=0.7;
pred_block_mfmlp9stg(find(pred_block_mfmlp9stg>0.7))=0.7;
pred_block_mfmlp11stg(find(pred_block_mfmlp11stg>0.7))=0.7;
pred_block_mfmlp_plus(find(pred_block_mfmlp_plus>0.7))=0.7;

f = 9; %frame number

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(squeeze(truth(f, :, :, 18)));
rect2crop=round(rect2crop);
close(123);

figure; 

spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_lambda = mean(mean(squeeze(pred_block_lambda(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_cst_plus = mean(mean(squeeze(pred_block_cst_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dauhst_9stg = mean(mean(squeeze(pred_block_dauhst_9stg(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_l = mean(mean(squeeze(pred_block_mst_l(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_tsanet = mean(mean(squeeze(pred_block_tsanet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dgsmp = mean(mean(squeeze(pred_block_dgsmp(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mfmlp9stg = mean(mean(squeeze(pred_block_mfmlp9stg(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);


spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_lambda = spec_mean_lambda./max(spec_mean_lambda);
spec_mean_cst_plus = spec_mean_cst_plus./max(spec_mean_cst_plus);
spec_mean_dauhst_9stg = spec_mean_dauhst_9stg./max(spec_mean_dauhst_9stg);
spec_mean_mst_l = spec_mean_mst_l./max(spec_mean_mst_l);
spec_mean_tsanet = spec_mean_tsanet./max(spec_mean_tsanet);
spec_mean_dgsmp = spec_mean_dgsmp./max(spec_mean_dgsmp);
spec_mean_mfmlp9stg = spec_mean_mfmlp9stg./max(spec_mean_mfmlp9stg);


corr_lambda = roundn(corr(spec_mean_truth(:),spec_mean_lambda(:)),-4);
corr_cst_plus = roundn(corr(spec_mean_truth(:),spec_mean_cst_plus(:)),-4);
corr_dauhst_9stg = roundn(corr(spec_mean_truth(:),spec_mean_dauhst_9stg(:)),-4);
corr_mst_l = roundn(corr(spec_mean_truth(:),spec_mean_mst_l(:)),-4);
corr_tsanet = roundn(corr(spec_mean_truth(:),spec_mean_tsanet(:)),-4);
corr_dgmsp = roundn(corr(spec_mean_truth(:),spec_mean_dgsmp(:)),-4);
corr_mfmlp9stg = roundn(corr(spec_mean_truth(:),spec_mean_mfmlp9stg(:)),-4);


X = lam28;

Y(1,:) = spec_mean_truth(:); 
Y(2,:) = spec_mean_lambda(:); Corr(1)=corr_lambda;
Y(3,:) = spec_mean_cst_plus(:); Corr(2)=corr_cst_plus;
Y(4,:) = spec_mean_dauhst_9stg(:); Corr(3)=corr_dauhst_9stg;
Y(5,:) = spec_mean_mst_l(:); Corr(4)=corr_mst_l;
Y(6,:) = spec_mean_tsanet(:); Corr(5)=corr_tsanet;
Y(7,:) = spec_mean_dgsmp(:); Corr(6)=corr_dgmsp;
Y(8,:) = spec_mean_mfmlp9stg(:); Corr(7)=corr_mfmlp9stg;


createfigure(X,Y,Corr)



