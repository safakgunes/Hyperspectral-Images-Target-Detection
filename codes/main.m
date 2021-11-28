close all;
clear;
clc;
start = cputime;
%% dataseti iceri al ve normalize hale getir
load airport.mat;
M = data;
normalized_m= (M-min(min(min(M))))/(max(max(max(M)))-min(min(min(M))));
M = normalized_m;
figure;subplot(3,3,1);imagesc(M(:,:,30));title('Hyperspectral Image');
axis image; 

[h, w, p] = size(M);
%% spektral imzayi hesapla
target = squeeze(M(45,42,:));
M = hyperConvert2d(M);
%% Constrained Energy Minimization (CEM)
r = hyperCem(M, target);
out_1 = hyperConvert3d(r, h, w, 1);
subplot(3,3,2); imagesc(abs(out_1)); title('CEM Detector Results');
axis image;
%% Adaptive Cosine Estimator (ACE)
r = hyperAce(M, target);
out_2 = hyperConvert3d(r, h, w, 1);
subplot(3,3,3); imagesc(out_2); title('ACE Detector Results'); 
axis image;
%% Signed Adaptive Cosine Estimator (S-ACE)
r = hyperSignedAce(M, target);
out_3 = hyperConvert3d(r, h, w, 1);
subplot(3,3,4);imagesc(out_3); title('Signed ACE Detector Results'); 
axis image;
%% Matched Filter
r = hyperMatchedFilter(M, target);
out_4 = hyperConvert3d(r, h, w, 1);
subplot(3,3,5);imagesc(out_4); title('MF Detector Results'); 
axis image;    
%% Generalized Likehood Ratio Test (GLRT) detector
r = hyperGlrt(M, target);
out_5 = hyperConvert3d(r, h, w, 1);
subplot(3,3,6);imagesc(out_5); title('GLRT Detector Results'); 
axis image;
%% Estimate background endmembers
U = hyperAtgp(M, 5);
%% Hybrid Unstructured Detector (HUD)
r = hyperHud(M, U, target);
out_6 = hyperConvert3d(r, h, w, 1);
subplot(3,3,7);imagesc(abs(out_6)); title('HUD Detector Results'); 
axis image;
%% Adaptive Matched Subspace Detector (AMSD)
r = hyperAmsd(M, U, target);
out_7 = hyperConvert3d(r, h, w, 1);
subplot(3,3,8); imagesc(abs(out_7)); title('AMSD Detector Results'); 
axis image;
% mesh(out_7); title('AMSD Detector Results');
%% Orthogonal Subspace Projection (OSP)
r = hyperOsp(M, U, target);
out_8 = hyperConvert3d(r, h, w, 1);
subplot(3,3,9);imagesc(abs(out_8)); title('OSP Detector Results'); 
axis image;    
%% plots  
[pd1,fa1] = hyperRoc(out_1);
[pd2,fa2] = hyperRoc(out_2);
[pd3,fa3] = hyperRoc(out_3);
[pd4,fa4] = hyperRoc(out_4);
[pd5,fa5] = hyperRoc(out_5);
[pd6,fa6] = hyperRoc(out_6);
[pd7,fa7] = hyperRoc(out_7);
[pd8,fa8] = hyperRoc(out_8);
figure; 
hold on;
plot(fa1,pd1);
plot(fa2,pd2);
plot(fa3,pd3);
plot(fa4,pd4);
plot(fa5,pd5);
plot(fa6,pd6);
plot(fa7,pd7);
plot(fa8,pd8);
legend('CEM','ACE','S-ACE','SMF','GLRT','HUD','AMSD','OSP');
figure;
subplot(3,3,1);
imshow(normalized_m(:,:,30),[]);title('Original Data')
subplot(3,3,2);
imshow(out_1(:,:,1)>0.3,[]);title('CEM')
subplot(3,3,3);
imshow(out_2(:,:,1)>0.1,[]);title('ACE')
subplot(3,3,4);
imshow(out_3(:,:,1)>0.1,[]);title('S-ACE')
subplot(3,3,5);
imshow(out_4(:,:,1)>0.25,[]);title('SMF')
subplot(3,3,6);
imshow(out_5(:,:,1)>0.1,[]);title('GLRT')
subplot(3,3,7);
imshow(out_6(:,:,1)>0.15,[]);title('HUD')
subplot(3,3,8);
imshow(out_7(:,:,1)>0.9,[]);title('AMSD')
subplot(3,3,9);
imshow(out_8(:,:,1)>0.6,[]);title('OSP')
    
 stop = cputime;
disp(['target detection is finished in ' num2str(stop-start) 'seconds.']);

