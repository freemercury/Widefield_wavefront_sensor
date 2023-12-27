%% calculate annulus or circle mask and projection matrix P
clc; clear;

% config for 15x15 annulus remote
ps_phasesize = 601;
outer_ratio = 0.95;
obs_ratio = 0.3;
desired_modes = 2:35;
num_view = 15;
type = "annulus";   % "annulus" or "circle"
save_path = 'D:\hyh\Project\LFM\data\prep_data_param\';
save_name = 'S2Zmatrix.mat';

% config for 8x8 circle local
% ps_phasesize = 601;
% outer_ratio = 1.0;
% obs_ratio = 0.3;
% desired_modes = 2:35;
% num_view = 8;
% type = "circle";   % "annulus" or "circle"
% save_path = 'E:\Project\LFM\data\prep_data_param\';
% save_name = '64_S2Zmatrix.mat';

% mask
if type == "annulus"
    [ii, jj] = meshgrid(1:ps_phasesize, 1:ps_phasesize);
    half_phase_size_big = round((ps_phasesize+1)/ 2);
    masktele_big = ones(ps_phasesize, ps_phasesize); 
    masktele_big((ii-half_phase_size_big).^2 + (jj-half_phase_size_big).^2 > (outer_ratio*(ps_phasesize-1)/2)^2) = 0;
    masktele_big((ii-half_phase_size_big).^2 + (jj-half_phase_size_big).^2 < (((ps_phasesize-1)*obs_ratio/2)^2)) = 0;
    masktele_big_nan = masktele_big;
    masktele_big_nan(masktele_big == 0) = nan;
    masktele_big_nan34 = repmat(masktele_big_nan,1,1,length(desired_modes));
    Zernike_grad = ZernikeCalc(desired_modes, ones(length(desired_modes),1),masktele_big,'ANNULUS',obs_ratio);
elseif type == "circle"
    [ii, jj] = meshgrid(1:ps_phasesize, 1:ps_phasesize);
    half_phase_size_big = round((ps_phasesize+1)/ 2);
    masktele_big = ones(ps_phasesize, ps_phasesize); 
    masktele_big((ii-half_phase_size_big).^2 + (jj-half_phase_size_big).^2 > (outer_ratio*(ps_phasesize-1)/2)^2) = 0;
    masktele_big_nan = masktele_big;
    masktele_big_nan(masktele_big == 0) = nan;
    masktele_big_nan34 = repmat(masktele_big_nan,1,1,length(desired_modes));
    Zernike_grad = ZernikeCalc(desired_modes, ones(length(desired_modes),1),masktele_big,'STANDARD',obs_ratio);
end

% calculate
[Z_x,Z_y] = gradient(Zernike_grad.*masktele_big_nan34);
Z_xs = imresize(Z_x,[num_view,num_view],'nearest');
Z_ys = imresize(Z_y,[num_view,num_view],'nearest');
for i = 1:length(desired_modes)
    tmp = Z_xs(:,:,i);
    tmp = tmp(:);
    tmp(isnan(tmp)) = [];
    Z_xs_linear(:,i) = tmp;
    tmp = Z_ys(:,:,i);
    tmp = tmp(:);
    tmp(isnan(tmp)) = [];
    Z_ys_linear(:,i) = tmp;
end
Z_STACK = cat(1,Z_xs_linear,Z_ys_linear);
ttt = Z_STACK'*Z_STACK;
ttmp = 1e-5*ones(1,length(desired_modes));
output = (ttt+diag(ttmp))\Z_STACK';

% save
mkdir(save_path);
S2Zmatrix = output;
mask = Z_xs(:,:,1);
mask(~isnan(mask)) = 1;
% save([save_path, save_name], 'S2Zmatrix','mask');





