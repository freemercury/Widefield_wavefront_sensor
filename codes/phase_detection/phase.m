%% slope to zernike
clear; clc;

% config
data_path = 'E:\Project\Widefield_wavefront_sensor\data\phase_data\230407\set1';
mask_path = 'E:\Project\Widefield_wavefront_sensor\data\settings\mask.mat';
desired_indices = 1:35; % zernike orders (Noll)
crop_bound = 1;     % remove the outermost band (width = crop_bound)
expand = 3;         % used in nearest up-sampling
phase_size = 165;   % size of phase for fitting
obs_ratio = 0.3;    % obs ratio for annular shape
outer_ratio = 1.0;  % outer ratio for annular shape 
factor = -3638 * 2.34;  % adjust the scale, need calibration

% masks
mask = load(mask_path).mask;
weight=cat(3,mask,mask);
weight(isnan(weight)) = 0;
half_phase_size = (phase_size + 1) / 2;
inner_mask = ones(phase_size, phase_size); 
obs_ratio = obs_ratio * size(mask,1) / (size(mask,1) - 2 * crop_bound);
[ii, jj] = meshgrid(1:phase_size, 1:phase_size);
inner_mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 < ((obs_ratio * half_phase_size)^2)) = nan;
inner_mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 > (outer_ratio * half_phase_size)^2) = nan;
mask = ones(phase_size, phase_size); 
mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 > (outer_ratio * half_phase_size)^2) = 0;
mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 < ((obs_ratio * half_phase_size)^2)) = 0;

% files
filePattern = fullfile(data_path, '\*_slope.mat');
files = {dir(filePattern).name};
for i = 1:length(files)
    slopemap = load([data_path, '\', files{i}]).slope;
    [xx, yy] = meshgrid(linspace(-1,1,phase_size),linspace(-1,1,phase_size));
    zernike = zeros(length(desired_indices), size(slopemap,3), size(slopemap,4));
    for idx = 1:size(slopemap,3)
        for idy = 1:size(slopemap,4)
            waveShape = double(squeeze(slopemap(:,:,idx,idy,:))).* weight;
            waveShape = waveShape(1+crop_bound:end-crop_bound,1+crop_bound:end-crop_bound,:);
            waveShape_expand = imresize(waveShape,expand,'nearest');
            calcu_dephase = imresize(waveShape_expand,phase_size/size(waveShape_expand,1),'cubic');
            calcu_phase = mask.*sli2q(double(calcu_dephase(:,:,2).*inner_mask), double(calcu_dephase(:,:,1).*inner_mask), xx, yy);
            temp_res = ZernikeCalc(desired_indices, calcu_phase, mask, 'ANNULUS', obs_ratio, []);
            zernike(:,idx,idy) = squeeze(temp_res);
            fprintf("%s, patch%d-%d\n", files{i}, idx, idy);
        end
    end
    zernike = zernike * factor;
    save([data_path, '\', replace(files{i}, 'slope', 'zernike')], 'zernike');
end

%% projection matrix
clear; clc;

% config
save_path = 'E:\Project\Widefield_wavefront_sensor\data\settings';
phase_size = 75;
desired_indices = 1:35; % zernike orders (Noll)
n_zernike = length(desired_indices);

% processing
Z2P = zeros(n_zernike, phase_size, phase_size);
for i = 1:n_zernike
    zernike_coefs = zeros(n_zernike,1);
    zernike_coefs(i,:) = 1.0;
    [rephase, ~] = ZernikeCalc(desired_indices, zernike_coefs, phase_size, 'NOLL');
    Z2P(i,:,:) = rephase;
end
ids = find(squeeze(Z2P(1,:,:)));
Z2p = zeros(35, size(ids, 1));
for i = 1:35
    temp_p = Z2P(i,:,:);
    Z2p(i,:) = temp_p(ids);
end
p2Z = (inv(Z2p * Z2p') * Z2p)';
save([save_path, '\', sprintf('zernike_phase%d.mat',phase_size)], 'Z2P', 'Z2p', 'p2Z');







