%% phase & zernike
clear; clc;
% path
data_prep_path = 'D:\hyh\Project\LFM\data\prep_data_230407_new_1';
% config
set_start = 2;  % 1
set_stop = 2;   % 7
meta_start = 0; % 0
meta_stop = 999;  % 999
desired_indices = 1:35;
% integral config
mask = [
   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
   NaN   NaN   NaN   NaN     1     1     1     1     1     1     1   NaN   NaN   NaN   NaN
   NaN   NaN   NaN     1     1     1     1     1     1     1     1     1   NaN   NaN   NaN
   NaN   NaN     1     1     1     1     1     1     1     1     1     1     1   NaN   NaN
   NaN     1     1     1     1     1     1     1     1     1     1     1     1     1   NaN
   NaN     1     1     1     1     1   NaN   NaN   NaN     1     1     1     1     1   NaN
   NaN     1     1     1     1   NaN   NaN   NaN   NaN   NaN     1     1     1     1   NaN
   NaN     1     1     1     1   NaN   NaN   NaN   NaN   NaN     1     1     1     1   NaN
   NaN     1     1     1     1   NaN   NaN   NaN   NaN   NaN     1     1     1     1   NaN
   NaN     1     1     1     1     1   NaN   NaN   NaN     1     1     1     1     1   NaN
   NaN     1     1     1     1     1     1     1     1     1     1     1     1     1   NaN
   NaN   NaN     1     1     1     1     1     1     1     1     1     1     1   NaN   NaN
   NaN   NaN   NaN     1     1     1     1     1     1     1     1     1   NaN   NaN   NaN
   NaN   NaN   NaN   NaN     1     1     1     1     1     1     1   NaN   NaN   NaN   NaN
   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
];
weight=cat(3,mask,mask);
weight(isnan(weight)) = 0;
big_weight = imresize(weight(2:14,2:14,:),3,'nearest');
big_weight = imresize(big_weight,165/39,'cubic');
% figure(); imshow(big_weight(:,:,1),[]);
% upsampling config
phase_size = 165;
half_phase_size = (phase_size + 1) / 2;
% inner mask
inner_mask = ones(phase_size, phase_size); 
obs_ratio = 0.3 * 15 / 13;
outer_ratio = 1.0;
[ii, jj] = meshgrid(1:phase_size, 1:phase_size);
inner_mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 < ((obs_ratio * half_phase_size)^2)) = nan;
inner_mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 > (outer_ratio * half_phase_size)^2) = nan;
% mask
mask = ones(phase_size, phase_size); 
[ii, jj] = meshgrid(1:phase_size, 1:phase_size);
mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 > (outer_ratio * half_phase_size)^2) = 0;
mask((ii-half_phase_size).^2 + (jj-half_phase_size).^2 < ((obs_ratio * half_phase_size)^2)) = 0;
% calc phase
for set_id = set_start:set_stop
    set_path = [data_prep_path, sprintf('/%d',set_id)];
    for meta_id = meta_start:meta_stop
        % pause
        file_path = [set_path, sprintf('/shiftmap%d.mat',meta_id)];
        % load & adjust
        shiftmap = load(file_path).shiftmap; % (15,15,19,25,2)
        % mesh for integral
        [xx, yy] = meshgrid(linspace(-1,1,phase_size),linspace(-1,1,phase_size));
        zernike_full = zeros(length(desired_indices), size(shiftmap,3), size(shiftmap,4));
        % calc
        tic();
        for idx = 1:size(shiftmap,3)
            for idy = 1:size(shiftmap,4)
                waveShape = double(squeeze(shiftmap(:,:,idx,idy,:))).* weight;
                waveShape = waveShape(2:14,2:14,:);
                % expand
                waveShape_expand = imresize(waveShape,3,'nearest');
                calcu_dephase = imresize(waveShape_expand,165/39,'cubic');
                % integral to phase    
                calcu_phase = mask.*sli2q(double(calcu_dephase(:,:,2).*inner_mask), double(calcu_dephase(:,:,1).*inner_mask), xx, yy);
                % rephase & zernike
                temp_res = ZernikeCalc(desired_indices, calcu_phase, mask, 'ANNULUS', obs_ratio, []);
                zernike_full(:,idx,idy) = squeeze(temp_res);
                fprintf("set%d, meta%d, patch%d-%d\n", set_id, meta_id, idx, idy);
            end
        end
        zernike = zernike_full * -3638 * 2.34;
        save([set_path, sprintf('/zernike_full%d.mat',meta_id)], 'zernike');
        fprintf("%.4f sec\n", toc());
    end
end




