%% calculate annulus or circle mask and projection matrix P
clc; clear;
save_path = '.\';
save_name = 'valid_views.mat';

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

S2Zmatrix = zeros(15,15);
save([save_path, save_name], 'mask');


