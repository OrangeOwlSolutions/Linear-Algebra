clear all
close all
clc

Nrows = 5000;
Ncols = 5000;

NumTests = 10;

h_A = rand(Nrows, Ncols);
d_A = gpuArray.rand(Nrows, Ncols);

timingCPU = 0;
timingGPU = 0;

for k = 1 : NumTests
    % --- Host
    tic
%     [h_U, h_S, h_V] = svd(h_A);
    h_S = svd(h_A);
    timingCPU = timingCPU + toc;

    % --- Device
    tic
%     [d_U, d_S, d_V] = svd(d_A);
    d_S = svd(d_A);
    timingGPU = timingGPU + toc;
end

fprintf('Timing CPU = %f; Timing GPU = %f\n', timingCPU / NumTests, timingGPU / NumTests);

