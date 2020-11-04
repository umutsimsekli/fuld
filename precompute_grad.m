% This code precomputes the quantites that are required for gradient computation when 1<alpha<2.
% The code requires the qastable toolbox which can be obtained from https://gitlab.com/s_ament/qastable

clear
close all
clc

alps = linspace(1,2,5);
alps = alps(2:end-1);


v_range = -100:0.0001:100;
v_range = v_range(:);


for i = 1:length(alps)
    alp = alps(i);
    grad = grad_log_sas(v_range,alp);
    fname = sprintf('./precomputed_grads/%2.4f.mat',alp);
    save(fname,'grad','v_range');
end