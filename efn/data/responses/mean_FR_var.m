% log gaussian cox process

num_neurons = 83;
num_oris = 12;

means = zeros(num_neurons, num_oris);
vars = zeros(num_neurons, num_oris);

log_means = zeros(num_neurons, num_oris);
log_vars = zeros(num_neurons, num_oris);

for i = 1:num_neurons
    for j = 1:num_oris
        fname = sprintf('spike_counts_neuron%d_ori%d.mat', i, j);
        M = load(fname);
        x = M.x;
        % FRs
        meanx = mean(x, 2);
        trial_FRs = meanx;
        means(i,j) = mean(trial_FRs);
        vars(i,j) = var(trial_FRs);
        % log FRs
        meanx = meanx(meanx ~= 0);
        trial_log_FRs = log(meanx);
        log_means(i,j) = mean(trial_log_FRs);
        log_vars(i,j) = var(trial_log_FRs);
    end
end

fprintf('dataset mean mean: %.4f\n', mean(mean(means)));
fprintf('dataset mean var: %.4f\n', mean(mean(vars)));

fprintf('dataset mean log mean: %.4f\n', mean(mean(log_means)));
fprintf('dataset mean log var: %.4f\n', mean(mean(log_vars)));
