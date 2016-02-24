clear;

% classes = {'apple', 'burger', 'coffee', 'donuts', 'french_fries', 'fried_rice', 'ice_cream', 'ramen', 'salad' 'sashimi'};
classes = {'apple', 'burger', 'coffee'};

% Plot data
% figure();
% boxplot(F_Train,'orientation','horizontal','labels',classes);

correlation = corr(F_Train, F_Train);

[wcoeff,score,latent,tsquared,explained] = pca(F_Train,...
'VariableWeights','variance');

c3 = wcoeff(:,1:3);

coefforth = inv(diag(std(F_Train)))*wcoeff;

I = c3'*c3;

cscores = zscore(F_Train)*coefforth;

figure()
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

metro = [927 354 1225 1080 1466 1391 1126];

figure()
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

[st2,index] = sort(tsquared,'descend'); % sort in descending order
extreme = index(1);