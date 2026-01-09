load ('Users_Data.mat');


% Converting table to array:
Array_d = table2array(User_d);


% --------------------------------------------------------------------------------------------
% Removing the third class (2):

mat_ddd = (Array_d(:,1) > 1);


% Check if there are any rows to remove
if any(mat_ddd)
    % Remove the rows
    Matrix_d = User_d(~mat_ddd,:);
end

Matrix_d = table2array(Matrix_d);


% ---------------------------------------------------------------------------------------------


% Epoching 

% Initialize empty array for delta data
delta_data_d = [];
theta_data_d = [];
alpha_data_d = [];
beta_data_d = [];

% Set number of channels (14x2) "2 Features for the 14 channels"
num_channels = 28;

% Set indices of delta columns
delta_col_indices = [2 3 10 11 18 19 26 27 34 35 42 43 50 51 58 59 66 67 74 75 82 83 90 91 98 99 106 107];

% Set indices of theta columns
theta_col_indices = [4 5 12 13 20 21 28 29 36 37 44 45 52 53 60 61 68 69 76 77 84 85 92 93 100 101 108 109];

% Set indices of alpha columns
alpha_col_indices = [6 7 14 15 22 23 30 31 38 39 46 47 54 55 62 63 70 71 78 79 86 87 94 95 102 103 110 111];

% Set indices of beta columns
beta_col_indices = [8 9 16 17 24 25 32 33 40 41 48 49 56 57 64 65 72 73 80 81 88 89 96 97 104 105 112 113];


% Iterate over rows of Matrix_d in 160-sample intervals
for i = 1:160:size(Matrix_d,1)-159
% Extract 1 second of data for delta waves from Matrix_d
delta_data_second = []; % initialize empty array for delta data of current second
for j = delta_col_indices % iterate over columns containing delta data
    delta_data_second = [delta_data_second Matrix_d(i:i+159, j)']; % select rows i to i+159 and column j, transpose, and concatenate horizontally
end
delta_data_d = [delta_data_d; delta_data_second]; % concatenate vertically to add current second of delta data to the array

% Extract 1 second of data for theta waves from Matrix_d
theta_data_second = []; % initialize empty array for theta data of current second
for j = theta_col_indices % iterate over columns containing theta data
    theta_data_second = [theta_data_second Matrix_d(i:i+159, j)']; % select rows i to i+159 and column j, transpose, and concatenate horizontally
end
theta_data_d = [theta_data_d; theta_data_second]; % concatenate vertically to add current
% Extract 1 second of data for alpha waves from Matrix_d
alpha_data_second = []; % initialize empty array for alpha data of current second
for j = alpha_col_indices % iterate over columns containing alpha data
    alpha_data_second = [alpha_data_second Matrix_d(i:i+159, j)']; % select rows i to i+159 and column j, transpose, and concatenate horizontally
end
alpha_data_d = [alpha_data_d; alpha_data_second]; % concatenate vertically to add current second of alpha data to the array

% Extract 1 second of data for beta waves from Matrix_d
beta_data_second = []; % initialize empty array for beta data of current second
for j = beta_col_indices % iterate over columns containing beta data
    beta_data_second = [beta_data_second Matrix_d(i:i+159, j)']; % select rows i to i+159 and column j, transpose, and concatenate horizontally
end
beta_data_d = [beta_data_d; beta_data_second]; % concatenate vertically to add current second of beta data to the array
end


% ---------------------------------------------------------------------------

% Scaling data:

% Scale the data using normalize
scaled_delta_data_d = normalize(delta_data_d);
scaled_theta_data_d = normalize(theta_data_d);
scaled_alpha_data_d = normalize(alpha_data_d);
scaled_beta_data_d = normalize(beta_data_d);



% ---------------------------------------------------------------------------

% Initialize empty array for labels
Labels = [];

% concatenate the scaled data together of the 4 waves toghether horizontally:
Waves = [scaled_delta_data_d scaled_theta_data_d scaled_alpha_data_d scaled_beta_data_d];


% Creating the Labels matrix:

% Iterate over rows of Matrix_d in 160-sample intervals
for i = 1:160:size(Matrix_d,1)-159
Labels = [Labels Matrix_d(i, 1)]; % add value of the first column at row i to the array
end
Labels = Labels.';

WavesWithLabels = [ Waves Labels ];



% ---------------------------------------------------------------------------

% Randomization:

% Shuffling the array with labels added to the 2 subs
shuffledArray = WavesWithLabels(randperm(size(WavesWithLabels,1)),:);


% Take the shuffled labels aside
Shuffledlabels = shuffledArray(:,end);


% Take the labels out
shuffledArrayNoLabels = shuffledArray(:,1:end-1);


% ---------------------------------------------------------------------------

% Splitting data:

% Splitting data into 50% train and 50% test
cv = cvpartition(size(shuffledArrayNoLabels,1),'HoldOut',0.5);
idx = cv.test;

dataTrain = shuffledArrayNoLabels(~idx,:);
dataTest  = shuffledArrayNoLabels(idx,:);


% Splitting Labels into 50% train and 50% test
cv = cvpartition(size(Shuffledlabels,1),'HoldOut',0.5);
idx = cv.test;

LabelsTrain = Shuffledlabels(~idx,:);
LabelsTest  = Shuffledlabels(idx,:);



% ---------------------------------------------------------------------------

% Classification:

% Creating the diaglinear classifier
class = classify(dataTest,dataTrain,LabelsTrain,'diaglinear');


% Measuring accuracy
Correct_Labels = 0 ;

for i=1:length(class)
    if class(i) == LabelsTest(i)
        Correct_Labels = Correct_Labels + 1 ;    
    end
end
    diagLinear_Accuracy = (Correct_Labels/length(class))*100;
    
    
% Make predictions using the diaglinear classifier
predictions_diag = class;

% Calculate performance measures
[Diagaccuracy, Diagsensitivity, Diagspecificity] = getPerformanceMeasures(predictions_diag, LabelsTest, 'DiagLinear');

% Print the results
fprintf('DiagLinear Model Performance:\n');
fprintf('Accuracy_diaglinear: %.2f%%\n', Diagaccuracy*100);
fprintf('Sensitivity_diaglinear: %.2f%%\n', Diagsensitivity*100);
fprintf('Specificity_diaglinear: %.2f%%\n', Diagspecificity*100);

    
%----------------------------------------------------------------------------------

% KNN classification model 

% Train KNN classifier
knnModel = fitcknn(dataTrain, LabelsTrain, 'NumNeighbors', 5);

% Make predictions on test data
predictions_KNN = predict(knnModel, dataTest);

% Calculate performance measures
[knnAccuracy, knnSensitivity, knnSpecificity] = getPerformanceMeasures(predictions_KNN, LabelsTest, 'KNN');

% Print the results
fprintf('KNN Model Performance:\n');
fprintf('Accuracy_KNN: %.2f%%\n', knnAccuracy*100);
fprintf('Sensitivity_KNN: %.2f%%\n', knnSensitivity*100);
fprintf('Specificity_KNN: %.2f%%\n', knnSpecificity*100);



%----------------------------------------------------------------------------------
%DT classification model
% Set number of folds for stratified k-fold cross-validation
nfolds = 2;

% Generate indices for stratified k-fold cross-validation
partition = cvpartition(Labels,'KFold',nfolds, 'Stratify', true);

% Iterate over folds
for i = 1:nfolds
% Get training and test sets for current fold
idx = partition.training(i);
dataTrain = shuffledArrayNoLabels(idx,:);
LabelsTrain = Labels(idx);
idx = partition.test(i);
dataTest = shuffledArrayNoLabels(idx,:);
LabelsTest = Labels(idx);

% Train Decision Trees classifier on training data
DecisionTreeModel = fitctree(dataTrain,LabelsTrain);

% Make predictions on test data
predictions_DT = predict(DecisionTreeModel, dataTest);

% Calculate performance measures
[dtaccuracy(i), dtsensitivity(i), dtspecificity(i)] = getPerformanceMeasures(predictions_DT, LabelsTest, 'Decision Tree');
end

% Calculate mean and standard deviation of performance measures across folds
meanAccuracy_DT = mean(dtaccuracy);
meanSensitivity_DT = mean(dtsensitivity);
meanSpecificity_DT = mean(dtspecificity);

% Print the results
fprintf('DT Model Performance:\n');
fprintf('Accuracy_DT: %.2f%%\n', meanAccuracy_DT*100);
fprintf('Sensitivity_DT: %.2f%%\n', meanSensitivity_DT*100);
fprintf('Specificity_DT: %.2f%%\n', meanSpecificity_DT*100);



%----------------------------------------------------------------------------------

% Logistic classification model 

% Train logistic classifier on training data
Logisticclassifier = fitclinear(dataTrain,LabelsTrain,'Learner','logistic');

% Make predictions on test data
predictions_LOG = predict(Logisticclassifier, dataTest);

% Calculate performance measures
[dtaccuracy(i), dtsensitivity(i), dtspecificity(i)] = getPerformanceMeasures(predictions_LOG, LabelsTest, 'Logistic');


% Calculate mean and standard deviation of performance measures across folds
meanAccuracy_Logistic = mean(dtaccuracy);
meanSensitivity_Logistic = mean(dtsensitivity);
meanSpecificity_Logistic = mean(dtspecificity);

% Print the results
fprintf('Logistic Model Performance:\n');
fprintf('Accuracy_Logistic: %.2f%%\n', meanAccuracy_Logistic*100);
fprintf('Sensitivity_Logistic: %.2f%%\n', meanSensitivity_Logistic*100);
fprintf('Specificity_Logistic: %.2f%%\n', meanSpecificity_Logistic*100);
%----------------------------------------------------------------------------------

% SVM classification model 

% Train SVM classifier on training data
SVMModel = fitcsvm(dataTrain,LabelsTrain);

% Make predictions on test data
predictions_SVM = predict(SVMModel, dataTest);

% Calculate performance measures for SVM model
[svmAccuracy, svmSensitivity, svmSpecificity] = getPerformanceMeasures(predictions_SVM, LabelsTest, 'SVM');

% Display performance measures
fprintf('SVM Model Performance:\n');
fprintf('Accuracy_SVM: %.2f%%\n', svmAccuracy*100);
fprintf('Sensitivity_SVM: %.2f%%\n', svmSensitivity*100);
fprintf('Specificity_SVM: %.2f%%\n', svmSpecificity*100);


%----------------------------------------------------------------------------------
% Naive Bayes classification model 

% Train the Naive Bayes model
NBModel = fitcnb(dataTrain, LabelsTrain);

% Make predictions on the test data
predictions_NB = predict(NBModel, dataTest);

% Calculate performance measures
[nbAccuracy, nbSensitivity, nbSpecificity] = getPerformanceMeasures(predictions_NB, LabelsTest, 'Naive Bayes');

% Print the results
fprintf('NB Model Performance:\n');
fprintf('Accuracy_NB: %.2f%%\n', nbAccuracy*100);
fprintf('Sensitivity_NB: %.2f%%\n', nbSensitivity*100);
fprintf('Specificity_NB: %.2f%%\n', nbSpecificity*100);


%----------------------------------------------------------------------------------

% Results Table

% Create a cell array with the names of the models
models = {'DiagLinear', 'KNN', 'Decision Tree', 'Logistic', 'SVM', 'Naive Bayes'};

% Create a table to store the results
resultsTable = table('Size',[length(models),4], ...
'VariableTypes',{'string','double','double','double'}, ...
'VariableNames',{'Model','Accuracy','Sensitivity','Specificity'});

% Iterate over the models
for i = 1:length(models)
% Calculate the performance measures for the current model
switch models{i}
case 'DiagLinear'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_diag, LabelsTest, models{i});
case 'KNN'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_KNN, LabelsTest, models{i});
case 'Decision Tree'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_DT, LabelsTest, models{i});
case 'Logistic'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_LOG, LabelsTest, models{i});
case 'SVM'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_SVM, LabelsTest, models{i});
case 'Naive Bayes'
[accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions_NB, LabelsTest, models{i});
end

% Add the results to the table
resultsTable(i,:) = {models{i}, accuracy, sensitivity, specificity};
end

% Display the results table
disp(resultsTable);

%----------------------------------------------------------------------------------

%----------------------------------------------------------------------------------

% PLOTS:



% Plotting the whole data before and after scaling:


figure; % create a new figure window

% Set the x-axis values for the plots
x = 1:size(delta_data_d,2); % set x-axis values to be the indices of the columns in delta_data_d

% Plot the delta data before scaling
subplot(2,2,1); % create a subplot for the delta data
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(delta_data_d,1)
    % Plot the i-th row of delta_data_d
    plot(x, delta_data_d(i,:));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Delta Data Before Scaling'); % set plot title

% Plot the delta data after scaling
subplot(2,2,2); % create a subplot for the scaled delta data
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(scaled_delta_data_d,1)
    % Plot the i-th row of scaled_delta_data_d
    plot(x, scaled_delta_data_d(i,:));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Delta Data After Scaling'); % set plot title

% Plot the theta data before scaling
subplot(2,2,3); % create a sub plot for theta data
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(theta_data_d,1)
% Plot the i-th row of theta_data_d
plot(x, theta_data_d(i,:));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Theta Data Before Scaling'); % set plot title

% Plot the theta data after scaling
subplot(2,2,4); % create a subplot for the scaled theta data
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(scaled_theta_data_d,1)
% Plot the i-th row of scaled_theta_data_d
plot(x, scaled_theta_data_d(i,:));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Theta Data After Scaling'); % set plot title





%----------------------------------------------------------------------------------

% Plotting only 6 seconds of the data before and after scaling

% Plot half of the scaled delta data before scaling
figure; % create a new figure window
subplot(1,2,1); % create a subplot for the delta data before scaling
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(delta_data_d,1)
% Plot the first half of the i-th row of delta_data_d
plot(x(1:size(delta_data_d,2)/2), delta_data_d(i,1:size(delta_data_d,2)/2));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Delta Data Before Scaling'); % set plot title

% Plot half of the scaled delta data after scaling
subplot(1,2,2); % create a subplot for the scaled delta data
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(scaled_delta_data_d,1)
% Plot the first half of the i-th row of scaled_delta_data_d
plot(x(1:size(scaled_delta_data_d,2)/2), scaled_delta_data_d(i,1:size(scaled_delta_data_d,2)/2));
end
xlabel('Time (s)'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Delta Data After Scaling'); % set plot title

%----------------------------------------------------------------------------------

% Plotting the waves with their labels

% Initialize empty arrays to store the data for each label
waves0 = [];
waves1 = [];

% Iterate over the rows of WavesWithLabels
for i = 1:size(WavesWithLabels,1)
    % Get the current label
    label = WavesWithLabels(i,1);
    % Get the current wave data
    wave = WavesWithLabels(i,2:end);
    % Check the label and add the wave data to the appropriate array
    if label == 0
        waves0 = [waves0; wave];
    else
        waves1 = [waves1; wave];
    end
end

% Plot the waves with label 0
figure; % create a new figure window
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(waves0,1)
    % Plot the i-th row of waves0
    plot(waves0(i,:));
end
xlabel('Time'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Waves with Label 0'); % set plot title

% Plot the waves with label 1
figure; % create a new figure window
hold on; % keep all of the plotted lines on the same graph
for i = 1:size(waves1,1)
    % Plot the i-th row of waves1
plot(waves1(i,:));
end
xlabel('Time'); % set x-axis label
ylabel('Amplitude'); % set y-axis label
title('Waves with Label 1'); % set plot title

%----------------------------------------------------------------------------------

% FUNCTION:

function [accuracy, sensitivity, specificity] = getPerformanceMeasures(predictions, labels, modelType)
% Calculate the number of true positive predictions
truePositives = sum((predictions == 1) & (labels == 1));

% Calculate the number of true negative predictions
trueNegatives = sum((predictions == 0) & (labels == 0));

% Calculate the number of false positive predictions
falsePositives = sum((predictions == 1) & (labels == 0));

% Calculate the number of false negative predictions
falseNegatives = sum((predictions == 0) & (labels == 1));

% Calculate the accuracy
accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);

% Calculate the sensitivity
sensitivity = truePositives / (truePositives + falseNegatives);

% Calculate the specificity
specificity = trueNegatives / (trueNegatives + falsePositives);

% Store the performance measures for the current model
switch modelType
    
    case 'DiagLinear'
        diagAccuracy = accuracy;
        diagSensitivity = sensitivity;
        diagSpecificity = specificity;
    
    case 'SVM'
        svmAccuracy = accuracy;
        svmSensitivity = sensitivity;
        svmSpecificity = specificity;
    case 'KNN'
        knnAccuracy = accuracy;
        knnSensitivity = sensitivity;
        knnSpecificity = specificity;
    case 'Naive Bayes'
        nbAccuracy = accuracy;
        nbSensitivity = sensitivity;
        nbSpecificity = specificity;
    case 'Decision Tree'
        dtAccuracy = accuracy;
        dtSensitivity = sensitivity;
        dtSpecificity = specificity;
    case 'Random Forest'
        rfAccuracy = accuracy;
        rfSensitivity = sensitivity;
        rfSpecificity = specificity;
        end

end








    

    
