% Handwritten Digit Recognition Using imageDatastore
clear; clc;

% Load dataset
disp('Loading dataset...');
datasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
if ~isfolder(datasetPath)
    error('DigitDataset folder does not exist in the specified path.');
end
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into training, validation, and testing data
disp('Splitting dataset...');
[trainData, testData] = splitEachLabel(imds, 0.8, 'randomize');
[trainData, valData] = splitEachLabel(trainData, 0.8, 'randomize');

% Define the neural network
disp('Defining the neural network...');
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batch1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batch2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout')
    
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Specify training options
disp('Setting training options...');
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 128, ...
    'ValidationData', valData, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the network
disp('Training the network...');
net = trainNetwork(trainData, layers, options);

% Test the network
disp('Testing the network...');
predictions = classify(net, testData);
accuracy = sum(predictions == testData.Labels) / numel(testData.Labels);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Save the trained model
disp('Saving the trained model...');
save('trainedDigitNet.mat', 'net');
disp('Model saved to trainedDigitNet.mat.');

% Predict a single test image
disp('Predicting on a single test image...');
testImage = readimage(testData, 1);
imshow(testImage);
actualLabel = testData.Labels(1);
predictedLabel = classify(net, testImage);
title(['Actual: ', char(actualLabel), ', Predicted: ', char(predictedLabel)]);
disp(['Predicted Digit: ', char(predictedLabel)]);
