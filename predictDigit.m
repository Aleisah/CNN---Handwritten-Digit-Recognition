% Prediction Script for Random Group of Images Using Saved Model (5x5 Grid)

clc; % Clear command window without clearing variables

% Load the trained model (do not reload or retrain)
disp('Loading trained model...');
load('trainedDigitNet.mat', 'net');

% Load the test data (assuming dataset is already loaded and split)
disp('Loading test data...');
datasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
if ~isfolder(datasetPath)
    error('DigitDataset folder does not exist in the specified path.');
end
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the data (assuming you already did it before)
disp('Splitting dataset...');
[~, testData] = splitEachLabel(imds, 0.8, 'randomize');

% Number of random images to display (5x5 grid = 25 images)
numImagesToDisplay = 25; % Adjust for 5x5 grid
randomIndices = randperm(numel(testData.Files), numImagesToDisplay); % Randomly select image indices

% Predict random test images and display results
disp('Predicting on random test images...');
figure;
for i = 1:numImagesToDisplay
    % Read the image
    testImage = readimage(testData, randomIndices(i));
    
    % Classify the image
    predictedLabel = classify(net, testImage);
    
    % Create subplot for each image (5x5 grid)
    subplot(5, 5, i); % Display images in a 5x5 grid
    imshow(testImage);
    title(['Predicted: ', char(predictedLabel)]);
end
disp('Prediction results displayed.');
