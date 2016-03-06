clc
clear all

% Load Datasets

Dataset = '../FoodData/Data/Train';   
Testset  = '../FoodData/Data/Test';


% we need to process the images first.
% Convert your images into grayscale
% Resize the images

width=100; height=100;
DataSet      = cell([], 1);

 for i=1:length(dir(fullfile(Dataset,'*.jpg')))

     % Training set process
     k = dir(fullfile(Dataset,'*.jpg'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage       = imread(horzcat(Dataset,filesep,k{j}));
        imgInfo         = imfinfo(horzcat(Dataset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
            DataSet{j}   = double(imresize(tempImage,[width height])); % array of images
         else
            DataSet{j}   = double(imresize(rgb2gray(tempImage),[width height])); % array of images
         end
     end
 end
 
TestSet =  cell([], 1);
  for i=1:length(dir(fullfile(Testset,'*.jpg')))

     % Training set process
     k = dir(fullfile(Testset,'*.jpg'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage       = imread(horzcat(Testset,filesep,k{j}));
        imgInfo         = imfinfo(horzcat(Testset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
%             fprintf('j:%d is gray',j);
            TestSet{j}   = double(imresize(tempImage,[width height])); % array of images
         else
            TestSet{j}   = double(imresize(rgb2gray(tempImage),[width height])); % array of images
         end
     end
  end
fprintf('Image finished reading');
  
% Prepare class label for first run of svm
% I have arranged labels 1 & 2 as per my convenience.
% It is always better to label your images numerically
% Please note that for every image in our Dataset we need to provide one label.
% we have 30 images and we divided it into two label groups here.
train_label = zeros(size(306,1),1);
train_label(1:174,1) = 1;         % 1 = Burger % True Class 1 1-174 Burgers
train_label(175:306,1) = 2;         % 2 = Coffee % True Class 2 175-306 Coffee

% Prepare numeric matrix for svmtrain
Training_Set=zeros(length(DataSet),1);
for i=1:length(DataSet)
    Training_Set(i)=reshape(DataSet{i},1, 100*100);
end
fprintf('Training set finished');
Test_Set=zeros(length(TestSet),1);
for j=1:length(TestSet)
    Test_Set(j)=reshape(TestSet{j},1, 100*100);
end
fprintf('Test set finished');

% Perform first run of svm
% True Class 1 1-155 Burgers
% True Class 2 156-285 Coffee
test_label = zeros(size(285, 1),1);
test_label(1:155,1) = 1;
test_label(156:285,1) = 2;

model = svmtrain(train_label, Training_Set, '-t 0');
[predict_label, accuracy, dec_values] = svmpredict(test_label, Test_Set, model)
