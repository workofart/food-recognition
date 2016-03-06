% read image files
clear
% img = imread('../FoodData/apple/0001.jpg');
% % imgRGB = convertRGB(img);
% 
% 
% inprof = iccread('USSheetfedCoated.icc');
% outprof = iccread('sRGB.icm');
% C = makecform('icc',inprof,outprof);
% I_rgb = applycform(img,C);
% I_gray = rgb2gray(I_rgb);
% grayimg= double(rgb2gray(imgRGB));
% img_resized = imresize(grayimg, [128 128]);
% 
% 
% img2 = imread('../FoodData/burger/0002.jpg');
% img2_resized = imresize(img2, [128 128]);
% img2_resized = double(img2_resized);

% # image size
sz = [200,200];

%# training images
% True Class 1 1-196 Burgers
T_Train = ones(196, 1);
% True Class 2 197-350 Coffee
T_Train = [T_Train; ones(154, 1)*-1];

numTrain = 350;
F_Train = zeros(numTrain,prod(sz)); % flatten the training data
for i=1:numTrain
    fprintf('\ntrain: %d', i);
    img = imresize(double(rgb2gray(imread( sprintf('../FoodData/Data/Train/%d.jpg',i) ))), sz); % 3 digit-numbers, leading space replaced by zeros
    img = reshape(double(edge(img, 'Canny')),[40000,1]);
    F_Train(i,:) = img(:);
end

%# testing images
% True Class 1 1-154 Burgers
T_Test = ones(154, 1);
% True Class 2 155-262 Coffee
T_Test = [T_Test; ones(108, 1)*-1];

numTest = 262; % first 88 images are RGB
F_Test = zeros(numTest,prod(sz));
for i=1:numTest
    fprintf('\ntest: %d', i);
    img = imresize(double(rgb2gray(imread( sprintf('../FoodData/Data/Test/%d.jpg',i) ))), sz); % 3 digit-numbers, leading space replaced by zeros
    img = reshape(double(edge(img, 'Canny')),[40000,1]);
    F_Test(i,:) = img(:);
end

k =3;
pred = classifyKnn(F_Train, T_Train, F_Test,k);

[CONF,err] = confusionMatrix(pred,T_Test)

pred = classifyKnnNormalized(F_Train, T_Train, F_Test, k);

[CONF,err] = confusionMatrix(pred,T_Test)

iter = 10000;
pred = PerceptronBatchNormalized(F_Train,T_Train,F_Train, iter);

[CONF,err] = confusionMatrix(pred,T_Train)
% k = 3, 128x128 22.86%
% k = 3, 256x256 23.33%
% k = 3, 64x64 24.76%
