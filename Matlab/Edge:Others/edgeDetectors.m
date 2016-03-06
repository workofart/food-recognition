% % % % % % % % % % % % % Comparison of different edge detectors % % % % % % % % 

% Read a sample class of images
% Resize the image to 500x500 for faster processing
img = imread('../FoodData/Data/Train/1.jpg');
imgresized = imresize(rgb2gray(img),[500 500]);

% Corner Detection
corners = detectFASTFeatures(imgresized,'MinContrast',0.1);
J = insertMarker(imgresized,corners,'circle');
imshow(J);

% BRISK Points
points = detectBRISKFeatures(imgresized);
imshow(imgresized); hold on;
plot(points.selectStrongest(20));

% MSER Regions
regions = detectMSERFeatures(imgresized);
figure; imshow(imgresized); hold on;
plot(regions, 'showPixelList', true, 'showEllipses', false);

% Display Ellipses
figure; imshow(imgresized); hold on;
plot(regions); % by default, plot displays ellipses and centroids

% Interest Point Descriptors
corners = detectHarrisFeatures(imgresized);
[features, valid_corners] = extractFeatures(imgresized, corners);
figure; imshow(imgresized); hold on
plot(valid_corners);

% Histogram of Oriented Gradients (HOG) Features
[featureVec, hogVis] = extractHOGFeatures(imgresized);
figure;
imshow(imgresized); hold on;
plot(hogVis);

s = regionprops(imgresized, 'centroid');
centroids = cat(1, s.Centroid);
imshow(imgresized);
hold on;
plot(centroids(:,1), centroids(:, 2), 'b*');
hold off;

% Comparison among 'Canny' 'Prewitt' 'Sobel'
BW1 = edge(imgresized, 'Canny');
BW2 = edge(imgresized, 'Prewitt');
BW3 = edge(imgresized, 'Sobel');
figure;
imshowpair(BW2, BW3, 'montage');
figure;
imshow(BW3);
data = double(BW2);