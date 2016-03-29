% % % % % % % % % % % % % Comparison of different edge detectors % % % % % % % % 
clear;
% Read a sample class of images
% Resize the image to 500x500 for faster processing
img = imread('../../FoodData/EdgeDetection/apple.jpg');
imgresized = imresize(rgb2gray(img),[500 500]);

% % Corner Detection
% corners = detectFASTFeatures(imgresized,'MinContrast',0.1);
% J = insertMarker(imgresized,corners,'circle');
% imshow(J);
% 
% % BRISK Points
% points = detectBRISKFeatures(imgresized);
% imshow(imgresized); hold on;
% plot(points.selectStrongest(20));
% 
% % MSER Regions
% regions = detectMSERFeatures(imgresized);
% figure; imshow(imgresized); hold on;
% plot(regions, 'showPixelList', true, 'showEllipses', false);
% 
% % Display Ellipses
% figure; imshow(imgresized); hold on;
% plot(regions); % by default, plot displays ellipses and centroids
% 
% % Interest Point Descriptors
% corners = detectHarrisFeatures(imgresized);
% [features, valid_corners] = extractFeatures(imgresized, corners);
% figure; imshow(imgresized); hold on
% plot(valid_corners);
% 
% % Histogram of Oriented Gradients (HOG) Features
% [featureVec, hogVis] = extractHOGFeatures(imgresized);
% figure;
% imshow(imgresized); hold on;
% plot(hogVis);
% 
% s = regionprops(imgresized, 'centroid');
% centroids = cat(1, s.Centroid);
% imshow(imgresized);
% hold on;
% plot(centroids(:,1), centroids(:, 2), 'b*');
% hold off;



% SURF Features 
img_db = double(imgresized);
points = detectSURFFeatures(imgresized);
[features, valid_points] = extractFeatures(img_db, points);

% Visualize 10 strongest SURF features, including their scales and orientation which were determined during the descriptor extraction process.
imshow(imgresized); hold on;
strongestPoints = valid_points.selectStrongest(10);
strongestPoints.plot('showOrientation',true);

% Comparison among 'Canny' 'Prewitt' 'Sobel'
BW1 = edge(imgresized, 'Canny');
BW2 = edge(imgresized, 'Prewitt');
BW3 = edge(imgresized, 'Sobel');

% set(hFig, 'Position', [10 10 600 800]);

h(1) = subplot(1,4,1);
% set(h(1),'position',[10 10 200 300]);
imshow(imgresized); title('Original');
h(2) = subplot(1,4,2);
% set(h(1),'position',[10 10 200 300]);
imshow(BW1); title('Canny');
h(3) = subplot(1,4,3);
% set(h(2),'position',[10 10 200 300]);
imshow(BW2); title('Prewitt');
h(4) = subplot(1,4,4);
% set(h(3),'position',[10 10 200 300]);
imshow(BW3); title('Sobel');





% set(title,'Position',[150 300],'VerticalAlignment','top','Color',[1 0 0]);

% descr = {'Prewitt(left)'; 'Sobel(right)'};
% imshowpair(BW2, BW3, 'montage');
% imshowpair(BW2,BW3,'diff');
% imshowpair(BW2, BW3, 'blend','Scaling','joint');
% figure;
% descr = {'Canny'};
% imshow(BW3);
% data = double(BW2);