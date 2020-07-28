clc; close all;
%img = imread('filename.png'); % Read image
I = im2double(pic3);
R = I(:,:,1); % Red channel
G = I(:,:,2); % Green channel
B = I(:,:,3); % Blue channel

subplot(2, 4, 1)
imshow(I), title('Original image') % 'figure, imshow(I), title('Original image')
subplot(2, 4, 2)
imshow([R; G; B]), title('Original RGB image')

% Нормализация от 0 до 1. Но nnstart делает это сам
S = R+G+B;
Rn = R./S;
Gn = G./S;
Bn = B./S;
subplot(2, 4, 3)
imshow([Rn; Gn; Bn]), title('Normalize RGB image')
subplot(2, 4, 4)
imshow(cat(3, Rn, Gn, Bn)), title('Normalize image')
Input = [Rn; Gn; Bn]; %Вход для нейросети. Каналы R G B. 
OutTest = cat(3, Rn, Gn, Bn);
output = net(Input);% myNeuralNetworkFunction_MakeMouse(Input);
%net(Input);% 
pic_h = size(I, 1); %высота картинки
NNR = output(1:pic_h, :);
NNG = output(pic_h+1:2*pic_h, :);
NNB = output(2*pic_h+1:3*pic_h, :);
NNImage = cat(3, NNR.*S, NNG.*S, NNB.*S);
subplot(2, 4, 5)
imshow([NNR; NNG; NNB]), title('Normalize RGB output image')
subplot(2, 4, 6)
imshow([NNR.*S; NNG.*S; NNB.*S]), title('RGB output image')
subplot(2, 4, 7)
imshow(cat(3, NNR.*1, NNG.*1, NNB.*1)), title('Normalize output image')
subplot(2, 4, 8)
imshow(cat(3, NNR.*S, NNG.*S, NNB.*S)), title('NN output image')

a = zeros(size(I, 1), size(I, 2));
just_red = cat(3, red, a, a);
just_green = cat(3, a, green, a);
just_blue = cat(3, a, a, blue);
back_to_original_img = cat(3, red, green, blue);

% figure, imshow(img), title('Original image')
% figure, imshow(just_red), title('Red channel')
% figure, imshow(just_green), title('Green channel')
% figure, imshow(just_blue), title('Blue channel')
% figure, imshow(back_to_original_img), title('Back to original image')

% layers = [imageInputLayer([134 200 1]);
% convolution2dLayer(5,20);
% reluLayer();
% maxPooling2dLayer(2,'Stride',2);
% fullyConnectedLayer(10);
% softmaxLayer();
% classificationLayer()];
% options = trainingOptions('sgdm','MaxEpochs',20,...
% 'InitialLearnRate',0.0001);
% convnet = trainNetwork(pic,pic, layers,options);