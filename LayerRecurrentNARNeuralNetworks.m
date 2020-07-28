% Layer-Recurrent NAR Neural Networks (layrecnet)
% Нейросеть обучается именно на том типе данных, который был на неё подан!
% Если был подан тип cell, то и выдавать тот же результат НС будет на cell
% load phoneme
% X = con2seq(y);
% T = con2seq(t);
%[X,T] = simpleseries_dataset;
name = 'LRNAR_';
if ~iscell(X) % Вход
    IN = con2seq(X);
else
    IN = X;
end
if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
%IN = con2seq(X); OUT = con2seq(T); % IN - вход, OUT - выход
%IN = X; OUT = T; % IN - вход, OUT - выход
minHN = 10; % Минимальное количесвто скрытых нейронов
maxHN = 10; % Максимальное количесвто скрытых нейронов
trainDelayArr = [5 10]; % Массив возможных задержек
maxEpochs = 1000; 
maxLoop = 10; % Максимальное количество циклов поиска лучшей нейросети
trainParamStr = ["trainlm" "trainbr" "trainbfg" "trainrp" "trainscg" "traincgb" "traincgf" "traincgp" "trainoss" "traingdx" "traingdm" "traingd"];
%trainParamStr = ["trainlm"];
trainParamIW = ones(1, size(trainParamStr, 2)); % Веса алгоритмов тренировки
trainParamIWadd = 0.01; % значение для управления весами алгоритмов
Result = []; % Разные параметры результатов
ResultNet = {}; % Массив нейросетей
bestIdx = 0;
if maxLoop >= 100
    autoSaveEvery = round(maxLoop*0.25/(maxLoop/100)); 
else
    autoSaveEvery = round(maxLoop*0.25); 
end
for i = 1:maxLoop
    tic; % Начало таймера
    HN = round((maxHN-minHN).*rand(1) + minHN); % Hidden neurons, количество скрытых нейронов
    trainRand = round((size(trainParamStr,2)-1).*rand(1) + 1); % Случайный алгоритм тренировки
    if rand(1) >= 0.5 && bestIdx ~= 0
        [bestTrainVal, bestTrainIdx] = max(trainParamIW(1,:)); % Выбираем лучший алгоритм (с наибольшим весом)
        trainRand = bestTrainIdx; % Выбираем лучший алгоритм по его индексу
    end
    trainDelayRand = round((size(trainDelayArr,2)-1).*rand(1) + 1); % Случайная задержка
    nnet = layrecnet(1:trainDelayArr(trainDelayRand), HN); % Создание нейросети RNN
    nnet.trainFcn = char(trainParamStr(trainRand)); % Тренировочный алгоритм
    %nnet.trainParam.show = 5;
    nnet.trainParam.epochs = maxEpochs;
    nnet.trainParam.showWindow = false; % Не показывать окно тренировки
    nnet.divideParam.trainRatio = 70/100; % Set up Division of Data for Training, Validation, Testing
    nnet.divideParam.valRatio = 15/100;
    nnet.divideParam.testRatio = 15/100;
    [Xs,Xi,Ai,Ts] = preparets(nnet,IN, OUT); % Подготовка данных
    nnet = train(nnet,Xs,Ts,Xi,Ai); %  Тренировка нейросети
    %IN = Xs; OUT = Ts; % Мистичный новый вход и выход
    y = nnet(Xs,Xi,Ai); % Выходной ответ нейросети
    %plot(cell2mat(y))
    perf = perform(nnet,y,Ts); % Ошибка на выходе нейросети и данных
    r = mean(regression(Ts, y)); % Значение регрессии нейросети и данных
    Result = [Result; HN perf r trainRand trainDelayArr(trainDelayRand) toc];
    ResultNet{end+1} = nnet;
    if mod(i, autoSaveEvery) == 0
        [bestVal, bestIdx] = max(Result(:,3)); % Выбираем лучшую регрессию
        bestNet = ResultNet{bestIdx}; % Выбираем нейросеть с лучшей регрессией
        for j = 1:size(trainParamIW, 2) 
            if j ~= Result(bestIdx, 4) % Увеличиваем веса у хороших алгоритмов, уменьшаем у плохих
                trainParamIW(1, j) = trainParamIW(1, j) - trainParamIWadd;
            else
                trainParamIW(1, j) = trainParamIW(1, j) + trainParamIWadd;
            end
        end
        fprintf('iter = %i; ', i);
        ResultStr = sprintf('№ %i; HN = %i; Delay = %i; trainParam = %s; R = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, Result(bestIdx, 1), Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), bestVal, Result(bestIdx, 2), Result(bestIdx, 6));
        fprintf(ResultStr); fprintf('\n');
        dateTime = char(replace(string(datetime('now')),":","-"));
        fileName = ['bestNet_' name sprintf('HN=%i R=%0.3f', Result(bestIdx, 1), bestVal) '_' dateTime '.mat'];
        save(fileName, 'bestNet', 'ResultStr');  % Сохраняем лучшую нейросеть
    end
end
[bestVal, bestIdx] = max(Result(:,3)); % Выбираем лучшую регрессию
bestNet = ResultNet{bestIdx}; % Выбираем нейросеть с лучшей регрессией
ResultStr = sprintf('№ %i; HN = %i; Delay = %i; trainParam = %s; R = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, Result(bestIdx, 1), Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), bestVal, Result(bestIdx, 2), Result(bestIdx, 6));
dateTime = char(replace(string(datetime('now')),":","-"));
fileName = ['bestNet_' name sprintf('HN=%i R=%0.3f', Result(bestIdx, 1), bestVal) '_' dateTime '.mat'];
save(fileName, 'bestNet', 'ResultStr');  % Сохраняем лучшую нейросеть
[Xs,Xi,Ai,Ts] = preparets(bestNet,IN, OUT); % Подготовка данных
y = bestNet(Xs,Xi,Ai); % Выходной ответ нейросети
ResultStr = sprintf('№ %i; HN = %i; Delay = %i; trainParam = %s; \nR = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, Result(bestIdx, 1), Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), bestVal, Result(bestIdx, 2), Result(bestIdx, 6));
if size(Ts, 1) == 1 % Если выход 1, то выводим график, иначе будет белеберда
    figOutput = figure('Name','Target and NAR output');
    figure(figOutput); % figOutput figRegress
    if iscell(Ts) % Выход
        plot(cell2mat(Ts), 'LineWidth',2); hold on; 
        plot(cell2mat(y))
    else
        plot(Ts, 'LineWidth',2); hold on; 
        plot(y)
    end
    legend('Terget', sprintf('HN=%i R=%0.3f', Result(bestIdx, 1), bestVal))
    title(ResultStr)
end
figRegress = figure('Name','Target and NAR regression');
figure(figRegress); % figOutput figRegress
plotregression(Ts, y, ResultStr)
view(bestNet)
% net = closeloop(net); net = openloop(net);

