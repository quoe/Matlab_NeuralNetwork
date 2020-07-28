function [ outputMatrix ] = Input2Categorial( InputColumn )
%Input2Categorial Создание матрицы на основе колонны с данными
%   Создание матрицы для категориальной нейросети на основе
%колонны с данными
    if size(InputColumn, 1) == 1
        InputSize = size(InputColumn', 1); % Row count
    else
        InputSize = size(InputColumn, 1); % Row count
    end
    output = zeros(InputSize);
    CatMap = containers.Map('KeyType','char','ValueType','int32'); % Dictionary
    itemsCount = 1;
    for i = 1:InputSize % Заполнение словаря
        key = char(InputColumn(i));
        if isKey(CatMap, key) == 0
            CatMap(key) = itemsCount;
            itemsCount = itemsCount + 1;
        end
    end

    for i = 1:InputSize % Заполнение матрицы
        key = char(InputColumn(i));
        value = CatMap(key);
        output(i, value) = 1;
    end
    outputMatrix = output;
end

