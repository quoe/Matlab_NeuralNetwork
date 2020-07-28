delay = 10;
for i = 250:1:257
    splitStart = 1000+i; pt = cell2mat([OUT(splitStart-1000:splitStart)]);
    py = cell2mat(bestNet(OUT, OUT(splitStart-delay:splitStart-1)));
    plot([pt py]); hold on;
end
