% Guillaume's

x = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre','novembre', 'decembre'];
x1 = 1:12;

y_old = [6 7 12 8 8 2 11 5 8 7 3 6];
y_new = [10 13 20 15 15 4 16 11 14 12 5 7];

ysum = sum(y_new);

bar(x1,y_new/ysum)
hold on
title("Nombre de Guillaume nés pour chaque mois")
xlabel("Mois [janvier (1) - decembre (12)]")
ylabel("Nombre de Guillaume")
plot(xlim,[0.0833 0.0833], 'r')
hold off

