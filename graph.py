import matplotlib.pyplot as plt

legend = ["16","32","64","128","256"]
x = [(i + 1) for i in range(16)]
y_enc = [
    [214,343,647,646,566,892,1482,2690,3509,118,117,113,112,114,115,121],
    [282,417,646,780,555,659,1301,2785,3843,142,138,136,136,135,138,144],
    [356,495,663,506,515,585,1155,2730,4059,164,161,160,159,159,161,166],
    [506,633,508,350,405,555,992,2568,3920,210,206,204,202,200,201,206],
    [762,732,439,293,444,596,932,1963,3614,289,287,280,277,272,272,277]
]

y_dec = [
    [195,114,80,65,62,53,46,63,90,89,88,86,86,87,89,94],
    [210,121,84,68,66,56,48,52,94,93,92,91,91,91,94,99],
    [223,127,88,70,70,61,54,53,98,98,96,95,95,96,98,104],
    [257,142,98,77,83,66,57,64,106,104,101,100,100,101,104,109],
    [306,166,113,88,92,76,67,75,117,114,109,107,106,107,112,118]
]

def draw(y, type, type_name):
    plt.title("Время "+type+" от кол-ва потоков для различных размеров блока")
    plt.xlabel("Потоки, шт")
    plt.ylabel("Время, мкс")
    for i in range(len(legend)):
        plt.plot(x, y[i])
        plt.legend(legend)

    plt.savefig(type_name+"_time.png")
    plt.clf()

    plt.title("Ускорение "+type+" от кол-ва потоков для различных размеров блока")
    plt.xlabel("Потоки, шт")
    plt.ylabel("Ускорение")
    for i in range(len(legend)):
        accel = [y[i][0] / y[i][j] for j in range(16)]
        plt.plot(x, accel)
        plt.legend(legend)

    plt.savefig(type_name+"_accel.png")
    plt.clf()

    plt.title("Эффективность "+type+" от кол-ва потоков для различных размеров блока")
    plt.xlabel("Потоки, шт")
    plt.ylabel("Эффективность")
    for i in range(len(legend)):
        eff = [y[i][0] / y[i][j] / (j + 1) for j in range(16)]
        plt.plot(x, eff)
        plt.legend(legend)
        
    plt.savefig(type_name+"_eff.png")
    plt.clf()

draw(y_enc, "шифрования", "enc")
draw(y_dec, "дешифрования", "dec")