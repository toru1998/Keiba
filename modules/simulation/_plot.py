import matplotlib.pyplot as plt

def plot_single_threshold(df, N_SAMPLES, label=' '):
    plt.figure(dpi=100)
    #標準偏差で幅をつけて薄くプロット
    plt.fill_between(
        df.index,
        y1=df['return_rate']-df['std'],
        y2=df['return_rate']+df['std'],
        alpha=0.3
        )
    #回収率を実線でプロット
    plt.plot(df.index, df['return_rate'], label=label)
    for i in range(10):
        idx = i * N_SAMPLES // 10
        plt.text(df.iloc[idx].name, df.iloc[idx]['return_rate'], str(int(df.iloc[idx]['n_bets'])))
    plt.legend() #labelで設定した凡例を表示させる
    plt.grid(True) #グリッドをつける
    plt.xlabel('threshold')
    plt.ylabel('return_rate')
    plt.show()