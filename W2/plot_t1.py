import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt


def create_table(input_path):
    df = pd.read_csv(input_path, delimiter=",",names=["rho","AP"], skiprows=[0])
    print(df.head())
    print(df['rho'])
    with sns.axes_style("darkgrid"):
        sns.lineplot(data=df, x="rho", y="AP")
    plt.title('Rho value vs Average Precision with alpha=3 (ZOOM)')
    plt.show()

if __name__== "__main__" :
    create_table(input_path=sys.argv[1])