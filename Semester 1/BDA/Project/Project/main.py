from ucimlrepo import fetch_ucirepo

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X = iris.data.features
    y = iris.data.targets

    # metadata
    print(iris.metadata)

    # variable information
    print(iris.variables)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
