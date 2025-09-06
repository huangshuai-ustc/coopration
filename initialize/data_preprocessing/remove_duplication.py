import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_dedup = df.drop_duplicates()
    return df_dedup


def remove_duplicates_by_doi(df: pd.DataFrame) -> pd.DataFrame:
    dedup_columns = ["DOI"]

    # 去掉这些列中任意一个为空的行
    df_filtered = df.dropna(subset=dedup_columns)

    # 去重
    df_dedup = df_filtered.drop_duplicates(subset=dedup_columns)

    # 如果你还想保留原始的空值行，可以加回来（可选）
    df_final = pd.concat([df_dedup, df[df[dedup_columns].isnull().any(axis=1)]])
    return df_final


if __name__ == '__main__':
    df = pd.read_csv("My EndNote Library.csv")
    print(df.shape)
    df = remove_duplicates(df)
    df = df[df['Abstract'].notna()]
    print(df.shape)
    df = remove_duplicates_by_doi(df)
    print(df.shape)
    df.to_csv("no_duplication.csv", index=False)
    # df = pd.read_excel('ok_data_enhancement.xlsx')
    # print(df.shape)
    # df = remove_duplicates(df)
    # df = df[df['Abstract'].notna()]
    # print(df.shape)
    # df = remove_duplicates_by_doi(df)
    # print(df.shape)
    # df.to_csv("ok_data_enhancement.csv", index=False)
