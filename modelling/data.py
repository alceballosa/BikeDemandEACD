import datetime
import typing as t
import pandas as pd
import typing_extensions as te



class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def _split_by_years(X, y, years_train):
    train_indices = X["yr"].isin(years_train)
    test_indices = ~train_indices
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def get_dataset(
    reader: DatasetReader, splits: t.Iterable[SplitName], years_train: t.List
):
    df = reader()
    df = clean_dataset(df)
    (
        df,
        shifted_varnames_cnt,
        shifted_varnames_num,
        shifted_varnames_cat,
    ) = expand_dataset(df)
    feature_columns = (
        [
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ]
        + shifted_varnames_cnt
        + shifted_varnames_cat
        + shifted_varnames_num
    )
    target_column = "cnt"
    y = df[target_column]
    X = df[feature_columns]

    X_train, X_test, y_train, y_test = _split_by_years(X, y, years_train)

    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return (
        {k: split_mapping[k] for k in splits},
        shifted_varnames_cnt,
        shifted_varnames_num,
        shifted_varnames_cat,
    )


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [_add_dateindex, _drop_columns_stage_1, _fix_date_columns, _fix_other_columns]
    )
    df = cleaning_fn(df)
    return df


def _add_dateindex(df):
    """
    Generate a datetime index for the dataset, which will be used for some tasks
    further down the pipeline
    """
    df.index = pd.to_datetime(df["dteday"] + " " + df["hr"].astype("str") + ":00:00")
    df = df.reindex(pd.date_range(start=min(df.index), end=max(df.index), freq="1H"))
    return df


def _drop_columns_stage_1(df):
    """
    Drop columns not required after generating the datetime index.
    """
    cols_to_drop = ["dteday", "registered", "casual", "instant"]
    df = df.drop(cols_to_drop, axis=1)
    return df


def _fix_date_columns(df):
    """
    Fixes all columns that can be
    fixed using the datetime index alone.
    """
    df["yr"] = df.index.year
    df["mnth"] = df.index.month
    df["hr"] = df.index.hour
    df["weekday"] = df.index.weekday
    # df["instant"] = np.arange(1, len(df) + 1, 1)
    return df


def _fix_other_columns(df):
    """
    Fills all other columns using reasonably similar rows.
    """
    cols_to_fill1 = [
        "season",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
    ]
    dhour = datetime.timedelta(hours=1)
    dday = datetime.timedelta(days=1)
    null_idxs = df[df.isna().any(axis=1)].index

    for idx in null_idxs:
        if (idx - dhour).dayofweek == idx.dayofweek:
            similar_row = df.loc[idx - dhour]
        else:
            similar_row = df.loc[idx + dhour]
        if similar_row.isna().any():
            similar_row = df.loc[idx - dday]
        df.loc[idx, cols_to_fill1] = similar_row[cols_to_fill1]

    cols_to_fill2 = ["season", "holiday"]

    for idx in null_idxs:

        candidates = df[
            (~df.isna().any(axis=1))
            & (df.index.day == idx.day)
            & (df.index.month == idx.month)
            & (df.index.year == idx.year)
        ]
        similar_row = candidates.iloc[0, :]
        df.loc[idx, cols_to_fill2] = similar_row[cols_to_fill2]

    return df


def expand_dataset(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, t.List, t.List, t.List]:

    (
        df,
        shifted_varnames_cnt,
        shifted_varnames_num,
        shifted_varnames_cat,
    ) = _get_shifted_timeseries(df)
    df, shifted_varnames_cnt = _get_diffd_timeseries(df, shifted_varnames_cnt)
    df = _fix_ts_nulls(df)
    return df, shifted_varnames_cnt, shifted_varnames_num, shifted_varnames_cat


def _get_shifted_timeseries(df):
    def get_rolled_variables(df, shifts, variables_to_shift):
        shifted_varnames = []
        df_rolld = df.copy()
        for var in variables_to_shift:
            for shift in shifts:
                col_name = (
                    var + "_" + str(shift) + "_hours"
                    if shift < 24
                    else var + "_" + str(shift // 24) + "_days"
                )
                shifted_varnames.append(col_name)
                df_rolld[col_name] = df_rolld[var].shift(shift)
        return df_rolld, shifted_varnames

    df_rolld, shifted_varnames_cnt = get_rolled_variables(
        df, shifts=[1, 2, 3, 24 * 7], variables_to_shift=["cnt"]
    )

    df_rolld, shifted_varnames_num = get_rolled_variables(
        df_rolld,
        shifts=[1, 2, 3],
        variables_to_shift=["temp", "atemp", "hum", "windspeed"],
    )

    df_rolld, shifted_varnames_cat = get_rolled_variables(
        df_rolld, shifts=[24 * 7], variables_to_shift=["holiday", "workingday"]
    )

    return df_rolld, shifted_varnames_cnt, shifted_varnames_num, shifted_varnames_cat


def _get_diffd_timeseries(df, shifted_varnames_cnt):
    if "cnt_1_hours" in df.columns:
        df["cnt_last_hour_diff"] = df["cnt_1_hours"].diff()
        df[["cnt_1_hours", "cnt_2_hours", "cnt", "cnt_last_hour_diff"]].head()
        shifted_varnames_cnt = shifted_varnames_cnt + ["cnt_last_hour_diff"]

    return df, shifted_varnames_cnt


def _fix_ts_nulls(df):
    df = df.dropna(axis=0)
    return df


def aggregate_features(
    shifted_varnames_cnt, shifted_varnames_num, shifted_varnames_cat
):
    model_features = (
        [
            "season",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ]
        + shifted_varnames_cnt
        + shifted_varnames_cat
        + shifted_varnames_num
    )

    categorical_features = [
        "season",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ] + shifted_varnames_cat

    numerical_features = (
        ["temp", "atemp", "hum", "windspeed"]
        + shifted_varnames_num
        + shifted_varnames_cnt
    )

    return model_features, categorical_features, numerical_features