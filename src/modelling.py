import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#  Load Impact Links and Event Data
def load_event_impact_data(
        file_path,
        data_sheet="data",
        impact_sheet="impact_links"):
    """
    Load events and impact_links sheets.
    """
    print("\n========== LOADING DATA ==========")

    # Load sheets
    data_df = pd.read_excel(file_path,sheet_name=data_sheet)

    impact_df = pd.read_excel(file_path,sheet_name=impact_sheet)

    print("Data records:", len(data_df))

    print("Impact links:", len(impact_df))

    return data_df, impact_df

#  Join Events with Impact Links
def join_events_with_impacts(
        data_df,
        impact_df,
        event_id_column="record_id",
        parent_id_column="parent_id"):
    """
    Join events with impact_links using parent_id.

    Parameters
    ----------
    data_df:
        Main data sheet containing events

    impact_df:
        impact_links sheet

    event_id_column:
        Event identifier column in data sheet

    parent_id_column:
        Linking column in impact_links
    """

    print("\n========== JOIN EVENT IMPACT DATA ==========")

    # Select events only
    events = data_df[data_df["record_type"] == "impact_link"].copy()


    print("Events found:",len(events))

    # Check columns exist
    if event_id_column not in events.columns:

        raise ValueError(
            f"{event_id_column} not found in data sheet. "
            f"Available columns: {events.columns.tolist()}"
        )

    if parent_id_column not in impact_df.columns:

        raise ValueError(
            f"{parent_id_column} not found in impact_links. "
            f"Available columns: {impact_df.columns.tolist()}"
        )

    # Merge
    event_impacts = impact_df.merge(
        events,
        left_on=parent_id_column,
        right_on=event_id_column,
        how="left",
        suffixes=("_impact", "_event")
    )

    print("Joined records:",len(event_impacts))

    return event_impacts


#  Create Event Impact Summary
def create_event_effect_curve(event_date,end_date,lag_months,impact_strength):
    """
    Create time-based event impact curve.

    Parameters:
    event_date      : event start date
    end_date        : analysis end date
    lag_months      : delay before impact starts
    impact_strength : size of event impact
    """

    # Convert strings to datetime
    event_date = pd.to_datetime(event_date)

    end_date = pd.to_datetime(end_date)

    # Create monthly timeline
    dates = pd.date_range(start=event_date,end=end_date,freq="MS")

    effects = []

    for date in dates:

        # Calculate months after event
        months_after = ((date.year - event_date.year) * 12 +(date.month - event_date.month))

        # Before lag period
        if months_after < lag_months:
            effect = 0

        # Gradual impact after lag
        else:

            effect = (impact_strength*(1 -np.exp(-(months_after - lag_months) / 12)))

        effects.append(effect)

    return pd.DataFrame({"date": dates,"event_effect": effects})

#  Prepare Event Impact Links
def prepare_event_impacts(impact_df):
    """
    Prepare impact links for modeling.

    Required columns:
    parent_id
    related_indicator
    impact_direction
    impact_magnitude
    lag_months
    """

    print("\n========== PREPARING EVENT IMPACTS ==========")

    df = impact_df.copy()

    # Convert impact direction to numerical effect
    direction_map = {"positive": 1,"negative": -1,"neutral": 0}

    df["direction_score"] = (df["impact_direction"].map(direction_map).fillna(0))

    # Convert impact magnitude
    magnitude_map = {"high": 3,"medium": 2,"low": 1}

    df["magnitude_score"] = (df["impact_magnitude"].map(magnitude_map).fillna(1))

    # Calculate total impact strength
    df["impact_strength"] = (df["direction_score"]*df["magnitude_score"])

    print(df.head())

    return df


#  Create Time-Based Event Effects
def create_event_effect_curve(event_date,end_date,lag_months,impact_strength):
    """
    Represent how event impact changes over time.

    Assumption:
    - No effect during lag period
    - Gradual increase after lag
    """

    dates = pd.date_range(start=event_date,end=end_date,freq="MS")

    effects = []

    for date in dates:

        months_after = ((date.year - event_date.year) * 12 +(date.month - event_date.month))

        # Before lag period
        if months_after < lag_months:
            effect = 0

        # After lag period
        else:

            # gradual growth
            effect = (impact_strength *(1 - np.exp(-(months_after-lag_months)/12)))

        effects.append(effect)

    return pd.DataFrame({ "date": dates,"event_effect": effects})


#  Combine Multiple Event Effects
def combine_event_effects(event_curves):
    """
    Combine effects from multiple events.
    """

    print("\n========== COMBINING EVENT EFFECTS ==========")

    combined = (event_curves.groupby("date")["event_effect"].sum().reset_index())

    return combined


#  Predict Indicator Change
def predict_indicator_change(historical_df,event_effect_df,
        date_column="observation_date",
        value_column="value_numeric"):
    """
    Add event effects to historical indicator values.
    """

    print("\n========== PREDICTED INDICATOR CHANGE ==========")

    df = historical_df.copy()

    df[date_column] = pd.to_datetime(df[date_column])

    merged = df.merge(event_effect_df,left_on=date_column,
        right_on="date",how="left")

    merged["event_effect"] = (merged["event_effect"].fillna(0))

    # predicted value
    merged["predicted_value"] = (merged[value_column] + merged["event_effect"])

    plt.figure(figsize=(10,5))

    plt.plot(merged[date_column],merged[value_column],label="Observed")

    plt.plot(merged[date_column], merged["predicted_value"],label="Event Adjusted")

    plt.title("Indicator Change After Events")

    plt.xlabel("Date")

    plt.ylabel("Indicator Value")

    plt.legend()

    plt.grid(True)

    plt.show()

    return merged