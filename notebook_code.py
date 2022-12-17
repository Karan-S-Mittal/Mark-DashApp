import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from plotnine import *
from plotnine import ggplot, geom_map, aes, scale_fill_cmap, theme, labs
from plotnine.data import mpg
import plotnine as p9
import squarify

# from pandas_profiling import ProfileReport

import plotly.express as px
import plotly.io as pio
import geopandas as gpd
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

# Importing Dataframe
# ------------------------------------------------------------------------------------
df = pd.read_csv("madrid_transactions.csv", index_col=0)
countries = pd.read_csv("country-and-continent-codes-list.csv")
df = df.merge(countries, left_on="customer_country", right_on="Two_Letter_Country_Code")

df.tx_date_proc = df.tx_date_proc.apply(pd.to_datetime)
df["Day"] = [d.date() for d in df["tx_date_proc"]]
df["Time"] = [d.time() for d in df["tx_date_proc"]]

country_code = pd.read_csv("all.csv")

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Data Cleaning
# ------------------------------------------------------------------------------------
# Reducing Country name to only first word
df["Country_Name"] = df["Country_Name"].apply(lambda x: str(x).split(",")[0])
# Reducing United Kingdom of Great Britain & Northern Ireland to United Kingdom
df["Country_Name"] = df["Country_Name"].apply(
    lambda x: "United Kingdom"
    if x == "United Kingdom of Great Britain & Northern Ireland"
    else x
)

# Traslating to English all purchase categories
df["category"] = df["category"].apply(
    lambda x: "Travel Agency" if x == "Agencias de viajes" else x
)
df["category"] = df["category"].apply(
    lambda x: "Home and reforms" if x == "Hogar y reformas" else x
)
df["category"] = df["category"].apply(
    lambda x: "Automotive" if x == "Automoción" else x
)

# Generating Aggregated Measures
# ------------------------------------------------------------------------------------
df3 = (
    df.merge(country_code, left_on="customer_country", right_on="alpha-2")
    .groupby(["customer_country", "alpha-3", "Country_Name"])["amount"]
    .sum()
    .reset_index(name="Total_Expenditure")
)
df4 = (
    df.merge(country_code, left_on="customer_country", right_on="alpha-2")
    .groupby(["customer_country", "alpha-3", "Country_Name"])["amount"]
    .count()
    .reset_index(name="Total_Transactions")
)
df5 = (
    df.merge(country_code, left_on="customer_country", right_on="alpha-2")
    .groupby(["customer_country", "alpha-3", "Country_Name"])["amount"]
    .mean()
    .reset_index(name="Avg_Ticket")
)
df_merged = df3.merge(df4, on=["customer_country", "alpha-3", "Country_Name"]).merge(
    df5, on=["customer_country", "alpha-3", "Country_Name"]
)
df_merged.sort_values(by="Total_Expenditure", ascending=False, inplace=True)


# Graphs

# ### Pareto Analysis
# --------------------
def get_pareto_chart():
    # Preparing df for pareto chart
    df_merged["cumulative_sum"] = df_merged.Total_Expenditure.cumsum()
    df_merged["cumulative_perc"] = (
        100 * df_merged.cumulative_sum / df_merged.Total_Expenditure.sum()
    )
    df_merged.sort_values(by="Total_Expenditure", ascending=False, inplace=True)

    # Pareto Chart Total Expenditure
    trace_0 = go.Bar(
        x=df_merged["Country_Name"],
        y=df_merged["Total_Expenditure"],
        marker=dict(color=df_merged["Total_Expenditure"], coloraxis="coloraxis"),
        text=df_merged["Total_Expenditure"],
        textposition="outside",
        textfont=dict(color="black"),
        texttemplate="%{text:.3s}",
    )

    trace_1 = go.Scatter(
        x=df_merged["Country_Name"],
        y=df_merged["cumulative_perc"],
        mode="markers+lines",
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(trace_0)

    fig.add_trace(trace_1, secondary_y=True)

    fig.update_layout(
        title="Pareto Analysis: Total Expenditure by Country of Origin",
        showlegend=False,
        coloraxis_showscale=False,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Total Expenditure", secondary_y=False),
    fig.update_yaxes(title_text="Cummulativee % Total Expenditure", secondary_y=True)

    return fig


df6 = df_merged.merge(world, left_on="alpha-3", right_on="iso_a3")

# geographical plot
def get_geographical_plot():

    chart = ggplot(df6, mapping=aes(fill="Total_Expenditure"))
    map_proj = geom_map(alpha=0.7)
    labels = labs(title="Total Credit Card Expenditure by Country of Origin")
    theme_details = theme(figure_size=(12, 6))
    fill_colormap = scale_fill_cmap(cmap_name="plasma")
    color_colormap = scale_color_cmap(cmap_name="plasma")
    world_map_card = (
        chart
        + map_proj
        + labels
        + theme_details
        + fill_colormap
        + color_colormap
        + theme(panel_background=element_blank())
    )
    return world_map_card


# group by & sort descending
df_sorted = (
    df6.groupby("Country_Name")
    .sum()
    .sort_values("Total_Expenditure", ascending=False)
    .reset_index()
)

# Selecting Top 25 nationalities
def get_top_countries_based_total_expenses():
    # Selecting Top XX nationalities
    # number of top-n you want
    n = 25

    # group by & sort descending
    df_sorted = (
        df6.groupby("Country_Name")
        .sum()
        .sort_values("Total_Expenditure", ascending=False)
        .reset_index()
    )

    # rename rows other than top-n to 'Others'
    df_sorted.loc[df_sorted.index >= n, "Country_Name"] = "Others"

    df_sorted = df_sorted.loc[df_sorted["Country_Name"] != "Others"]
    # re-group by again
    # df_sorted.groupby('customer_country').mean()
    # Bar Chart for top 25 countries ordered by total expenses
    return (
        ggplot(
            df_sorted,
            aes(
                x="reorder(Country_Name, Total_Expenditure, fun=sum)",
                y="Total_Expenditure",
                fill="Total_Expenditure",
            ),
        )
        + geom_bar(stat="identity", alpha=0.7)
        + scale_x_discrete()
        + coord_flip()
        + scale_fill_cmap(cmap_name="plasma")
        + ggtitle("Total Expenditure Top 25 Countries")
        + labs(y="Total Expenditure ", x="Top 25 Countries")
    )


df7 = pd.merge(
    left=df,
    right=df_sorted[["Country_Name", "Total_Expenditure"]],
    on="Country_Name",
    how="left",
)
df7["Total_Expenditure"].fillna("Other", inplace=True)
df7["Total_Expenditure"] = np.where(
    df7["Total_Expenditure"] != "Other", df7["Country_Name"], "Other"
)
df7.rename(columns={"Total_Expenditure": "Top_Expenditure"}, inplace=True)
df7 = df7.loc[df7["Top_Expenditure"] != "Other"]


def get_top_nations_based_total_expenses_vs_product_category():

    df7_pivot = pd.pivot_table(
        df7, index=["customer_country", "category"], values="amount", aggfunc=["sum"]
    ).reset_index()
    df7_pivot1 = df7_pivot.reset_index()
    df7_pivot1.columns = ["id", "customer_country", "category", "amount"]

    return (
        ggplot(df7_pivot1, aes(x="reorder(customer_country, amount)", y="category"))
        + geom_point(aes(size="amount", color="amount"), alpha=0.7)
        + theme(legend_title=element_blank())
        + scale_color_cmap(cmap_name="plasma")
        + scale_y_discrete(limits=reversed)
        + scale_size(range=(0, 12))
        + ggtitle("Total Expenditure per Category and Top 25 Countries")
        + labs(y="Category", x="Country")
        + theme(figure_size=(12, 5))
    )


def get_expenditure_per_hour_25_countries():
    df7_pivot2 = pd.pivot_table(
        df7, index=["customer_country", "hour"], values="amount", aggfunc=["sum"]
    ).reset_index()
    df7_pivot2 = df7_pivot2.reset_index()
    df7_pivot2.columns = ["id", "customer_country", "hour", "amount"]

    return (
        ggplot(df7_pivot2)
        + geom_tile(
            aes(x="hour", y="reorder(customer_country,amount)", fill="amount"),
            alpha=0.7,
        )
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Total Expenditure per Hour by Top Countries")
        + labs(y="Top 25 Country", x="Hour")
        + theme(figure_size=(12, 4))
    )


def get_expenditure_per_hour_per_category():
    df7_pivot3 = pd.pivot_table(
        df7, index=["category", "hour"], values="amount", aggfunc=["sum"]
    ).reset_index()
    df7_pivot3 = df7_pivot3.reset_index()
    df7_pivot3.columns = ["id", "category", "hour", "amount"]

    return (
        ggplot(df7_pivot3)
        + geom_tile(
            aes(x="hour", y="reorder(category,amount)", fill="amount"), alpha=0.7
        )
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Total Expenditure per Hour by Category")
        + labs(y="Top 25 Country", x="Hour")
        + theme(figure_size=(12, 4))
    )


# Credit Card Transactions Volume

# Preparing df for pareto chart
df_merged["cumulative_sum_tran"] = df_merged.Total_Transactions.cumsum()
df_merged["cumulative_perc_tran"] = (
    100 * df_merged.cumulative_sum_tran / df_merged.Total_Transactions.sum()
)
df_merged.sort_values(by="Total_Transactions", ascending=False, inplace=True)


def get_credit_pareto_analysis_chart():
    # Pareto Chart Total Expenditure
    trace_2 = go.Bar(
        x=df_merged["Country_Name"],
        y=df_merged["Total_Transactions"],
        marker=dict(color=df_merged["Total_Transactions"], coloraxis="coloraxis"),
        text=df_merged["Total_Transactions"],
        textposition="outside",
        textfont=dict(color="black"),
        texttemplate="%{text:.3s}",
    )

    trace_3 = go.Scatter(
        x=df_merged["Country_Name"],
        y=df_merged["cumulative_perc_tran"],
        mode="markers+lines",
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(trace_2)

    fig.add_trace(trace_3, secondary_y=True)

    fig.update_layout(
        title="Pareto Analysis: Total Transactions by Country of Origin",
        showlegend=False,
        coloraxis_showscale=False,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Total Transactions", secondary_y=False),
    fig.update_yaxes(title_text="Cummulativee % Total Transactions", secondary_y=True)

    return fig


def get_credit_georgraphical_chart():
    chart = ggplot(df6, mapping=aes(fill="Total_Transactions"))
    map_proj = geom_map(alpha=0.7)
    labels = labs(title="Credit Card Transactions by Country of Origin")
    theme_details = theme(figure_size=(12, 6))
    fill_colormap = scale_fill_cmap(cmap_name="plasma")
    color_colormap = scale_color_cmap(cmap_name="plasma")
    world_map_card = (
        chart
        + map_proj
        + labels
        + theme_details
        + fill_colormap
        + color_colormap
        + theme(panel_background=element_blank())
    )
    return world_map_card


def get_credit_top_countries():
    # Selecting Top XX nationalities
    # number of top-n you want
    n = 25

    # group by & sort descending
    df_sorted_tran = (
        df6.groupby("Country_Name")
        .sum()
        .sort_values("Total_Transactions", ascending=False)
        .reset_index()
    )
    # rename rows other than top-n to 'Others'
    df_sorted_tran.loc[df_sorted_tran.index >= n, "Country_Name"] = "Others"

    df_sorted_tran = df_sorted_tran.loc[df_sorted_tran["Country_Name"] != "Others"]
    # re-group by again
    # df_sorted.groupby('customer_country').mean()
    return (
        # Bar Chart for top 25 countries ordered by volume of transactions
        ggplot(
            df_sorted_tran,
            aes(
                x="reorder(Country_Name, Total_Transactions, fun=sum)",
                y="Total_Transactions",
                fill="Total_Transactions",
            ),
        )
        + geom_bar(stat="identity", alpha=0.7)
        + scale_x_discrete()
        + coord_flip()
        + scale_fill_cmap(cmap_name="plasma")
        + ggtitle("Credit Card Transactions Top 25 Countries")
        + labs(y="Volumne of Transactions ", x="Top 25 Countries")
        + guides(color=guide_legend(title="Volume"))
    )


# group by & sort descending
df_sorted_tran = (
    df6.groupby("Country_Name")
    .sum()
    .sort_values("Total_Transactions", ascending=False)
    .reset_index()
)

df8 = pd.merge(
    left=df,
    right=df_sorted_tran[["Country_Name", "Total_Transactions"]],
    on="Country_Name",
    how="left",
)
df8["Total_Transactions"].fillna("Other", inplace=True)
df8["Total_Transactions"] = np.where(
    df8["Total_Transactions"] != "Other", df["Country_Name"], "Other"
)
df8.rename(columns={"Total_Transactions": "Top_Transactions"}, inplace=True)
df8 = df8.loc[df8["Top_Transactions"] != "Other"]


def get_credit_transaction_volume_per_category():
    df8_pivot = pd.pivot_table(
        df8, index=["customer_country", "category"], values="amount", aggfunc=["count"]
    ).reset_index()
    df8_pivot1 = df8_pivot.reset_index()
    df8_pivot1.columns = ["id", "customer_country", "category", "count"]

    return (
        ggplot(df8_pivot1, aes(x="reorder(customer_country, count)", y="category"))
        + geom_point(aes(size="count", color="count"), alpha=0.7)
        + theme(legend_title=element_blank())
        + scale_color_cmap(cmap_name="plasma")
        + scale_y_discrete(limits=reversed)
        + scale_size(range=(0, 12))
        + ggtitle("Transaction Volume per Category and Top 25 Countries")
        + labs(y="Category", x="Country")
        + theme(figure_size=(12, 5))
    )


def get_credit_transaction_vlolume_per_hour_top_countries():
    df8_pivot2 = pd.pivot_table(
        df8, index=["customer_country", "hour"], values="amount", aggfunc=["count"]
    ).reset_index()
    df8_pivot2 = df8_pivot2.reset_index()
    df8_pivot2.columns = ["id", "customer_country", "hour", "count"]
    return (
        ggplot(df8_pivot2)
        + geom_tile(
            aes(x="hour", y="reorder(customer_country,count)", fill="count"), alpha=0.7
        )
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Transaction Volume per Hour by Top Countries")
        + labs(y="Top 25 Country", x="Hour")
        + theme(figure_size=(12, 4))
    )


def get_credit_transaction_volume_per_category_2():
    df8_pivot3 = pd.pivot_table(
        df8, index=["category", "hour"], values="amount", aggfunc=["count"]
    ).reset_index()
    df8_pivot3 = df8_pivot3.reset_index()
    df8_pivot3.columns = ["id", "category", "hour", "count"]

    return (
        ggplot(df8_pivot3)
        + geom_tile(aes(x="hour", y="reorder(category,count)", fill="count"), alpha=0.7)
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Transaction Volume per Hour by Category")
        + labs(y="Product Category", x="Hour")
        + theme(figure_size=(12, 4))
    )


# Average Ticket Size
def average_ticket_size_geographical_plot():
    chart = ggplot(data=df6, mapping=aes(fill="Avg_Ticket"))
    map_proj = geom_map(alpha=0.7)
    labels = labs(title="Average Ticket Value by Country of Origin")
    theme_details = theme(figure_size=(12, 6))
    fill_colormap = scale_fill_cmap(cmap_name="plasma")
    color_colormap = scale_color_cmap(cmap_name="plasma")
    world_map_card = (
        chart
        + map_proj
        + labels
        + theme_details
        + fill_colormap
        + color_colormap
        + theme(panel_background=element_blank())
    )
    return world_map_card


def average_ticket_size_top_n_countries():
    # Selecting Top XX nationalities
    # number of top-n you want
    n = 25

    # group by & sort descending
    df_sorted_avg = (
        df6.groupby("customer_country")
        .sum()
        .sort_values("Avg_Ticket", ascending=False)
        .reset_index()
    )
    # rename rows other than top-n to 'Others'
    df_sorted_avg.loc[df_sorted_avg.index >= n, "customer_country"] = "Others"

    df_sorted_avg1 = df_sorted_avg.loc[df_sorted_avg["customer_country"] != "Others"]
    # re-group by again
    # df_sorted.groupby('customer_country').mean()

    return (
        # Bar Chart for top 25 countries ordered by Average Ticket Value
        ggplot(
            df_sorted_avg1,
            aes(
                x="reorder(customer_country, Avg_Ticket, fun=sum)",
                y="Avg_Ticket",
                fill="Avg_Ticket",
            ),
        )
        + geom_bar(stat="identity", alpha=0.7)
        + scale_x_discrete()
        + coord_flip()
        + scale_fill_cmap(cmap_name="plasma")
        + ggtitle("Average Ticket Value Top 25 Countries")
        + labs(y="Average Ticket Value ", x="Top 25 Countries")
        + guides(color=guide_legend(title="Avg Ticket Value"))
    )


n = 25

# group by & sort descending
df_sorted_avg = (
    df6.groupby("customer_country")
    .sum()
    .sort_values("Avg_Ticket", ascending=False)
    .reset_index()
)
# rename rows other than top-n to 'Others'
df_sorted_avg.loc[df_sorted_avg.index >= n, "customer_country"] = "Others"

df_sorted_avg1 = df_sorted_avg.loc[df_sorted_avg["customer_country"] != "Others"]

df9 = pd.merge(
    left=df,
    right=df_sorted_avg1[["customer_country", "Avg_Ticket"]],
    on="customer_country",
    how="left",
)
df9["Avg_Ticket"].fillna("Other", inplace=True)
df9["Avg_Ticket"] = np.where(
    df9["Avg_Ticket"] != "Other", df9["customer_country"], "Other"
)
df9.rename(columns={"Avg_Ticket": "Top_Ticket"}, inplace=True)
df9 = df9.loc[df9["Top_Ticket"] != "Other"]


def average_ticket_size_category_top_n_countires():
    df9_pivot = pd.pivot_table(
        df9, index=["Top_Ticket", "category"], values="amount", aggfunc=["mean"]
    ).reset_index()
    df9_pivot1 = df9_pivot.reset_index()
    df9_pivot1.columns = ["id", "Top_Ticket", "category", "mean"]

    return (
        ggplot(df9_pivot1, aes("Top_Ticket", "category"))
        + geom_point(aes(size="mean", color="mean"), alpha=0.7)
        + theme(legend_title=element_blank())
        + scale_color_cmap(cmap_name="plasma")
        + scale_y_discrete(limits=reversed)
        + scale_size(range=(0, 12))
        + ggtitle("Average Ticket per Category and Top 25 Countries")
        + labs(y="Category", x="Country")
        + theme(figure_size=(12, 5))
    )


def average_ticket_size_per_hour_top_25_countries():
    df9_pivot2 = pd.pivot_table(
        df9, index=["customer_country", "hour"], values="amount", aggfunc=["mean"]
    ).reset_index()
    df9_pivot2 = df9_pivot2.reset_index()
    df9_pivot2.columns = ["id", "customer_country", "hour", "mean"]

    return (
        ggplot(df9_pivot2)
        + geom_tile(
            aes(x="hour", y="reorder(customer_country,mean)", fill="mean"), alpha=0.7
        )
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Average Ticket per Hour by Top Countries")
        + labs(y="Top 25 Country", x="Hour")
        + theme(figure_size=(12, 4))
    )


def avergage_ticket_per_categroy():
    df9_pivot3 = pd.pivot_table(
        df9, index=["category", "hour"], values="amount", aggfunc=["mean"]
    ).reset_index()
    df9_pivot3 = df9_pivot3.reset_index()
    df9_pivot3.columns = ["id", "category", "hour", "mean"]

    return (
        ggplot(df9_pivot3)
        + geom_tile(aes(x="hour", y="reorder(category,mean)", fill="mean"), alpha=0.7)
        + scale_fill_cmap(cmap_name="plasma")
        + theme(legend_title=element_blank())
        + ggtitle("Average Ticket per Hour by Category")
        + labs(y="Top 25 Country", x="Hour")
        + theme(figure_size=(12, 4))
    )


def avergae_ticket_value_per_Category():
    df9_pivot4 = pd.pivot_table(
        df9, index=["category"], values="amount", aggfunc=["mean"]
    ).reset_index()
    df9_pivot4 = df9_pivot4.reset_index()
    df9_pivot4.columns = ["id", "category", "mean"]

    return (
        ggplot(df9_pivot4, aes(x="reorder(category,mean)", y="mean", fill="mean"))
        + geom_bar(stat="identity", alpha=0.7)
        + scale_x_discrete()
        + coord_flip()
        + scale_fill_cmap(cmap_name="plasma")
        + ggtitle("Average Ticket Value per category")
        + labs(y="Average Ticket Value ", x="Category")
        + guides(color=guide_legend(title="Avg Ticket Value"))
    )


if __name__ == "__main__":
    p_chart = get_pareto_chart()
    # print(f"{p_chart=}")
    g_chart = get_geographical_plot()
    # print(f"{g_chart=}")
    # print(f"{type(g_chart)=}")
    # print(f"{dir(g_chart)=}")
    # print(g_chart.save(filename="./assets/total_expenditure_geographical_chart.png"))
    top_chart = get_top_countries_based_total_expenses()
    # print(f"{top_chart=}")
    top_chart_2 = get_top_nations_based_total_expenses_vs_product_category()
    # print(f"{top_chart_2=}")
    per_hour = get_expenditure_per_hour_25_countries()
    # print(get_expenditure_per_hour_25_countries())
    per_hour_per_category = get_expenditure_per_hour_per_category()
    # print(per_hour_per_category())
    credit_pareto_analysis_chart = get_credit_pareto_analysis_chart()
    # print()
    credit_georgraphical_chart = get_credit_georgraphical_chart()
    # print()
    credit_top_countries = get_credit_top_countries()
    # print()
    credit_transaction_volume_per_category = (
        get_credit_transaction_volume_per_category()
    )
    # print()
    credit_transaction_vlolume_per_hour_top_countries = (
        get_credit_transaction_vlolume_per_hour_top_countries()
    )
    # print()
    credit_transaction_volume_per_category_2 = (
        get_credit_transaction_volume_per_category_2()
    )
    # print()
