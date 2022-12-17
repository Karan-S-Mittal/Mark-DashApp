import dash
from dash_extensions.enrich import DashProxy, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

# notebook modules
from notebook_code import (
    get_pareto_chart,
    get_geographical_plot,
    get_top_countries_based_total_expenses,
    get_top_nations_based_total_expenses_vs_product_category,
    get_expenditure_per_hour_25_countries,
    get_expenditure_per_hour_per_category,
    # Credit
    get_credit_pareto_analysis_chart as credit_pareto_analysis_chart,
    get_credit_georgraphical_chart,
    get_credit_top_countries as credit_top_countries,
    get_credit_transaction_volume_per_category as credit_transaction_volume_per_category,
    get_credit_transaction_vlolume_per_hour_top_countries,
    get_credit_transaction_volume_per_category_2,
    # Average_ticket_size
    average_ticket_size_geographical_plot,
    average_ticket_size_top_n_countries,
    average_ticket_size_category_top_n_countires,
    average_ticket_size_per_hour_top_25_countries,
    avergage_ticket_per_categroy,
    avergae_ticket_value_per_Category,
    # Daily Purchase
)

app = DashProxy(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,",
        }
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server


# Total Expenditure images
# -------------------------------------------
geographical_plot_path = "./assets/total_expenditure_geographical_chart.png"
top_countries_based_total_expenses = (
    "./assets/get_top_countries_based_total_expenses.png"
)
top_nations_based_total_expenses_vs_product_category = (
    "./assets/get_top_nations_based_total_expenses_vs_product_category.png"
)
expenditure_per_hour_25_countries = "./assets/get_expenditure_per_hour_25_countries.png"
expenditure_per_hour_per_category = "./assets/get_expenditure_per_hour_per_category.png"

# Run Graph functions
get_geographical_plot().save(filename=geographical_plot_path, dpi=1000)
get_top_countries_based_total_expenses().save(
    filename=top_countries_based_total_expenses, dpi=1000
)
get_top_nations_based_total_expenses_vs_product_category().save(
    filename=top_nations_based_total_expenses_vs_product_category, dpi=1000
)
get_expenditure_per_hour_25_countries().save(
    filename=expenditure_per_hour_25_countries, dpi=1000
)
get_expenditure_per_hour_per_category().save(
    filename=expenditure_per_hour_per_category, dpi=1000
)

# layout
total_expenditure_layout = dmc.Stack(
    [
        dmc.Title("Total Expenditure"),
        html.Hr(),
        dbc.Card(dcc.Graph(figure=get_pareto_chart())),
        html.Hr(),
        dbc.Card(html.Img(src=geographical_plot_path)),
        html.Hr(),
        dbc.Card(html.Img(src=top_countries_based_total_expenses)),
        html.Hr(),
        dbc.Card(html.Img(src=top_nations_based_total_expenses_vs_product_category)),
        html.Hr(),
        dbc.Card(html.Img(src=expenditure_per_hour_25_countries)),
        html.Hr(),
        dbc.Card(html.Img(src=expenditure_per_hour_per_category)),
    ]
)

# Credit
# -------------------------------------------
credit_top_countries_path = "./assets/credit_top_countries.png"
credit_georgraphical_chart = "./assets/get_credit_georgraphical_chart.png"
credit_transaction_volume_per_category_path = (
    "./assets/credit_transaction_volume_per_category.png"
)
credit_transaction_vlolume_per_hour_top_countries_path = (
    "./assets/get_credit_transaction_vlolume_per_hour_top_countries.png"
)
credit_transaction_volume_per_category_2 = (
    "./assets/get_credit_transaction_volume_per_category_2.png"
)
# run graph functions
credit_top_countries().save(filename=credit_top_countries_path, dpi=1000)
get_credit_georgraphical_chart().save(filename=credit_georgraphical_chart, dpi=1000)
credit_transaction_volume_per_category().save(
    filename=credit_transaction_volume_per_category_path, dpi=1000
)
get_credit_transaction_vlolume_per_hour_top_countries().save(
    filename=credit_transaction_vlolume_per_hour_top_countries_path, dpi=1000
)
get_credit_transaction_volume_per_category_2().save(
    filename=credit_transaction_volume_per_category_2, dpi=1000
)

# layout
credit_layout = dmc.Stack(
    [
        dmc.Title("Credit Card Transactions Volume"),
        html.Hr(),
        dbc.Card(dcc.Graph(figure=credit_pareto_analysis_chart())),
        html.Hr(),
        dbc.Card(html.Img(src=credit_georgraphical_chart)),
        html.Hr(),
        dbc.Card(html.Img(src=credit_top_countries_path)),
        html.Hr(),
        dbc.Card(html.Img(src=credit_transaction_volume_per_category_path)),
        html.Hr(),
        dbc.Card(html.Img(src=credit_transaction_vlolume_per_hour_top_countries_path)),
        html.Hr(),
        dbc.Card(html.Img(src=credit_transaction_volume_per_category_2)),
    ]
)

# Average Ticket Size
average_ticket_size_geographical_plot_png = (
    "./assets/average_ticket_size_geographical_plot.png"
)
average_ticket_size_top_n_countries_png = (
    "./assets/average_ticket_size_top_n_countries.png"
)
average_ticket_size_category_top_n_countires_png = (
    "./assets/average_ticket_size_category_top_n_countires.png"
)
average_ticket_size_per_hour_top_25_countries_png = (
    "./assets/average_ticket_size_per_hour_top_25_countries.png"
)
avergage_ticket_per_categroy_png = "./assets/avergage_ticket_per_categroy.png"
avergae_ticket_value_per_Category_png = "./assets/avergae_ticket_value_per_Category.png"

# run graph functions
average_ticket_size_geographical_plot().save(
    filename=average_ticket_size_geographical_plot_png, dpi=1000
)
average_ticket_size_top_n_countries().save(
    filename=average_ticket_size_top_n_countries_png, dpi=1000
)
average_ticket_size_category_top_n_countires().save(
    filename=average_ticket_size_category_top_n_countires_png, dpi=1000
)
average_ticket_size_per_hour_top_25_countries().save(
    filename=average_ticket_size_per_hour_top_25_countries_png, dpi=1000
)
avergage_ticket_per_categroy().save(filename=avergage_ticket_per_categroy_png, dpi=1000)
avergae_ticket_value_per_Category().save(
    filename=avergae_ticket_value_per_Category_png, dpi=1000
)

# layout
average_ticket_size_layout = dmc.Stack(
    [
        dmc.Title("Average Ticket Size"),
        html.Hr(),
        dbc.Card(html.Img(src=average_ticket_size_geographical_plot_png)),
        html.Hr(),
        dbc.Card(html.Img(src=average_ticket_size_top_n_countries_png)),
        html.Hr(),
        dbc.Card(html.Img(src=average_ticket_size_category_top_n_countires_png)),
        html.Hr(),
        dbc.Card(html.Img(src=average_ticket_size_per_hour_top_25_countries_png)),
        html.Hr(),
        dbc.Card(html.Img(src=avergage_ticket_per_categroy_png)),
        html.Hr(),
        dbc.Card(html.Img(src=avergae_ticket_value_per_Category_png)),
    ]
)

# layout
app.layout = dbc.Container(
    children=[
        html.H3("Mark's Dashboard"),
        html.Hr(),
        dmc.Tabs(
            orientation="horizontal",
            children=[
                dmc.TabsList(
                    [
                        dmc.Tab("Total Expenditure", value="total-expenditure"),
                        dmc.Tab(
                            "Credit Card Transactions Volume",
                            value="credit-card-transactions-volume",
                        ),
                        dmc.Tab("Average Ticket Size", value="average-ticket-size"),
                    ]
                ),
                dmc.TabsPanel(total_expenditure_layout, value="total-expenditure"),
                dmc.TabsPanel(
                    credit_layout,
                    value="credit-card-transactions-volume",
                ),
                dmc.TabsPanel(average_ticket_size_layout, value="average-ticket-size"),
            ],
        ),
    ],
    fluid=True,
)

# running
if __name__ == "__main__":
    app.run(debug=True)
