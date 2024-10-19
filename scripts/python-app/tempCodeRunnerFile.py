import pandas as pd
import numpy as np
import os
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pyarrow

#-----Read in and set up data
url = 'https://raw.githubusercontent.com/statzenthusiast921/setlist_analysis/main/data/setlists/full_setlist_df.parquet'
setlist_df = pd.read_parquet(url, engine='pyarrow')
setlist_df['Date'] = pd.to_datetime(setlist_df['Date'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')


#----- Artist choices
artist_choices = np.sort(setlist_df['ArtistName'].unique())

#----- Country choices
country_choices = np.sort(setlist_df['Country'].unique())

#----- Artist --> Country Dictionary
df_for_dict = setlist_df[['ArtistName','Country']]
df_for_dict = df_for_dict.drop_duplicates(keep='first')
artist_country_dict = df_for_dict.groupby('ArtistName')['Country'].apply(list).to_dict()



#----- Define style for different pages in app
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'color':'white',
    'backgroundColor': '#222222'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#626ffb',
    'color': 'white',
    'padding': '6px'
}



app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Welcome',value='tab-1',style=tab_style, selected_style=tab_selected_style,
               children=[
                   html.Div([
                       html.H1(dcc.Markdown('''**Welcome to my Setlist Dashboard!**''')),
                       html.Br()
                   ]),
                   
                   html.Div([
                        html.P(dcc.Markdown('''**What is the purpose of this dashboard?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("This dashboard was created as a tool to s.",style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What data is being used for this analysis?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   
                   html.Div([
                       html.P("The data ",style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What are the limitations of this data?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("1.) Blah",style={'color':'white'}),

                   ])


               ]),
        dcc.Tab(label='Concerts',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown1',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        )
                    ], width = 6),
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown2',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width = 6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='card1')
                    ],width=4),
                    dbc.Col([
                        dbc.Card(id='card2')
                    ],width=4),
                    dbc.Col([
                        dbc.Card(id='card3')
                    ],width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='concert_map')
                    ], width = 6),
                    dbc.Col([
                        html.Div(id='city_list')
                    ], width = 6)
                ])
            ]
        ),
        dcc.Tab(label='Historical Setlists',value='tab-3',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        
                    ], width =6),
                    dbc.Col([
                  
                    ], width = 6),
                    dbc.Col([
                    ])
                ])
            ]
        ),
        dcc.Tab(label='Subject 3',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                       
                    ], width =6),
                    dbc.Col([
               
                    ], width =6),
                    dbc.Col([
                     
                    ], width = 12),
                  

                ]),
                    dbc.Col([

                    ], width = 6),
                    dbc.Col([
                    
                    ], width = 6)

                ])
            ]
        )
    ])


@app.callback(
    Output('dropdown2', 'options'),  # --> filter country
    Output('dropdown2', 'value'),
    Input('dropdown1', 'value')  # --> choose artist
)
def set_parent_route_options(selected_artist):
    if selected_artist in artist_country_dict:
        available_countries = artist_country_dict[selected_artist]
        # Check if 'United States' is in the available countries
        if 'United States' in available_countries:
            default_value = 'United States'
        else:
            default_value = available_countries[0]  # Use the first country if 'United States' is not available
        return [{'label': i, 'value': i} for i in available_countries], default_value
    else:
        return [], None


#----- Cards for Concert Map tab
@app.callback(
    Output('card1','children'),
    Output('card2','children'),
    Output('card3','children'),
    Input('dropdown1','value'),
    Input('dropdown2','value')
)

def cards_against_concerts(dd1, dd2):


    filtered_df = setlist_df[setlist_df['ArtistName']==dd1]
    filtered_df = filtered_df[filtered_df['Country']==dd2]

    num_concerts_per_country = filtered_df[['RecordID']].drop_duplicates(keep = "first")

    #Metric 1 --> # of episodes
    metric1 = num_concerts_per_country.shape[0]

    #Metric 2 --> avg distance travelling away
    num_concerts_per_country['City_State'] = filtered_df['City'] + ', ' + filtered_df['State']
    metric2 = len(num_concerts_per_country['City_State'].unique())

    #Metric 3 --> max distance
    num_songs_per_country = filtered_df[['City','State','RecordID']]
    metric3 = int(num_songs_per_country.shape[0])


    card1 = dbc.Card([
            dbc.CardBody([
                html.P(f'# concerts in {dd2}'),
                html.H5(f"{metric1}")
            ])
        ],
        style={'display': 'inline-block',
            'width': '100%',
            'text-align': 'center',
            'background-color': '#70747c',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':16},
        outline=True)

    card2 = dbc.Card([
            dbc.CardBody([
                html.P(f'# cities played in {dd2}'),
                html.H5(f"{metric2}")
            ])
        ],
        style={'display': 'inline-block',
            'width': '100%',
            'text-align': 'center',
            'background-color': '#70747c',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':16},
        outline=True)

    card3 = dbc.Card([
            dbc.CardBody([
                html.P(f'# songs played in {dd2}'),
                html.H5(f"{metric3}")
            ])
        ],
        style={'display': 'inline-block',
            'width': '100%',
            'text-align': 'center',
            'background-color': '#70747c',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':16},
        outline=True)

    return card1, card2, card3


#----- TAB 2: Concert Map
@app.callback(
    Output('concert_map', 'figure'),
    Input('dropdown1', 'value'),
    Input('dropdown2','value'),
)
def concert_map_refresh(dd1, dd2):

    filtered_df = setlist_df[setlist_df['ArtistName'] == dd1]
    filtered_df = filtered_df[filtered_df['Country'] == dd2]


   # Deduplicate by 'RecordID' column
    filtered_df = filtered_df.drop_duplicates(subset='RecordID')

    # Group by the required columns and count the number of rows for each group
    concert_map_df = filtered_df.groupby(['City', 'State', 'Country', 'Latitude', 'Longitude'], as_index=False).size()

    # Rename the 'size' column to something more descriptive like 'Concert Count'
    concert_map_df.rename(
        columns={'size': '# Concerts'}, 
        inplace=True
    )

    fig = px.density_map(
        concert_map_df,
        lat='Latitude',
        lon='Longitude',
        z='# Concerts',  # Size or frequency of events
        radius=10,  # Adjust the radius of the heat points
        #center=dict(lat=37.0902, lon=-95.7129),  # Center on US, adjust based on your data
        zoom=3,  # Zoom level for the map
        hover_data={
            'City': True,
            'State' : True,
            'Country' : True,
            '# Concerts': True ,
            'Latitude': False,  
            'Longitude': False
            }#,
            # labels={
            #     'business_line':'Business Line',
            #     'parent_route':'Parent Route',
            #     'station_name':'Station Name',
            #     'year':'Year',
            #     'rides':'Rides'
            # }
        )
    fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )
    return fig


@app.callback(
    Output('city_list','children'),
    Input('dropdown1','value'),
    Input('dropdown2','value')
)
def table(dd1,dd2):

    filtered_df = setlist_df[setlist_df['ArtistName']==dd1]
    filtered_df = filtered_df[filtered_df['Country']==dd2]

    table_df = filtered_df[['City','State','RecordID']].drop_duplicates(keep = "first", subset = 'RecordID')
    table_df = table_df.groupby(['City','State']).size().reset_index(name='count')


    table_df = table_df.rename(columns={table_df.columns[2]: "# Concerts"})
    table_df = table_df.sort_values(['# Concerts', 'City', 'State'], ascending=[False, False, False])


    return html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in table_df.columns],
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
                style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'},
                #filter_action='native',
                style_data={'width': '125px', 'minWidth': '125px', 'maxWidth': '125px','overflow': 'hidden','textOverflow': 'ellipsis'},
                sort_action='native',sort_mode="multi",
                page_action="native", page_current= 0,page_size= 14,                     
                data=table_df.to_dict('records')
            )
        ])
    
    


if __name__=='__main__':
	app.run_server()