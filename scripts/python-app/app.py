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
import datetime

#-----Read in and set up data
url = 'https://raw.githubusercontent.com/statzenthusiast921/setlist_analysis/main/data/setlists/full_setlist_df.parquet'
setlist_df = pd.read_parquet(url, engine='pyarrow')
setlist_df['Date'] = pd.to_datetime(setlist_df['Date'], format='%m/%d/%y')


#----- Choices for dropdown menus
artist_choices = np.sort(setlist_df['ArtistName'].unique())
country_choices = np.sort(setlist_df['Country'].unique())
state_choices = np.sort(setlist_df['State'].unique())
city_choices = np.sort(setlist_df['City'].unique())

#----- Artist --> Country Dictionary
df_for_dict = setlist_df[['ArtistName','Country']]
df_for_dict = df_for_dict.drop_duplicates(keep='first')
artist_country_dict = df_for_dict.groupby('ArtistName')['Country'].apply(list).to_dict()

#----- Country --> State Dictionary
df_for_dict2 = setlist_df[['Country','State']]
df_for_dict2 = df_for_dict2.drop_duplicates(keep='first')
country_state_dict = df_for_dict2.groupby('Country')['State'].apply(list).to_dict()

#----- State --> City Dictionary
df_for_dict3 = setlist_df[['State','City']]
df_for_dict3 = df_for_dict3.drop_duplicates(keep='first')
state_city_dict = df_for_dict3.groupby('State')['City'].apply(list).to_dict()



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
                        dcc.Dropdown(
                            id='dropdown3',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        ),
                        dcc.Dropdown(
                            id='dropdown4',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        ),
                        dcc.Dropdown(
                            id='dropdown5',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in state_choices],
                            value=state_choices[0]
                        ),
                        dcc.Dropdown(
                            id='dropdown6',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in city_choices],
                            value=city_choices[0]
                        )

                        
                    ], width =6),
                    dbc.Col([
                        dcc.Slider(
                            id='date_slider',
                            min=0,  # Will be updated dynamically
                            max=1,  # Will be updated dynamically
                            step=None,  # Discrete values only
                            marks={},  # Will be updated dynamically
                            value=0  # Initial value
                        ),
                    ], width = 6),
                    dbc.Col([
                        html.Div(id='setlist_list')
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



#----------------------------------------------------------------------------------#
#------------------------------- TAB 2: Concert Map -------------------------------#
#----------------------------------------------------------------------------------#

#----- 2a.) Cards above map
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
                html.P(f'# concerts'),
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
                html.P(f'# cities played'),
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
                html.P(f'# songs played'),
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


#----- 2b.) Concert Map 

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
        z='# Concerts',  
        radius=16,  
        zoom=2, 
        hover_data={
            'City': True,
            'State' : True,
            'Country' : True,
            '# Concerts': True ,
            'Latitude': False,  
            'Longitude': False
            }
        )
    fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )
    return fig
#----- 2c.) Table with city counts per country 

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

#----- 2d.) Callback for setting the country options based on artist selection

@app.callback(
    Output('dropdown2', 'options'),  # --> filter country
    Output('dropdown2', 'value'),
    Input('dropdown1', 'value')  # --> choose artist
)
def set_country_options_from_artist_selection(selected_artist):
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


#-------------------------------------------------------------------------------#
#------------------------------- TAB 3: Setlists -------------------------------#
#-------------------------------------------------------------------------------#

#----- 3a.) Callback for setting the state options based on country selection
@app.callback(
    Output('dropdown4', 'options'),  # --> filter country
    Output('dropdown4', 'value'),
    Input('dropdown3', 'value')  # --> choose artist
)
def set_country_options_for_artist(selected_artist):
        return [{'label': i, 'value': i} for i in artist_country_dict[selected_artist]], artist_country_dict[selected_artist][0]


@app.callback(
    Output('dropdown5', 'options'),  # --> filter state
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value')  # --> choose country
)
def set_state_options_in_country(selected_country):
        return [{'label': i, 'value': i} for i in country_state_dict[selected_country]], country_state_dict[selected_country][0]

#----- 3b.) Callback for setting the city options based on state selection

@app.callback(
    Output('dropdown6', 'options'),  # --> filter city
    Output('dropdown6', 'value'),
    Input('dropdown5', 'value')  # --> choose state
)
def set_city_options_in_state(selected_state):
        return [{'label': i, 'value': i} for i in state_city_dict[selected_state]], state_city_dict[selected_state][0]
    


@app.callback(
    Output('date_slider', 'min'),
    Output('date_slider', 'max'),
    Output('date_slider', 'marks'),
    Output('date_slider', 'value'),
    Output('date_slider', 'data'),  # New Output to pass unique_dates

    Input('dropdown3', 'value'),
    Input('dropdown4', 'value'),
    Input('dropdown5', 'value'),
    Input('dropdown6', 'value')
)
def update_date_slider(artist, country, state, city):
    #global unique_dates


    # Filter the dataframe based on dropdown values
    filtered_df = setlist_df[
        (setlist_df['ArtistName'] == artist) &
        (setlist_df['Country'] == country) &
        (setlist_df['State'] == state) &
        (setlist_df['City'] == city)
    ]
    
    # Ensure there are no issues with an empty dataframe
    if filtered_df.empty:
        return 0, 1, {}, 0

    # Get the unique concert dates and sort them
    unique_dates = sorted(filtered_df['Date'].dt.date.unique())
    
    # Create marks for the slider
    marks = {i: str(date) for i, date in enumerate(unique_dates)}
    
    return 0, len(unique_dates) - 1, marks, 0, unique_dates  # Return unique_dates list




@app.callback(
    Output('setlist_list','children'),
    Input('dropdown3','value'),
    Input('dropdown4','value'),
    Input('dropdown5','value'),
    Input('dropdown6','value'),
    Input('date_slider','value'),
    Input('date_slider', 'data')  # New input for unique_dates

)
def table(dd3,dd4, dd5, dd6, slider_value, unique_dates):


    filtered_df = setlist_df[setlist_df['ArtistName']==dd3]
    filtered_df = filtered_df[filtered_df['Country']==dd4]
    filtered_df = filtered_df[filtered_df['State']==dd5]
    filtered_df = filtered_df[filtered_df['City']==dd6]

    if not unique_dates or slider_value is None or not (0 <= slider_value < len(unique_dates)):
            return html.Div("No data available for the selected filters.")



    # Filter by selected date using the slider
    selected_date = unique_dates[slider_value]
    filtered_df = filtered_df[filtered_df['Date'] == pd.to_datetime(selected_date)]

    if filtered_df.empty:
        return html.Div("No setlist available for the selected filters and date.")


    # Create the setlist table with filtered data
    setlist_table = filtered_df[['VenueName', 'TourName', 'City', 'State', 'Country', 'song_num', 'album', 'SongName']]
    setlist_table = setlist_table.sort_values('song_num', ascending=True)
    setlist_table['song_num'] += 1

    #----- Debug 1: print filtered data
    print(f"Check 1:\n{filtered_df.shape}")

    setlist_table = setlist_table.rename(
        columns={
            setlist_table.columns[0]: "Venue Name",
            setlist_table.columns[1]: "Tour Name",
            setlist_table.columns[5]: "Song #",
            setlist_table.columns[6]: "Album",
            setlist_table.columns[7]: "Song Name"
        }
    )

    return html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in setlist_table.columns],
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
                style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'},
                #filter_action='native',
                style_data={'width': '125px', 'minWidth': '125px', 'maxWidth': '125px','overflow': 'hidden','textOverflow': 'ellipsis'},
                sort_action='native',sort_mode="multi",
                page_action="native", page_current= 0,page_size= 14,                     
                data=setlist_table.to_dict('records')
            )
        ])
    



if __name__=='__main__':
	app.run_server()