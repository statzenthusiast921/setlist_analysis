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
import scipy.stats as stats


#-----Read in and set up data
url = 'https://raw.githubusercontent.com/statzenthusiast921/setlist_analysis/main/data/setlists/full_setlist_df.parquet'
setlist_df = pd.read_parquet(url, engine='pyarrow')
setlist_df['Date'] = pd.to_datetime(setlist_df['Date'], format='%m/%d/%y')
setlist_df['Year'] = setlist_df['Date'].dt.year

#-----Remove covers
setlist_df = setlist_df[setlist_df['Cover']==""]


#----- Make df of emotions
setlist_emotions_df = setlist_df[['ArtistName','SongName','happy','angry','surprise','sad','fear']]
setlist_emotions_df = setlist_emotions_df.drop_duplicates(keep='first')
setlist_emotions_df = setlist_emotions_df.dropna()
setlist_emotions_df = setlist_emotions_df[~setlist_emotions_df['SongName'].str.contains('"The Lucky One"')]

#----- Read in processed rules-based setlist dataset
url2 = 'https://raw.githubusercontent.com/statzenthusiast921/setlist_analysis/refs/heads/main/data/setlists/ideal_setlists/all_setlists_rb.csv'
rb_setlist_df = pd.read_csv(url2)
rb_setlist_df = rb_setlist_df.drop(rb_setlist_df.columns[0], axis=1)
rb_setlist_df['Prioritized Emotion'] = rb_setlist_df['Prioritized Emotion'].fillna('None')

#----- Read in position by position dataset
url3 = 'https://raw.githubusercontent.com/statzenthusiast921/setlist_analysis/refs/heads/main/data/setlists/full_position_by_position_df.csv'
pos_by_pos_df = pd.read_csv(url3)
pos_by_pos_df['Emotion'] = pos_by_pos_df['Emotion'].fillna('None')


#----- Choices for dropdown menus
artist_choices = np.sort(setlist_df['ArtistName'].unique())
artist_choices = artist_choices[artist_choices != "Gorillaz"]

country_choices = np.sort(setlist_df['Country'].unique())
state_choices = np.sort(setlist_df['State'].unique())
city_choices = np.sort(setlist_df['City'].unique())



song_choices = np.sort(setlist_df['SongName'].unique())
year_choices = np.sort(setlist_df['Year'].unique())
emotion_choices = ['None','Angry','Fear','Happy','Sad','Surprise']
emotion_choices2 = ['Angry','Fear','Happy','Sad','Surprise']


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

#----- Artist --> Song Dictionary
df_for_dict4 = setlist_df[['ArtistName','SongName']]
df_for_dict4 = df_for_dict4.drop_duplicates(keep='first')
artist_song_dict = df_for_dict4.groupby('ArtistName')['SongName'].apply(list).to_dict()




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
                        dbc.Label('Choose an artist: '),
                        dcc.Dropdown(
                            id='dropdown3',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        ),
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose a country: '),
                        dcc.Dropdown(
                            id='dropdown4',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width = 6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a state: '),
                        dcc.Dropdown(
                            id='dropdown5',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in state_choices],
                            value=state_choices[0]
                        ),
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose a city: '),
                        dcc.Dropdown(
                            id='dropdown6',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in city_choices],
                            value=city_choices[0]
                        )    
                    ], width = 6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a concert date: '),
                        dcc.Slider(
                            id='date_slider',
                            min=0,  # Will be updated dynamically
                            max=1,  # Will be updated dynamically
                            step=None,  # Discrete values only
                            marks={},  # Will be updated dynamically
                            value=0  # Initial value
                        ),
                    ], width = 12),
                    dbc.Col([
                        html.Div(id='setlist_list')
                    ])
                ])
            ]
        ),
        dcc.Tab(label='Position Frequency',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                       dbc.Label('Choose an artist: '),
                        dcc.Dropdown(
                            id='dropdown7',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        ),
                    ], width =6),
                    dbc.Col([
                        dbc.Label('Choose a song: '),
                        dcc.Dropdown(
                            id='dropdown8',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in song_choices],
                            value=song_choices[0]
                        ),
               
                    ], width =6),
                    dbc.Col([
                        dbc.Label('Choose a year range: '),
                        dcc.RangeSlider(
                            id='rangeslider',
                            min=2000,
                            max=2024,
                            value=[2000, 2024],
                            step = 1,
                            allowCross=False,
                            marks={year: str(year) for year in range(2000, 2025)}  # Mark every year

                        )
                    ], width = 12),
                    dbc.Col([
                        dbc.Card(id='card4')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card5')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card6')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card7')
                    ],width=3),
                    dbc.Col([
                        dcc.Graph(id = 'position_frequency'),
                        html.Div(id = 'message_div')
                    ])
                ])
            ]
        ),
        dcc.Tab(label='Emotion Scores', value = 'tab-5', style = tab_style, selected_style=tab_selected_style,
            children = [
                dbc.Row([
                    dbc.Button("How does this work?",id='info2')
                ]),
                html.Div([
                    dbc.Modal(
                        children=[
                            dbc.ModalHeader("Emotion Scores"),
                            dbc.ModalBody(
                                children=[
                                    html.P('The emotion scores are processed using the text2emotion Python library through the following steps:'),
                                    html.P("1.) The input text is preprocessed to remove noise and prepare it for analysis by lowercasing, tokenizing, removing stopwords, removing punctuation, and stemming/lemmatizting."),
                                    html.P("2.) A predefined emotion lexicon maps words to emotions such as love --> Happy; anger --> Angry; sad --> Sad.  For each word in a song, the library checks its emotion association in the lexicon and accumulates emotion scores."),
                                    html.P("3.) The emotion scores for individual words are aggregated across the entire song.  Each emotion is assigned a numeric score based on the frequency and relevance of the associated words."),
                                    html.P('4.) The library outputs results among 5 key emotions: Happy, Angry, Sad, Fear, and Surprise.  The full song will receive scores for each of the 5 emotions which all sum to a value of 1.')
                                ]
                            ),
                            dbc.ModalFooter( 
                                dbc.Button("Close", id="close_info2")
                            ),
                        ],id="modal_info2", size="md"

                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose an artist: '),
                        dcc.Dropdown(
                            id='dropdown11',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        ),
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose an emotion: '),
                         dcc.Dropdown(
                            id='dropdown12',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in emotion_choices2],
                            value=emotion_choices2[0]
                        ),
                    ], width = 6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id = 'emotion_chart')
                    ])
                ])
            ]
        ),
        dcc.Tab(label='Rules-Based Setlists',value='tab-6',style=tab_style, selected_style=tab_selected_style,
            children = [
                dbc.Row([
                    dbc.Button("How does this work?",id='info1')
                ]),
                html.Div([
                    dbc.Modal(
                        children=[
                            dbc.ModalHeader("Rules-Based Setlist Process"),
                            dbc.ModalBody(
                                children=[
                                    html.P('The rules-based approach works as follows:'),
                                    html.P("1.) Calculate the median emotion scores (0-1) for each position in an artist's setlist.  Setlist lengths are determined from median length of historical setlists."),
                                    html.P('2.) Calculate emotion scores (0-1) for all songs for each artist.'),
                                    html.P('3.) Take scores from Step 1 for each successive position in the setlist and calculate the absolute differences from scores in Step 2.'),
                                    html.P('4.) Count # of emotions where result from Step 3 is within threshold of 0.05.  Filter down to songs with at least 3 emotions within tolerance threshold.'),
                                    html.P('5.) Sum up absolute differences across all emotions for songs resulting from Step 4 to get "Similarity Score".  Invert score for ease of interpretation.'),
                                    html.P("6.) Song with highest Similarity Score is chosen for the 1st position in the setlist.  Remove song from master list so it can not be chosen again."),
                                    html.P("7.) Repeat process for remaining positions in setlist.")

                                ]
                            ),
                            dbc.ModalFooter( 
                                dbc.Button("Close", id="close_info1")#,color='Secondary',className='me-1')
                            ),
                        ],id="modal_info1", size="md"

                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose an artist: '),
                        dcc.Dropdown(
                            id='dropdown9',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in artist_choices],
                            value=artist_choices[0]
                        ),
                    ], width = 3),
                    dbc.Col([
                        dbc.Label('Choose an emotion to prioritize: '),
                        dcc.Dropdown(
                            id='dropdown10',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in emotion_choices],
                            value=emotion_choices[0]
                        ),
                    ], width = 3),
                    dbc.Col([
                        dbc.Label('Choose a setlist position:'),
                        dcc.Slider(
                            id='position_slider',
                            min=1,  
                            max=15, 
                            step=1,  
                            marks={},  
                            value=1  
                        ),
                    ], width = 3),
                    dbc.Col([
                        dbc.Label('Switch views:'),
                        dcc.RadioItems(
                            id='radio-button-toggle',
                            options=[
                                {'label': ' Chart View', 'value': 'Chart View'},
                                {'label': ' Spotify View', 'value': 'Spotify View'},
                                {'label': ' Playlist View', 'value': 'Playlist View'}

                            ],
                        value='Chart View',  
                        labelStyle={'display': 'block'}  
                        ),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                         html.Div(id = 'output_toggle_container', children = [
                            dcc.Graph(id='closeness_score_chart'),
                            #----- Passion Pit
                            html.Iframe(
                                id = 'playlist1',
                                src="https://open.spotify.com/embed/playlist/2ZnH0vDq6zdxT8nqdSh2W9?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- TV on the Radio
                            html.Iframe(
                                id = 'playlist2',
                                src="https://open.spotify.com/embed/playlist/18yWRizPvyC7CSO7oedDZD?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Taylor Swift
                            html.Iframe(
                                id = 'playlist3',
                                src="https://open.spotify.com/embed/playlist/24Tyqqbn2QpvzYgOm1wspe?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Little Dragon
                            html.Iframe(
                                id = 'playlist4',
                                src="https://open.spotify.com/embed/playlist/40nRKuP7kl1pREQtWlZ6Rk?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Florence + the Machine
                            html.Iframe(
                                id = 'playlist5',
                                src="https://open.spotify.com/embed/playlist/6zSEB5Oq0ZRbIgWf99fo6L?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Dredg
                            html.Iframe(
                                id = 'playlist6',
                                src="https://open.spotify.com/embed/playlist/4wI3273u0X0xodPRWlgJkZ?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Tame Impala
                            html.Iframe(
                                id = 'playlist7',
                                src="https://open.spotify.com/embed/playlist/6fr5bhWEVn9Xr5xu7bKTOB?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            #----- Lord Huron
                            html.Iframe(
                                id = 'playlist8',
                                src="https://open.spotify.com/embed/playlist/32r6Wsrhrad237DVYclwgh?utm_source=generator",  
                                width="100%", height="380", style={'border': 'none', 'display': 'none', 'margin': '0 auto'},
                            ),
                            html.Div(id='playlist_table')

                         ])
                    ], width = 12),
                ]),
            ]

        ),
        dcc.Tab(label='ML-Based Setlists',value='tab-7',style=tab_style, selected_style=tab_selected_style,
            children = [
                dbc.Row([
                    dbc.Col([
                        
                    ])
                ])
            ]

        )
        
    ])

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
                html.P(f'# Concerts'),
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
                html.P(f'# Cities played'),
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
                html.P(f'# Songs played'),
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

    #----- Deduplicate by 'RecordID' column
    filtered_df = filtered_df.drop_duplicates(subset='RecordID')

    #----- Group by the required columns and count the number of rows for each group
    concert_map_df = filtered_df.groupby(['City', 'State', 'Country', 'Latitude', 'Longitude'], as_index=False).size()

    #----- Rename the 'size' column to something more descriptive like 'Concert Count'
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
    setlist_table = filtered_df[['VenueName', 'City', 'State', 'Country', 'song_num', 'album', 'SongName']]
    setlist_table = setlist_table.sort_values('song_num', ascending=True)
    setlist_table['song_num'] += 1

    setlist_table = setlist_table.rename(
        columns={
            setlist_table.columns[0]: "Venue Name",
            setlist_table.columns[4]: "Song #",
            setlist_table.columns[5]: "Album",
            setlist_table.columns[6]: "Song Name"
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


#----------------------------------------------------------------------------------#
#------------------------------- TAB 4: Position Frequency  -----------------------#
#----------------------------------------------------------------------------------#

    
#------ Position Frequency
@app.callback(
    Output('dropdown8', 'options'),  # --> filter song
    Output('dropdown8', 'value'),
    Input('dropdown7', 'value')  # --> choose artist
)
def set_song_options_per_artist(selected_artist):
        return [{'label': i, 'value': i} for i in artist_song_dict[selected_artist]], artist_song_dict[selected_artist][0]
    
@app.callback(
    Output('position_frequency','figure'),
    Output('position_frequency', 'style'),  # Output style to control chart visibility

    Output('message_div', 'children'),    
    Output('card4', 'children'),
    Output('card5', 'children'),
    Output('card6', 'children'),
    Output('card7', 'children'),
    Input('dropdown7','value'),
    Input('dropdown8','value'),
    Input('rangeslider','value')
)
def position_freq_chart(dd7, dd8, rs):

    filtered_df = setlist_df[setlist_df['ArtistName']==dd7]
    filtered_df = filtered_df[filtered_df['SongName']==dd8]
    filtered_df = filtered_df[filtered_df['Year']>= rs[0]]
    filtered_df = filtered_df[filtered_df['Year']<= rs[1]]
    
    song_freq_df = filtered_df['song_num'].value_counts().reset_index().sort_values(by = 'song_num')
    song_freq_df['song_num'] = song_freq_df['song_num'] + 1


    #----- Check if there's only one unique song_num value
    if len(song_freq_df['song_num'].unique()) <= 1:
        #----- Return empty figure and message if only one unique value
        empty_fig = go.Figure()
        hidden_style = {'display': 'none'} 

        message = f"### {dd8} by {dd7} has only one unique setlist position in the selected time range or was not played in the selected time range.  Choose a different song or different time range."
        return empty_fig, hidden_style, dcc.Markdown(message), None, None, None, None

    fig = px.bar(
        song_freq_df, 
        x='song_num', 
        y='count',
        title=f'Which positions did "{dd8}" by {dd7} occupy in the setlists between {rs[0]} & {rs[1]}',
        labels={'song_num':'Setlist Song Position','count':'Count'},
    )

    #----- Only attempt KDE if there’s more than one unique position
    if len(song_freq_df['song_num'].unique()) > 1:
        #----- Prepare data for KDE to get a more flexible multi-modal distribution
        song_positions = song_freq_df['song_num'].repeat(song_freq_df['count'])
        #----- Extend x-axis slightly beyond data range
        x_min = song_freq_df['song_num'].min() - 0.5
        x_max = song_freq_df['song_num'].max() + 0.5
        x_vals = np.linspace(x_min, x_max, 100)

        #----- Perform KDE and scale to match the total count of bars
        kde = stats.gaussian_kde(song_positions, bw_method=0.3)
        y_vals = kde(x_vals) * song_freq_df['count'].sum()

        #----- Add the distribution curve as a trace
        curve = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            fill='tozeroy',
            line=dict(color='red', dash='dash'),
            showlegend=False
        )
        #----- Update the figure with the distribution curve
        fig.add_trace(curve)


    #----- Metric for Card 4: Most played song
    metric_df = setlist_df[setlist_df['ArtistName']==dd7]

    #----- Songs that are not playing nice
    metric_df = metric_df[(metric_df['SongName']!= "")]

    metric_df = metric_df[metric_df['Year']>=rs[0]]
    metric_df = metric_df[metric_df['Year']<=rs[1]]

    metric4 = metric_df['SongName'].value_counts().reset_index()
    metric4_song_name = metric4['SongName'][0]
    metric4_song_played = metric4['count'][0]

    #----- Metric for Card 5: Most consistently placed song

    metric5_df = metric_df.groupby(['SongName', 'song_num']).size().reset_index(name='count')
    metric5_df['song_num'] = metric5_df['song_num'] + 1
    metric5 = metric5_df.loc[metric5_df['count'].idxmax()]
    metric5_name = metric5['SongName']
    metric5_position = metric5['song_num']
    metric5_count = metric5['count']


    #----- Metric for Card 6: Song most used as opener
    metric6_df = metric_df[['SongName','song_num']]
    metric6_df = metric6_df[metric6_df['song_num']==0]
    metric6 = metric6_df['SongName'].value_counts().reset_index()

    metric6_song_name = metric6['SongName'][0]
    metric6_song_played = metric6['count'][0]
    
    #----- Metric for Card 7: Song most used as closer
    metric7_df = metric_df[['RecordID','SongName','song_num']]
    metric7_df = metric7_df.groupby('RecordID').tail(1).reset_index(drop=True)
    metric7 = metric7_df['SongName'].value_counts().reset_index()

    metric7_song_name = metric7['SongName'][0]
    metric7_song_played = metric7['count'][0]


    card4 = dbc.Card([
            dbc.CardBody([
                html.P(f'Most Popular Song'),
                html.H5(f"'{metric4_song_name}'"),
                html.H5(f"{metric4_song_played} times")
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

    card5 = dbc.Card([
            dbc.CardBody([
                html.P(f'Most Consistently Placed Song'),
                html.H5(f"'{metric5_name}' ({metric5_position}) "),
                html.H5(f"{metric5_count} times")
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

    card6 = dbc.Card([
            dbc.CardBody([
                html.P(f'Song Most Used as Opener'),
                html.H5(f"'{metric6_song_name}'"),
                html.H5(f"{metric6_song_played} times")
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


    card7 = dbc.Card([
            dbc.CardBody([
                html.P(f'Song Most Used as Closer'),
                html.H5(f"'{metric7_song_name}'"),
                html.H5(f"{metric7_song_played} times")
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

    #----- Show the chart and clear the message
    visible_style = {'display': 'block'}  # Show the chart

    return fig,visible_style, "", card4, card5, card6, card7
#------------------------------------------------------------------#
#--------------------- TAB 5: Emotion Scores  ---------------------#
#------------------------------------------------------------------#
@app.callback(
    Output('emotion_chart','figure'),
    Input('dropdown11','value'),
    Input('dropdown12','value')
)
def emotion_chart(dd11,dd12):
    emotion_tbl = setlist_emotions_df[setlist_emotions_df['ArtistName']==dd11]
    emotion_tbl = emotion_tbl.rename(
            columns={
                emotion_tbl.columns[0]: "Artist Name",
                emotion_tbl.columns[1]: "Song Name",
                emotion_tbl.columns[2]: "Happy",
                emotion_tbl.columns[3]: "Angry",
                emotion_tbl.columns[4]: "Surprise",
                emotion_tbl.columns[5]: "Sad",
                emotion_tbl.columns[6]: "Fear"

            }
        )
  
    if 'Angry' in dd12:
        emotion_tbl = emotion_tbl[['Artist Name','Song Name','Angry']]
        emotion_tbl = emotion_tbl.sort_values('Angry', ascending=False)
        emotion_tbl = emotion_tbl.head(10)
     


        emotion_chart = px.bar(
            emotion_tbl,
            x='Song Name',
            y='Angry',
            color = 'Angry',
            color_continuous_scale=[
                    (0.0, 'lightblue'),  # Light blue for low values
                    (0.5, 'blue'),       # Midway blue for medium values
                    (1.0, 'darkblue')    # Dark blue for high values
            ],
            title = f'Top 10 Songs Ranked by {dd12} Scores'
        )

        return emotion_chart

    elif 'Surprise' in dd12:
        emotion_tbl = emotion_tbl[['Artist Name','Song Name','Surprise']]
        emotion_tbl = emotion_tbl.sort_values('Surprise', ascending=False)
        emotion_tbl = emotion_tbl.head(10)

        emotion_chart = px.bar(
            emotion_tbl,
            x='Song Name',
            y='Surprise',
            title = f'Top 10 Songs Ranked by {dd12} Scores',
            color = 'Surprise',
            color_continuous_scale=[
                    (0.0, 'lightblue'),  # Light blue for low values
                    (0.5, 'blue'),       # Midway blue for medium values
                    (1.0, 'darkblue')    # Dark blue for high values
            ],

        )

        return emotion_chart

    elif 'Sad' in dd12:
        emotion_tbl = emotion_tbl[['Artist Name','Song Name','Sad']]
        emotion_tbl = emotion_tbl.sort_values('Sad', ascending=False)
        emotion_tbl = emotion_tbl.head(10)

        emotion_chart = px.bar(
            emotion_tbl,
            x='Song Name',
            y='Sad',
            title = f'Top 10 Songs Ranked by {dd12} Scores',
            color = 'Sad',
            color_continuous_scale=[
                    (0.0, 'lightblue'),  # Light blue for low values
                    (0.5, 'blue'),       # Midway blue for medium values
                    (1.0, 'darkblue')    # Dark blue for high values
            ],

        )

        return emotion_chart

    elif 'Fear' in dd12:
        emotion_tbl = emotion_tbl[['Artist Name','Song Name','Fear']]
        emotion_tbl = emotion_tbl.sort_values('Fear', ascending=False)
        emotion_tbl = emotion_tbl.head(10)

        emotion_chart = px.bar(
            emotion_tbl,
            x='Song Name',
            y='Fear',
            title = f'Top 10 Songs Ranked by {dd12} Scores',
            color = 'Fear',
            color_continuous_scale=[
                    (0.0, 'lightblue'),  # Light blue for low values
                    (0.5, 'blue'),       # Midway blue for medium values
                    (1.0, 'darkblue')    # Dark blue for high values
            ],

        )

        return emotion_chart

    elif 'Happy' in dd12:
        emotion_tbl = emotion_tbl[['Artist Name','Song Name','Happy']]
        emotion_tbl = emotion_tbl.sort_values('Happy', ascending=False)
        emotion_tbl = emotion_tbl.head(10)

        emotion_chart = px.bar(
            emotion_tbl,
            x='Song Name',
            y='Happy',
            title = f'Top 10 Songs Ranked by {dd12} Scores',
            color = 'Happy',
            color_continuous_scale=[
                    (0.0, 'lightblue'),  # Light blue for low values
                    (0.5, 'blue'),       # Midway blue for medium values
                    (1.0, 'darkblue')    # Dark blue for high values
            ],

        )

        return emotion_chart


#----------------------------------------------------------------------------------#
#------------------------------- TAB 6: Rules Based Setlists  ---------------------#
#----------------------------------------------------------------------------------#

#----- Define length of slider based on setlist length
@app.callback(
    Output('position_slider', 'max'),
    Input('dropdown9', 'value')
)
def update_slider_max(dd9):
    if 'Little Dragon' in dd9:
        return 11
    elif 'Lord Huron' in dd9:
        return 17
    elif 'Taylor Swift' in dd9:
        return 15
    elif 'TV on the Radio' in dd9:
        return 12
    else:
        return 13

#----- Closeness Scores by Position Succession

@app.callback(
    Output('closeness_score_chart', 'figure'),
    Input('dropdown9', 'value'),
    Input('dropdown10','value'),
    Input('position_slider','value')
)
def update_closeness_chart(dd9, dd10, pos_slider):


    pos_df = pos_by_pos_df[pos_by_pos_df['ArtistName']==dd9]
    pos_df = pos_df[pos_df['Emotion']==dd10]

    #----- Sort by Closeness Score within Iteration
    pos_df = pos_df.groupby('Iteration').apply(lambda x: x.sort_values('Closeness_Score', ascending=True))
    pos_df = pos_df.reset_index(drop=True)

    #----- Calculate Row # 
    pos_df['Row #'] = pos_df.groupby('Iteration').cumcount() + 1

    pos_df = pos_df[pos_df['Iteration']==pos_slider]
    pos_df['Closeness_Score'] = 1 - pos_df['Closeness_Score']

    #pos_df['Closeness_Score'] = pos_df['Closeness_Score'].round(2)

    #----- Remove weird song names that are wrong
    pos_df = pos_df[~pos_df['name'].str.contains("Wrong Way", na=False)]


    pos_df = pos_df.head(10)

    chosen_song = pos_df['name'].values[0]

    fig = px.bar(
        pos_df, 
        x="Closeness_Score", y="name", orientation='h',
        title = f'Ranked Similarity Scores for Setlist Position: {pos_slider}',
            color = 'Closeness_Score',
            color_continuous_scale=[
                    (0.0, 'lightgreen'),  
                    (0.5, 'green'),       
                    (1.0, 'darkgreen')    
            ],
            labels={
                "name": "Song Name",
                "Closeness_Score": "Similarity Score"
                 },
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis=dict(range=[0, 1]),
        title=dict(
            xref='paper',  
            x=0.5,
            subtitle=dict(
                text=f"{chosen_song} is chosen for position {pos_slider} and then can not be chosen again.",
                font=dict(color="gray", size=13),
            ),      
        )

    )

    fig.update_traces(
    hovertemplate="Closeness Score %{x:.2f}<extra></extra>"
    )

    return fig


#----- Toggle between chart and playlist

@app.callback(
    Output('playlist1', 'style'), # Passion Pit
    Output('playlist2', 'style'), # TV on the Radio
    Output('playlist3', 'style'), # Taylor Swift
    Output('playlist4', 'style'), # Little Dragon
    Output('playlist5', 'style'), # Florence + the Machine
    Output('playlist6', 'style'), # Dredg
    Output('playlist7', 'style'), # Tame Impala
    Output('playlist8', 'style'), # Lord Huron


    Output('closeness_score_chart', 'style'),
    Output('playlist_table', 'style'),
    Input('radio-button-toggle', 'value'),
    Input('dropdown9', 'value')

)
def toggle_between_chart_and_playlist(radio_selection, dd9):
    if radio_selection == 'Chart View':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'},{'display': 'block'},{'display': 'none'}
    
    elif radio_selection == 'Spotify View' and "Passion Pit" in dd9:
        return {'display': 'block'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'},{'display': 'none'}
    
    elif radio_selection == 'Spotify View' and "TV on the Radio" in dd9:
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}
   
    elif radio_selection == 'Spotify View' and "Taylor Swift" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}
    
    elif radio_selection == 'Spotify View' and "Little Dragon" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}
    
    elif radio_selection == 'Spotify View' and "Florence + the Machine" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}
    
    elif radio_selection == 'Spotify View' and "Dredg" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}

    elif radio_selection == 'Spotify View' and "Tame Impala" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}

    elif radio_selection == 'Spotify View' and "Lord Huron" in dd9:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'},{'display': 'none'}
    
    elif radio_selection == "Playlist View":
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{'display': 'block'}




#----- Playlist tables
@app.callback(
    Output('playlist_table','children'),
    Input('dropdown9','value'),
    Input('dropdown10','value')
)
def table(dd9, dd10):

    pos_df = pos_by_pos_df[pos_by_pos_df['ArtistName']==dd9]
    pos_df = pos_df[pos_df['Emotion']==dd10]

    #----- Sort by Closeness Score within Iteration
    pos_df = pos_df.groupby('Iteration').apply(lambda x: x.sort_values('Closeness_Score', ascending=True))
    pos_df = pos_df.reset_index(drop=True)

    #----- Calculate Row # 
    pos_df['Row #'] = pos_df.groupby('Iteration').cumcount() + 1

    #----- Remove non 1st row values
    filtered_df = pos_df[pos_df['Row #']==1]

    filtered_df.rename(
        columns={
            'Iteration': 'Song #',
            'name': 'Song Name',
            'ArtistName': 'Artist Name',
            'Emotion':'Prioritized Emotion'
        }, 
        inplace=True
    )

    table_df = filtered_df[['Song #','Song Name','Artist Name','Prioritized Emotion']]

    return html.Div([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in table_df.columns],
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
                style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'},
                style_data={'width': '125px', 'minWidth': '125px', 'maxWidth': '125px','overflow': 'hidden','textOverflow': 'ellipsis'},
                sort_action='native',sort_mode="multi",
                page_action="native", page_current= 0,page_size= 14,                     
                data=table_df.to_dict('records')
            )
        ])


#----- Callback to disable slider
@app.callback(
    Output('position_slider', 'disabled'),
    Input('radio-button-toggle', 'value')
)
def toggle_slider_disable(selected_option):
    return selected_option in ['Spotify View', 'Playlist View']

#----- Callback to emotion dropdown
@app.callback(
    Output('dropdown10', 'disabled'),
    Input('radio-button-toggle', 'value')
)
def toggle_slider_disable(selected_option):
    return selected_option in ['Spotify View']


#------------- Info buttons --> opening and closing

@app.callback(
    Output("modal_info1", "is_open"),
    [Input("info1", "n_clicks"), 
    Input("close_info1", "n_clicks")],
    [State("modal_info1", "is_open")],
)

def toggle_modal_info1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal_info2", "is_open"),
    [Input("info2", "n_clicks"), 
    Input("close_info2", "n_clicks")],
    [State("modal_info2", "is_open")],
)

def toggle_modal_info2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



if __name__=='__main__':
	app.run_server()