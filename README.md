# Setlist Analysis

## Description

The purpose of this project was to:

- determine what an ideal order of songs would look like based on quantifying the emotions of lyrics
- practice pulling data from APIs
- practice fuzzy matching and fuzzy deduplication
- learn about emotion detection
- learn about Taylor Swift

## Data
The data used for this analysis was pulled using APIs from [setlist.fm](https://www.setlist.fm/) to gather all historical setlists for the artists I chose as well as [genius.com](https://genius.com/) to gather corresponding lyrics for each song.

### App
Click [here](https://setlist-analysis-jz-app.onrender.com/) to view the app.  It might need a few minutes to "wake up" if it has been idle for awhile.

## Challenges
The spelling and punctuation was not always consistent between the lyrics datasets and the setlists datasets.  Eg: one of my favorite songs, Zebraskin by Dredg was often written as Zebra skin or Zebra Skin.  Due to this issue, I had to use the fuzzy_merge function of the [fuzzy_pandas](https://github.com/jsoma/fuzzy_pandas) library to sufficiently join the data sets.

A similar issue popped up when ranking songs by emotions.  I had to use the dedupe_dataframe function of the [pandas_dedupe](https://pypi.org/project/pandas-dedupe/) library to perform fuzzy deduplication.
