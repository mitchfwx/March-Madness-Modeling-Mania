# Calculating Team Quality Rankings for NCAA March Madness Tournament Competition


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression

```

For this analysis, we are building the rankings off of the regular season results. 


```python
seeds = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')
regular_results_detailed = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MRegularSeasonDetailedResults.csv')
team_keys = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MTeams.csv')
seeds = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')
```


```python
def prepare_data_modified(df):
    
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'         
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]
    
    output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)

    output['T1_Possessions'] = output['T1_FGA'] - output['T1_OR'] + output['T1_TO'] + (0.475 * output['T1_FTA'])
    output['T2_Possessions'] = output['T2_FGA'] - output['T2_OR'] + output['T2_TO'] + (0.475 * output['T2_FTA'])

    output['T1_OER'] = output['T1_Score'] / ( output['T1_FGA'] +  ((output['T1_FTA'] * 0.9)/2 + output['T1_TO']))
    output['T1_DER'] = output['T2_Score'] / ( output['T2_FGA'] +  ((output['T2_FTA'] * 0.9)/2 + output['T2_TO']))

    output['T1_PAPP'] = output['T1_Score']/output['T1_Possessions']
    output['T2_PAPP'] = output['T2_Score']/output['T2_Possessions']

        
    return output
```


```python
regular_results_detailed = prepare_data_modified(regular_results_detailed)
```


```python
regular_results_detailed["GameNumber"] = regular_results_detailed.groupby(['Season', 'T1_TeamID'])['DayNum'].rank(method='dense', ascending=True)
```

### Splitting Data into Segments of the Season


It is common for teams to start out very strong but then flounder in the second half leading up to the NCAA Tournament. There are also a lot of teams that struggle in the early part of the season but then have a really strong finish to the regular season and are "hot" coming in to the tournament. Both teams on paper may be equal, but any smart person would put extra weighting towards the team that had a strong finish to the season than a strong start. 

To account for this, we will calculate rankings based off of the following periods: 
1. Full Season Rankings
2. First Half Rankings
3. Second Half Rankings


```python
FirstHalf = regular_results_detailed.groupby(['Season', 'T1_TeamID']).apply(lambda x: x.iloc[:x['T1_TeamID'].size//2]).reset_index(drop = True)
```


```python
SecondHalf = regular_results_detailed.groupby(['Season', 'T1_TeamID']).apply(lambda x: x.iloc[x['T1_TeamID'].size//2:]).reset_index(drop = True)
```


```python
# convert to str, so the model would treat TeamID them as factors
FirstHalf['T1_TeamID'] = FirstHalf['T1_TeamID'].astype(str)
FirstHalf['T2_TeamID'] = FirstHalf['T2_TeamID'].astype(str)

SecondHalf['T1_TeamID'] = SecondHalf['T1_TeamID'].astype(str)
SecondHalf['T2_TeamID'] = SecondHalf['T2_TeamID'].astype(str)

regular_results_detailed['T1_TeamID'] = regular_results_detailed['T1_TeamID'].astype(str)
regular_results_detailed['T2_TeamID'] = regular_results_detailed['T2_TeamID'].astype(str)

# make it a binary task
FirstHalf['win'] = np.where(FirstHalf['T1_Score'] > FirstHalf['T2_Score'], 1, 0)
SecondHalf['win'] = np.where(SecondHalf['T1_Score'] > SecondHalf['T2_Score'], 1, 0)
regular_results_detailed['win'] = np.where(regular_results_detailed['T1_Score'] > regular_results_detailed['T2_Score'], 1, 0)
```


```python
FirstHalf = FirstHalf[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]
SecondHalf = SecondHalf[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]
FullSeason = regular_results_detailed[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]
```

# Calculating Team Rankings

The process for calculating team quality rankings will be the same across the first half, second half, and full season. 

We will use a logistic regression approach and treat each team in our data set (for each season) as a dummy variable. Essentially what we are doing is comparing a given team to all of the opponents it faced during the season, and calculating the probability of winning against a given team. Then, by taking the coefficients of the logistic regression, we can see which teams are most positively associated with an increase in win probability. This coefficient is what will be used for the team quality ranking. 


```python
def team_quality(data, season):

    x_train = data[data.Season == season]
    x_train = pd.concat([pd.get_dummies(x_train.T1_TeamID, prefix='T1_TeamID'), pd.get_dummies(x_train.T2_TeamID, prefix='T2_TeamID')], axis=1)
    y_train = data[data.Season == season]['win']  
    
    logisticRegr = LogisticRegression(fit_intercept = True, 
                                      penalty = 'none', 
                                     max_iter = 5000)
    logisticRegr.fit(x_train, y_train)
    
    
    # extracting parameters from glm
    quality = pd.DataFrame(logisticRegr.coef_).transpose()    
    quality['TeamID'] = pd.DataFrame(x_train.columns)
    quality.columns = ['beta','TeamID']
    
    quality['Season'] = season
    # taking exp due to binomial model being used
    quality['quality'] = np.exp(quality['beta'])
    # only interested in glm parameters with T1_, as T2_ should be mirroring T1_ ones
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    quality = quality[['TeamID', 'beta', 'Season', 'quality']]
    return quality
```

It is important to note that quality rankings are calculated for each season individually. It does not make sense to compare rankings across seasons since with each season comes different players. For predicting results in a give tournament, we are only concerned with the rankings for that specific year. 

Since these are used as inputs for various modeling activities, I will go ahead and calculate the rankings for all seasons. 


```python
H1_team_quality = pd.concat([team_quality(FirstHalf, 2003),
                          team_quality(FirstHalf, 2004),
                          team_quality(FirstHalf, 2005),
                          team_quality(FirstHalf, 2006),
                          team_quality(FirstHalf, 2007),
                          team_quality(FirstHalf, 2008),
                          team_quality(FirstHalf, 2009),
                          team_quality(FirstHalf, 2010),
                          team_quality(FirstHalf, 2011),
                          team_quality(FirstHalf, 2012),
                          team_quality(FirstHalf, 2013),
                          team_quality(FirstHalf, 2014),
                          team_quality(FirstHalf, 2015),
                          team_quality(FirstHalf, 2016),
                          team_quality(FirstHalf, 2017),
                          team_quality(FirstHalf, 2018),
                          team_quality(FirstHalf, 2019),
                             team_quality(FirstHalf, 2020),
                            team_quality(FirstHalf, 2021),
                            team_quality(FirstHalf, 2022),
                            team_quality(FirstHalf, 2023)]).reset_index(drop=True)

H2_team_quality = pd.concat([team_quality(SecondHalf, 2003),
                          team_quality(SecondHalf, 2004),
                          team_quality(SecondHalf, 2005),
                          team_quality(SecondHalf, 2006),
                          team_quality(SecondHalf, 2007),
                          team_quality(SecondHalf, 2008),
                          team_quality(SecondHalf, 2009),
                          team_quality(SecondHalf, 2010),
                          team_quality(SecondHalf, 2011),
                          team_quality(SecondHalf, 2012),
                          team_quality(SecondHalf, 2013),
                          team_quality(SecondHalf, 2014),
                          team_quality(SecondHalf, 2015),
                          team_quality(SecondHalf, 2016),
                          team_quality(SecondHalf, 2017),
                          team_quality(SecondHalf, 2018),
                          team_quality(SecondHalf, 2019), 
                             team_quality(SecondHalf, 2020),
                            team_quality(SecondHalf, 2021), 
                            team_quality(SecondHalf, 2022), 
                            team_quality(SecondHalf, 2023)]).reset_index(drop=True)

Full_team_quality = pd.concat([team_quality(FullSeason, 2003),
                          team_quality(FullSeason, 2004),
                          team_quality(FullSeason, 2005),
                          team_quality(FullSeason, 2006),
                          team_quality(FullSeason, 2007),
                          team_quality(FullSeason, 2008),
                          team_quality(FullSeason, 2009),
                          team_quality(FullSeason, 2010),
                          team_quality(FullSeason, 2011),
                          team_quality(FullSeason, 2012),
                          team_quality(FullSeason, 2013),
                          team_quality(FullSeason, 2014),
                          team_quality(FullSeason, 2015),
                          team_quality(FullSeason, 2016),
                          team_quality(FullSeason, 2017),
                          team_quality(FullSeason, 2018),
                          team_quality(FullSeason, 2019), 
                               team_quality(FullSeason, 2020),
                              team_quality(FullSeason, 2021), 
                              team_quality(FullSeason, 2022), 
                              team_quality(FullSeason, 2023)]).reset_index(drop=True)
```


```python
H1_team_quality.to_csv('H1_team_quality.csv',index=False)
H2_team_quality.to_csv('H2_team_quality.csv',index=False)
Full_team_quality.to_csv('Full_team_quality.csv',index=False)
```


```python

```

# Top Teams from 2022 Season


```python
top10 = Full_team_quality.sort_values("quality", ascending = False)[Full_team_quality.Season == 2022].head(10)
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.



```python
top10 = pd.merge(top10, team_keys, how = "inner", left_on = ["TeamID"], right_on = ["TeamID"]).drop(columns = ['FirstD1Season', 'LastD1Season'])
top10 = pd.merge(top10, seeds, how = "inner", left_on = ["TeamID", "Season"], right_on = ["TeamID", "Season"])
```


```python
top10
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TeamID</th>
      <th>beta</th>
      <th>Season</th>
      <th>quality</th>
      <th>TeamName</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1112</td>
      <td>4.613485</td>
      <td>2022</td>
      <td>100.834979</td>
      <td>Arizona</td>
      <td>Z01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1211</td>
      <td>4.359189</td>
      <td>2022</td>
      <td>78.193720</td>
      <td>Gonzaga</td>
      <td>X01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1242</td>
      <td>4.275820</td>
      <td>2022</td>
      <td>71.939098</td>
      <td>Kansas</td>
      <td>Y01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1124</td>
      <td>4.182860</td>
      <td>2022</td>
      <td>65.553084</td>
      <td>Baylor</td>
      <td>W01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1437</td>
      <td>4.135864</td>
      <td>2022</td>
      <td>62.543608</td>
      <td>Villanova</td>
      <td>Z02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1344</td>
      <td>4.123159</td>
      <td>2022</td>
      <td>61.754001</td>
      <td>Providence</td>
      <td>Y04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1397</td>
      <td>3.980785</td>
      <td>2022</td>
      <td>53.559048</td>
      <td>Tennessee</td>
      <td>Z03</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1120</td>
      <td>3.922730</td>
      <td>2022</td>
      <td>50.538232</td>
      <td>Auburn</td>
      <td>Y02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1403</td>
      <td>3.817341</td>
      <td>2022</td>
      <td>45.483127</td>
      <td>Texas Tech</td>
      <td>X03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1345</td>
      <td>3.742305</td>
      <td>2022</td>
      <td>42.195121</td>
      <td>Purdue</td>
      <td>W03</td>
    </tr>
  </tbody>
</table>
</div>



Using this technique, we can calculate our own version of a teams sead for a given season. The seading process can often be manipulated and favor the popular teams as opposed to those that are truly the best. Every year there are upsets and head scratchers how a 15 seed overpowers a 2 seed. Every year it happens, could it be that the 2 seed was not worthy of a 2 seed? Possible. 

This can be seen very clearly with which teams it has put in the top 10. All the 1 seeds are ranked 1-4 per team quality rankings, this is expecte. After this is where it gets interesting. We see teams like Providence as the 6th best team per our quality rankings, but they're a 4 seed in the tournament. 

These quality rankings don't replace the normal seeds, but rather compliment them. They help adjust the seeds for the quality of their opponents, since that is what the model is doing: calculating the quality of each team relative to its opponents. Strictly accepting the tournament seeds puts you at the will of the analysts and selection comittees that ranked the teams. Using the team quality rankings, we now have a way to calculate the teams quality relative of its opponents in a methodical and explainable format. 
