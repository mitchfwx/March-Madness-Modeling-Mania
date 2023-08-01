{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "af254291",
   "metadata": {
    "papermill": {
     "duration": 0.025165,
     "end_time": "2023-08-01T20:28:42.527639",
     "exception": false,
     "start_time": "2023-08-01T20:28:42.502474",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Calculating Team Quality Rankings for NCAA March Madness Tournament Competition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a29cb6eb",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:42.586810Z",
     "iopub.status.busy": "2023-08-01T20:28:42.586053Z",
     "iopub.status.idle": "2023-08-01T20:28:44.264004Z",
     "shell.execute_reply": "2023-08-01T20:28:44.263297Z",
     "shell.execute_reply.started": "2023-08-01T19:21:07.863556Z"
    },
    "papermill": {
     "duration": 1.71299,
     "end_time": "2023-08-01T20:28:44.264179",
     "exception": false,
     "start_time": "2023-08-01T20:28:42.551189",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import statsmodels.api as sm\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import sklearn\n",
    "from sklearn.linear_model import LogisticRegression\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99124643",
   "metadata": {
    "papermill": {
     "duration": 0.023648,
     "end_time": "2023-08-01T20:28:44.312142",
     "exception": false,
     "start_time": "2023-08-01T20:28:44.288494",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "For this analysis, we are building the rankings off of the regular season results. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4ce02fe1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:44.365497Z",
     "iopub.status.busy": "2023-08-01T20:28:44.364791Z",
     "iopub.status.idle": "2023-08-01T20:28:44.841723Z",
     "shell.execute_reply": "2023-08-01T20:28:44.841180Z",
     "shell.execute_reply.started": "2023-08-01T20:02:51.238023Z"
    },
    "papermill": {
     "duration": 0.505845,
     "end_time": "2023-08-01T20:28:44.841866",
     "exception": false,
     "start_time": "2023-08-01T20:28:44.336021",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "seeds = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')\n",
    "regular_results_detailed = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MRegularSeasonDetailedResults.csv')\n",
    "team_keys = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MTeams.csv')\n",
    "seeds = pd.read_csv(r'/kaggle/input/march-machine-learning-mania-2023/MNCAATourneySeeds.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "352f4bed",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:44.893238Z",
     "iopub.status.busy": "2023-08-01T20:28:44.892600Z",
     "iopub.status.idle": "2023-08-01T20:28:44.905378Z",
     "shell.execute_reply": "2023-08-01T20:28:44.905934Z",
     "shell.execute_reply.started": "2023-08-01T19:21:16.882852Z"
    },
    "papermill": {
     "duration": 0.039599,
     "end_time": "2023-08-01T20:28:44.906080",
     "exception": false,
     "start_time": "2023-08-01T20:28:44.866481",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def prepare_data_modified(df):\n",
    "    \n",
    "    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', \n",
    "       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',\n",
    "       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',\n",
    "       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]\n",
    "\n",
    "    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'\n",
    "    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'\n",
    "    df.columns.values[6] = 'location'\n",
    "    dfswap.columns.values[6] = 'location'         \n",
    "    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]\n",
    "    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]\n",
    "    \n",
    "    output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)\n",
    "\n",
    "    output['T1_Possessions'] = output['T1_FGA'] - output['T1_OR'] + output['T1_TO'] + (0.475 * output['T1_FTA'])\n",
    "    output['T2_Possessions'] = output['T2_FGA'] - output['T2_OR'] + output['T2_TO'] + (0.475 * output['T2_FTA'])\n",
    "\n",
    "    output['T1_OER'] = output['T1_Score'] / ( output['T1_FGA'] +  ((output['T1_FTA'] * 0.9)/2 + output['T1_TO']))\n",
    "    output['T1_DER'] = output['T2_Score'] / ( output['T2_FGA'] +  ((output['T2_FTA'] * 0.9)/2 + output['T2_TO']))\n",
    "\n",
    "    output['T1_PAPP'] = output['T1_Score']/output['T1_Possessions']\n",
    "    output['T2_PAPP'] = output['T2_Score']/output['T2_Possessions']\n",
    "\n",
    "        \n",
    "    return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "41ff5fa5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:44.956476Z",
     "iopub.status.busy": "2023-08-01T20:28:44.955870Z",
     "iopub.status.idle": "2023-08-01T20:28:45.219962Z",
     "shell.execute_reply": "2023-08-01T20:28:45.220485Z",
     "shell.execute_reply.started": "2023-08-01T19:21:23.237314Z"
    },
    "papermill": {
     "duration": 0.290798,
     "end_time": "2023-08-01T20:28:45.220650",
     "exception": false,
     "start_time": "2023-08-01T20:28:44.929852",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "regular_results_detailed = prepare_data_modified(regular_results_detailed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "08920625",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:45.272390Z",
     "iopub.status.busy": "2023-08-01T20:28:45.271469Z",
     "iopub.status.idle": "2023-08-01T20:28:45.310056Z",
     "shell.execute_reply": "2023-08-01T20:28:45.310624Z",
     "shell.execute_reply.started": "2023-08-01T19:21:29.248268Z"
    },
    "papermill": {
     "duration": 0.065556,
     "end_time": "2023-08-01T20:28:45.310815",
     "exception": false,
     "start_time": "2023-08-01T20:28:45.245259",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "regular_results_detailed[\"GameNumber\"] = regular_results_detailed.groupby(['Season', 'T1_TeamID'])['DayNum'].rank(method='dense', ascending=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "505a6e5a",
   "metadata": {
    "papermill": {
     "duration": 0.023164,
     "end_time": "2023-08-01T20:28:45.358556",
     "exception": false,
     "start_time": "2023-08-01T20:28:45.335392",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Splitting Data into Segments of the Season"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe45d563",
   "metadata": {
    "papermill": {
     "duration": 0.023684,
     "end_time": "2023-08-01T20:28:45.405534",
     "exception": false,
     "start_time": "2023-08-01T20:28:45.381850",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "\n",
    "It is common for teams to start out very strong but then flounder in the second half leading up to the NCAA Tournament. There are also a lot of teams that struggle in the early part of the season but then have a really strong finish to the regular season and are \"hot\" coming in to the tournament. Both teams on paper may be equal, but any smart person would put extra weighting towards the team that had a strong finish to the season than a strong start. \n",
    "\n",
    "To account for this, we will calculate rankings based off of the following periods: \n",
    "1. Full Season Rankings\n",
    "2. First Half Rankings\n",
    "3. Second Half Rankings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "36e2f116",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:45.455767Z",
     "iopub.status.busy": "2023-08-01T20:28:45.454834Z",
     "iopub.status.idle": "2023-08-01T20:28:50.437276Z",
     "shell.execute_reply": "2023-08-01T20:28:50.437759Z",
     "shell.execute_reply.started": "2023-08-01T19:24:10.963891Z"
    },
    "papermill": {
     "duration": 5.008973,
     "end_time": "2023-08-01T20:28:50.437954",
     "exception": false,
     "start_time": "2023-08-01T20:28:45.428981",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "FirstHalf = regular_results_detailed.groupby(['Season', 'T1_TeamID']).apply(lambda x: x.iloc[:x['T1_TeamID'].size//2]).reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ea479b2b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:50.491402Z",
     "iopub.status.busy": "2023-08-01T20:28:50.490521Z",
     "iopub.status.idle": "2023-08-01T20:28:55.050633Z",
     "shell.execute_reply": "2023-08-01T20:28:55.051207Z",
     "shell.execute_reply.started": "2023-08-01T19:24:16.739837Z"
    },
    "papermill": {
     "duration": 4.586844,
     "end_time": "2023-08-01T20:28:55.051393",
     "exception": false,
     "start_time": "2023-08-01T20:28:50.464549",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "SecondHalf = regular_results_detailed.groupby(['Season', 'T1_TeamID']).apply(lambda x: x.iloc[x['T1_TeamID'].size//2:]).reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "aef3e25f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:55.101861Z",
     "iopub.status.busy": "2023-08-01T20:28:55.101248Z",
     "iopub.status.idle": "2023-08-01T20:28:56.037011Z",
     "shell.execute_reply": "2023-08-01T20:28:56.037502Z",
     "shell.execute_reply.started": "2023-08-01T19:24:56.646322Z"
    },
    "papermill": {
     "duration": 0.962112,
     "end_time": "2023-08-01T20:28:56.037682",
     "exception": false,
     "start_time": "2023-08-01T20:28:55.075570",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# convert to str, so the model would treat TeamID them as factors\n",
    "FirstHalf['T1_TeamID'] = FirstHalf['T1_TeamID'].astype(str)\n",
    "FirstHalf['T2_TeamID'] = FirstHalf['T2_TeamID'].astype(str)\n",
    "\n",
    "SecondHalf['T1_TeamID'] = SecondHalf['T1_TeamID'].astype(str)\n",
    "SecondHalf['T2_TeamID'] = SecondHalf['T2_TeamID'].astype(str)\n",
    "\n",
    "regular_results_detailed['T1_TeamID'] = regular_results_detailed['T1_TeamID'].astype(str)\n",
    "regular_results_detailed['T2_TeamID'] = regular_results_detailed['T2_TeamID'].astype(str)\n",
    "\n",
    "# make it a binary task\n",
    "FirstHalf['win'] = np.where(FirstHalf['T1_Score'] > FirstHalf['T2_Score'], 1, 0)\n",
    "SecondHalf['win'] = np.where(SecondHalf['T1_Score'] > SecondHalf['T2_Score'], 1, 0)\n",
    "regular_results_detailed['win'] = np.where(regular_results_detailed['T1_Score'] > regular_results_detailed['T2_Score'], 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9b9ecde9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:56.088858Z",
     "iopub.status.busy": "2023-08-01T20:28:56.088251Z",
     "iopub.status.idle": "2023-08-01T20:28:56.197194Z",
     "shell.execute_reply": "2023-08-01T20:28:56.196581Z",
     "shell.execute_reply.started": "2023-08-01T19:25:10.783675Z"
    },
    "papermill": {
     "duration": 0.13541,
     "end_time": "2023-08-01T20:28:56.197343",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.061933",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "FirstHalf = FirstHalf[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]\n",
    "SecondHalf = SecondHalf[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]\n",
    "FullSeason = regular_results_detailed[['Season', 'T1_TeamID', 'T2_TeamID', 'win']]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8fe10ab",
   "metadata": {
    "papermill": {
     "duration": 0.023139,
     "end_time": "2023-08-01T20:28:56.244352",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.221213",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Calculating Team Rankings"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e6c2754",
   "metadata": {
    "papermill": {
     "duration": 0.023642,
     "end_time": "2023-08-01T20:28:56.291400",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.267758",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The process for calculating team quality rankings will be the same across the first half, second half, and full season. \n",
    "\n",
    "We will use a logistic regression approach and treat each team in our data set (for each season) as a dummy variable. Essentially what we are doing is comparing a given team to all of the opponents it faced during the season, and calculating the probability of winning against a given team. Then, by taking the coefficients of the logistic regression, we can see which teams are most positively associated with an increase in win probability. This coefficient is what will be used for the team quality ranking. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6651834a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:56.348019Z",
     "iopub.status.busy": "2023-08-01T20:28:56.347090Z",
     "iopub.status.idle": "2023-08-01T20:28:56.349698Z",
     "shell.execute_reply": "2023-08-01T20:28:56.349156Z",
     "shell.execute_reply.started": "2023-08-01T19:29:13.336083Z"
    },
    "papermill": {
     "duration": 0.035402,
     "end_time": "2023-08-01T20:28:56.349822",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.314420",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def team_quality(data, season):\n",
    "\n",
    "    x_train = data[data.Season == season]\n",
    "    x_train = pd.concat([pd.get_dummies(x_train.T1_TeamID, prefix='T1_TeamID'), pd.get_dummies(x_train.T2_TeamID, prefix='T2_TeamID')], axis=1)\n",
    "    y_train = data[data.Season == season]['win']  \n",
    "    \n",
    "    logisticRegr = LogisticRegression(fit_intercept = True, \n",
    "                                      penalty = 'none', \n",
    "                                     max_iter = 5000)\n",
    "    logisticRegr.fit(x_train, y_train)\n",
    "    \n",
    "    \n",
    "    # extracting parameters from glm\n",
    "    quality = pd.DataFrame(logisticRegr.coef_).transpose()    \n",
    "    quality['TeamID'] = pd.DataFrame(x_train.columns)\n",
    "    quality.columns = ['beta','TeamID']\n",
    "    \n",
    "    quality['Season'] = season\n",
    "    # taking exp due to binomial model being used\n",
    "    quality['quality'] = np.exp(quality['beta'])\n",
    "    # only interested in glm parameters with T1_, as T2_ should be mirroring T1_ ones\n",
    "    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)\n",
    "    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)\n",
    "    quality = quality[['TeamID', 'beta', 'Season', 'quality']]\n",
    "    return quality"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "748b6420",
   "metadata": {
    "papermill": {
     "duration": 0.02269,
     "end_time": "2023-08-01T20:28:56.395652",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.372962",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "It is important to note that quality rankings are calculated for each season individually. It does not make sense to compare rankings across seasons since with each season comes different players. For predicting results in a give tournament, we are only concerned with the rankings for that specific year. \n",
    "\n",
    "Since these are used as inputs for various modeling activities, I will go ahead and calculate the rankings for all seasons. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7617a251",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:28:56.452365Z",
     "iopub.status.busy": "2023-08-01T20:28:56.451462Z",
     "iopub.status.idle": "2023-08-01T20:29:43.261953Z",
     "shell.execute_reply": "2023-08-01T20:29:43.263008Z",
     "shell.execute_reply.started": "2023-08-01T19:30:33.051397Z"
    },
    "papermill": {
     "duration": 46.843329,
     "end_time": "2023-08-01T20:29:43.263328",
     "exception": false,
     "start_time": "2023-08-01T20:28:56.419999",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "H1_team_quality = pd.concat([team_quality(FirstHalf, 2003),\n",
    "                          team_quality(FirstHalf, 2004),\n",
    "                          team_quality(FirstHalf, 2005),\n",
    "                          team_quality(FirstHalf, 2006),\n",
    "                          team_quality(FirstHalf, 2007),\n",
    "                          team_quality(FirstHalf, 2008),\n",
    "                          team_quality(FirstHalf, 2009),\n",
    "                          team_quality(FirstHalf, 2010),\n",
    "                          team_quality(FirstHalf, 2011),\n",
    "                          team_quality(FirstHalf, 2012),\n",
    "                          team_quality(FirstHalf, 2013),\n",
    "                          team_quality(FirstHalf, 2014),\n",
    "                          team_quality(FirstHalf, 2015),\n",
    "                          team_quality(FirstHalf, 2016),\n",
    "                          team_quality(FirstHalf, 2017),\n",
    "                          team_quality(FirstHalf, 2018),\n",
    "                          team_quality(FirstHalf, 2019),\n",
    "                             team_quality(FirstHalf, 2020),\n",
    "                            team_quality(FirstHalf, 2021),\n",
    "                            team_quality(FirstHalf, 2022),\n",
    "                            team_quality(FirstHalf, 2023)]).reset_index(drop=True)\n",
    "\n",
    "H2_team_quality = pd.concat([team_quality(SecondHalf, 2003),\n",
    "                          team_quality(SecondHalf, 2004),\n",
    "                          team_quality(SecondHalf, 2005),\n",
    "                          team_quality(SecondHalf, 2006),\n",
    "                          team_quality(SecondHalf, 2007),\n",
    "                          team_quality(SecondHalf, 2008),\n",
    "                          team_quality(SecondHalf, 2009),\n",
    "                          team_quality(SecondHalf, 2010),\n",
    "                          team_quality(SecondHalf, 2011),\n",
    "                          team_quality(SecondHalf, 2012),\n",
    "                          team_quality(SecondHalf, 2013),\n",
    "                          team_quality(SecondHalf, 2014),\n",
    "                          team_quality(SecondHalf, 2015),\n",
    "                          team_quality(SecondHalf, 2016),\n",
    "                          team_quality(SecondHalf, 2017),\n",
    "                          team_quality(SecondHalf, 2018),\n",
    "                          team_quality(SecondHalf, 2019), \n",
    "                             team_quality(SecondHalf, 2020),\n",
    "                            team_quality(SecondHalf, 2021), \n",
    "                            team_quality(SecondHalf, 2022), \n",
    "                            team_quality(SecondHalf, 2023)]).reset_index(drop=True)\n",
    "\n",
    "Full_team_quality = pd.concat([team_quality(FullSeason, 2003),\n",
    "                          team_quality(FullSeason, 2004),\n",
    "                          team_quality(FullSeason, 2005),\n",
    "                          team_quality(FullSeason, 2006),\n",
    "                          team_quality(FullSeason, 2007),\n",
    "                          team_quality(FullSeason, 2008),\n",
    "                          team_quality(FullSeason, 2009),\n",
    "                          team_quality(FullSeason, 2010),\n",
    "                          team_quality(FullSeason, 2011),\n",
    "                          team_quality(FullSeason, 2012),\n",
    "                          team_quality(FullSeason, 2013),\n",
    "                          team_quality(FullSeason, 2014),\n",
    "                          team_quality(FullSeason, 2015),\n",
    "                          team_quality(FullSeason, 2016),\n",
    "                          team_quality(FullSeason, 2017),\n",
    "                          team_quality(FullSeason, 2018),\n",
    "                          team_quality(FullSeason, 2019), \n",
    "                               team_quality(FullSeason, 2020),\n",
    "                              team_quality(FullSeason, 2021), \n",
    "                              team_quality(FullSeason, 2022), \n",
    "                              team_quality(FullSeason, 2023)]).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "605fd060",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:29:43.361374Z",
     "iopub.status.busy": "2023-08-01T20:29:43.360760Z",
     "iopub.status.idle": "2023-08-01T20:29:43.477836Z",
     "shell.execute_reply": "2023-08-01T20:29:43.477255Z",
     "shell.execute_reply.started": "2023-08-01T19:46:31.476870Z"
    },
    "papermill": {
     "duration": 0.165146,
     "end_time": "2023-08-01T20:29:43.477990",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.312844",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "H1_team_quality.to_csv('H1_team_quality.csv',index=False)\n",
    "H2_team_quality.to_csv('H2_team_quality.csv',index=False)\n",
    "Full_team_quality.to_csv('Full_team_quality.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb8bed29",
   "metadata": {
    "papermill": {
     "duration": 0.024154,
     "end_time": "2023-08-01T20:29:43.526869",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.502715",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "1041859b",
   "metadata": {
    "papermill": {
     "duration": 0.02287,
     "end_time": "2023-08-01T20:29:43.573326",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.550456",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Top Teams from 2022 Season"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "79b44fc6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:29:43.625876Z",
     "iopub.status.busy": "2023-08-01T20:29:43.625211Z",
     "iopub.status.idle": "2023-08-01T20:29:43.633665Z",
     "shell.execute_reply": "2023-08-01T20:29:43.633168Z",
     "shell.execute_reply.started": "2023-08-01T20:08:11.386442Z"
    },
    "papermill": {
     "duration": 0.036864,
     "end_time": "2023-08-01T20:29:43.633817",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.596953",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "top10 = Full_team_quality.sort_values(\"quality\", ascending = False)[Full_team_quality.Season == 2022].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6002ea9d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:29:43.685788Z",
     "iopub.status.busy": "2023-08-01T20:29:43.684866Z",
     "iopub.status.idle": "2023-08-01T20:29:43.697461Z",
     "shell.execute_reply": "2023-08-01T20:29:43.697982Z",
     "shell.execute_reply.started": "2023-08-01T20:08:13.477622Z"
    },
    "papermill": {
     "duration": 0.040158,
     "end_time": "2023-08-01T20:29:43.698135",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.657977",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "top10 = pd.merge(top10, team_keys, how = \"inner\", left_on = [\"TeamID\"], right_on = [\"TeamID\"]).drop(columns = ['FirstD1Season', 'LastD1Season'])\n",
    "top10 = pd.merge(top10, seeds, how = \"inner\", left_on = [\"TeamID\", \"Season\"], right_on = [\"TeamID\", \"Season\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1700fc34",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-01T20:29:43.748781Z",
     "iopub.status.busy": "2023-08-01T20:29:43.747952Z",
     "iopub.status.idle": "2023-08-01T20:29:43.765741Z",
     "shell.execute_reply": "2023-08-01T20:29:43.766343Z",
     "shell.execute_reply.started": "2023-08-01T20:08:16.083462Z"
    },
    "papermill": {
     "duration": 0.044644,
     "end_time": "2023-08-01T20:29:43.766488",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.721844",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>TeamID</th>\n",
       "      <th>beta</th>\n",
       "      <th>Season</th>\n",
       "      <th>quality</th>\n",
       "      <th>TeamName</th>\n",
       "      <th>Seed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1112</td>\n",
       "      <td>4.613485</td>\n",
       "      <td>2022</td>\n",
       "      <td>100.834979</td>\n",
       "      <td>Arizona</td>\n",
       "      <td>Z01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1211</td>\n",
       "      <td>4.359189</td>\n",
       "      <td>2022</td>\n",
       "      <td>78.193720</td>\n",
       "      <td>Gonzaga</td>\n",
       "      <td>X01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1242</td>\n",
       "      <td>4.275820</td>\n",
       "      <td>2022</td>\n",
       "      <td>71.939098</td>\n",
       "      <td>Kansas</td>\n",
       "      <td>Y01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1124</td>\n",
       "      <td>4.182860</td>\n",
       "      <td>2022</td>\n",
       "      <td>65.553084</td>\n",
       "      <td>Baylor</td>\n",
       "      <td>W01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1437</td>\n",
       "      <td>4.135864</td>\n",
       "      <td>2022</td>\n",
       "      <td>62.543608</td>\n",
       "      <td>Villanova</td>\n",
       "      <td>Z02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1344</td>\n",
       "      <td>4.123159</td>\n",
       "      <td>2022</td>\n",
       "      <td>61.754001</td>\n",
       "      <td>Providence</td>\n",
       "      <td>Y04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1397</td>\n",
       "      <td>3.980785</td>\n",
       "      <td>2022</td>\n",
       "      <td>53.559048</td>\n",
       "      <td>Tennessee</td>\n",
       "      <td>Z03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1120</td>\n",
       "      <td>3.922730</td>\n",
       "      <td>2022</td>\n",
       "      <td>50.538232</td>\n",
       "      <td>Auburn</td>\n",
       "      <td>Y02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1403</td>\n",
       "      <td>3.817341</td>\n",
       "      <td>2022</td>\n",
       "      <td>45.483127</td>\n",
       "      <td>Texas Tech</td>\n",
       "      <td>X03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1345</td>\n",
       "      <td>3.742305</td>\n",
       "      <td>2022</td>\n",
       "      <td>42.195121</td>\n",
       "      <td>Purdue</td>\n",
       "      <td>W03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   TeamID      beta  Season     quality    TeamName Seed\n",
       "0    1112  4.613485    2022  100.834979     Arizona  Z01\n",
       "1    1211  4.359189    2022   78.193720     Gonzaga  X01\n",
       "2    1242  4.275820    2022   71.939098      Kansas  Y01\n",
       "3    1124  4.182860    2022   65.553084      Baylor  W01\n",
       "4    1437  4.135864    2022   62.543608   Villanova  Z02\n",
       "5    1344  4.123159    2022   61.754001  Providence  Y04\n",
       "6    1397  3.980785    2022   53.559048   Tennessee  Z03\n",
       "7    1120  3.922730    2022   50.538232      Auburn  Y02\n",
       "8    1403  3.817341    2022   45.483127  Texas Tech  X03\n",
       "9    1345  3.742305    2022   42.195121      Purdue  W03"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "top10"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96c96550",
   "metadata": {
    "papermill": {
     "duration": 0.024201,
     "end_time": "2023-08-01T20:29:43.814798",
     "exception": false,
     "start_time": "2023-08-01T20:29:43.790597",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Using this technique, we can calculate our own version of a teams sead for a given season. The seading process can often be manipulated and favor the popular teams as opposed to those that are truly the best. Every year there are upsets and head scratchers how a 15 seed overpowers a 2 seed. Every year it happens, could it be that the 2 seed was not worthy of a 2 seed? Possible. \n",
    "\n",
    "This can be seen very clearly with which teams it has put in the top 10. All the 1 seeds are ranked 1-4 per team quality rankings, this is expecte. After this is where it gets interesting. We see teams like Providence as the 6th best team per our quality rankings, but they're a 4 seed in the tournament. \n",
    "\n",
    "These quality rankings don't replace the normal seeds, but rather compliment them. They help adjust the seeds for the quality of their opponents, since that is what the model is doing: calculating the quality of each team relative to its opponents. Strictly accepting the tournament seeds puts you at the will of the analysts and selection comittees that ranked the teams. Using the team quality rankings, we now have a way to calculate the teams quality relative of its opponents in a methodical and explainable format. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 71.583427,
   "end_time": "2023-08-01T20:29:44.449178",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-08-01T20:28:32.865751",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
