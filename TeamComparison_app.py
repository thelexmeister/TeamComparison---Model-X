import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime

# Set the layout to wide
st.set_page_config(layout="wide")

# Load the data from an Excel file
df  = pd.read_excel('player_median_scores.xlsx')
df2 = pd.read_excel('PlayerClusters - 2024 FFBall.xlsx')

# Calculate the lower and upper bounds based on the probability
df['Lower Bound'] = df['Adjusted Median Score'] - df['Adjusted Median Score'] * ((0.5 - df['Probability']) / 2)
df['Upper Bound'] = df['Adjusted Median Score'] + df['Adjusted Median Score'] * ((0.5 - df['Probability']) / 2)

# Map the confidence score group to specific colors
color_map = {
    'High Confidence': 'red', 
    'Moderate Confidence': 'blue', 
    'Low Confidence': 'green'
}

# Add the color column based on the Score Group
df['Color'] = df['Score Group'].map(color_map)

# If there are any missing colors (in case a score group is undefined), fill them with 'grey'
df['Color'] = df['Color'].fillna('grey')

# Check if 'Score Group' exists, if not, add it with default values
if 'Score Group' not in df.columns:
    df['Score Group'] = 'Moderate Confidence'  # Default group if not present

# Function to calculate total score mean and standard deviation
def calculate_team_stats(players):
    total_mean = 0
    total_stddev = 0
    
    for player in players:
        player_data = df[df['Player'] == player].iloc[0]
        predicted_score = player_data['Adjusted Median Score']
        stddev = predicted_score * 0.2  # Assuming a 20% standard deviation as an example, adjust if needed
        
        total_mean += predicted_score
        total_stddev += stddev**2  # Variance is stddev squared

    total_stddev = np.sqrt(total_stddev)  # Get the combined standard deviation
    return total_mean, total_stddev

# Function to plot the player's predicted scores with probability ranges
def plot_player_scores(players, team_name=""):
    fig = go.Figure()

    for player in players:
        player_data = df[df['Player'] == player].iloc[0]
        
        # Round the predicted score to 1 decimal place
        predicted_score = round(player_data['Adjusted Median Score'], 1)
        color = player_data['Color']  # Use the color associated with the player's confidence level

        # Plot the predicted score (current week) with the confidence color
        fig.add_trace(go.Scatter(
            x=[player],
            y=[predicted_score],
            mode='markers',
            marker=dict(size=12, color=color),
            showlegend=False
        ))

        # Plot the probability range with the confidence color
        fig.add_trace(go.Scatter(
            x=[player, player],
            y=[player_data['Lower Bound'], player_data['Upper Bound']],
            mode='lines',
            line=dict(width=8, color=color),
            showlegend=False
        ))

        # Add text labels for the predicted score above the probability bars
        fig.add_trace(go.Scatter(
            x=[player],
            y=[player_data['Upper Bound'] + 1],
            mode='text',
            text=[f"{predicted_score}"],
            textposition="bottom center",
            showlegend=False
        ))

    # Update the layout to fix the y-axis range between 0 and 40
    fig.update_layout(
        title=f"{team_name} - Predicted Scores with Probability Ranges",
        xaxis_title="Player",
        yaxis_title="Predicted Score",
        showlegend=False,
        yaxis=dict(range=[0, 30])  # Set y-axis to go from 0 to 40
    )
    return fig

# Function to plot historical scores for each player (last 4 weeks)
def plot_historical_scores(players, df2):
    fig = go.Figure()

    # Color mapping for historical scores
    historical_color_map = {
        'Week 15': 'red',
        'Week 14': 'yellow',
        'Week 13': 'green',
        'Week 12': 'blue'
    }

    # Columns corresponding to the last four weeks (adjust as new weeks come in)
    week_columns = ['Week 16', 'Week 15', 'Week 14', 'Week 13']

    for player in players:
        # Fetch the historical scores for the player from df2
        player_data = df2[df2['Player'] == player]
        
        # If the player exists in the historical data
        if not player_data.empty:
            # For each week, plot the score with corresponding color
            for week, color in zip(week_columns, historical_color_map.values()):
                historical_score = player_data[week].values[0]  # Get the score for the week
                fig.add_trace(go.Scatter(
                    x=[player],
                    y=[historical_score],
                    mode='markers',
                    marker=dict(size=12, color=color),
                    name=f"{week} ({player})",
                    showlegend=False
                ))

    # Update layout for the historical scores figure
    fig.update_layout(
        title="Player Historical Scores (Last 4 Weeks)",
        xaxis_title="Player",
        yaxis_title="Score",
        showlegend=True,
        yaxis=dict(range=[0, 30])  # You can adjust this range based on your data
    )
    return fig

# Elo Probability Calculation Function
def calculate_elo_probability(team_score, opponent_score):
    # Elo rating system formula
    probability = 1 / (1 + 10 ** ((opponent_score - team_score) / 400))
    return probability

# Streamlit User Interface
st.title('Western Wolves: NFL Fantasy Team Prediction Dashboard')
st.text('''All you have to do is start typing the name of your player in each slot and then click on it. For the flex players, just start typing 
the first one, click on it. Click again in the space, then type your second name until you find the player you want to add.
Then click outside the box when you have 2 names in red in the boxes.''')
st.text(' ')
st.text('''These scores are from my most current model, which will be MUCH lower than the ESPN predicted values, as they are calculated using a different scoring
system than our league, and they are also very conservative, providing the most opportunity to be correct and underpredict.''')
st.text(' ')
st.write('### MOST IMPORTANTLY - HAVE FUN!!')

# Display last updated date if available

    st.write('Last updated: 12/26/2024 2:00pm')


# Create two columns for layout
col1, col2 = st.columns(2)

# Left Column - Your Team
with col1:
    st.header("Your Team")

    # Select positions for your team
    qb = st.selectbox("Select Quarterback", df[df['Position'] == 'QB']['Player'].tolist(), key="qb_select")
    rb1 = st.selectbox("Select Running Back 1", df[df['Position'] == 'RB']['Player'].tolist(), key="rb1_select")
    rb2 = st.selectbox("Select Running Back 2", df[df['Position'] == 'RB']['Player'].tolist(), key="rb2_select")
    wr1 = st.selectbox("Select Wide Receiver 1", df[df['Position'] == 'WR']['Player'].tolist(), key="wr1_select")
    wr2 = st.selectbox("Select Wide Receiver 2", df[df['Position'] == 'WR']['Player'].tolist(), key="wr2_select")
    te = st.selectbox("Select Tight End", df[df['Position'] == 'TE']['Player'].tolist(), key="te_select")
    flex = st.multiselect("Select Flex Players", df[(df['Position'] == 'RB') | (df['Position'] == 'WR') | (df['Position'] == 'TE')]['Player'].tolist(), max_selections=2, key="flex_select")

    # Combine selected players for your team
    selected_players = [qb, rb1, rb2, wr1, wr2, te] + flex

    # Plot your team's predicted scores with probability ranges
    fig = plot_player_scores(selected_players, team_name="Your Team")
    st.plotly_chart(fig, key="your_team_plot")

    # Calculate and display the total predicted score for your team
    total_score = sum(df[df['Player'] == player]['Adjusted Median Score'].iloc[0] for player in selected_players)
    st.write(f"Your Team's Total Predicted Score: {total_score:.1f}")


# Right Column - Opponent's Team
with col2:
    st.header("Opponent's Team")

    # Select positions for the opponent's team
    opponent_qb = st.selectbox("Select Opponent's Quarterback", df[df['Position'] == 'QB']['Player'].tolist())
    opponent_rb1 = st.selectbox("Select Opponent's Running Back 1", df[df['Position'] == 'RB']['Player'].tolist())
    opponent_rb2 = st.selectbox("Select Opponent's Running Back 2", df[df['Position'] == 'RB']['Player'].tolist())
    opponent_wr1 = st.selectbox("Select Opponent's Wide Receiver 1", df[df['Position'] == 'WR']['Player'].tolist())
    opponent_wr2 = st.selectbox("Select Opponent's Wide Receiver 2", df[df['Position'] == 'WR']['Player'].tolist())
    opponent_te = st.selectbox("Select Opponent's Tight End", df[df['Position'] == 'TE']['Player'].tolist())
    opponent_flex = st.multiselect("Select Opponent's Flex Players", df[(df['Position'] == 'RB') | (df['Position'] == 'WR') | (df['Position'] == 'TE')]['Player'].tolist(), max_selections=2)

    # Combine selected players for the opponent's team
    opponent_selected_players = [opponent_qb, opponent_rb1, opponent_rb2, opponent_wr1, opponent_wr2, opponent_te] + opponent_flex

    # Plot the opponent's predicted scores with probability ranges
    fig_opponent = plot_player_scores(opponent_selected_players, team_name="Opponent's Team")
    st.plotly_chart(fig_opponent, key="opponent_team_plot")  # Added unique key for this plot

    # Calculate and display the total predicted score for the opponent's team
    opponent_total_score = sum(df[df['Player'] == player]['Adjusted Median Score'].iloc[0] for player in opponent_selected_players)
    st.write(f"Total Predicted Score for Opponent's Team: {round(opponent_total_score, 1)}")


st.text(' ')
st.text('''In the figures above, RED means high confidence in the prediction and zone of probability, BLUE means moderate confidence,
        GREEN means low confidence.''')
st.text(' ')

# Display historical scores for each player chosen
st.write("### Historical Performance of Selected Players (Last 4 Weeks)")
fig_historical = plot_historical_scores(selected_players, df2)
st.plotly_chart(fig_historical, key="historical_scores_plot")

st.text(' ')
st.text('''In the figure above showing the historical scores for each of the last 4 weeks: 'Week 16': 'red', 'Week 15': 'yellow', 'Week 14': 'green', 'Week 13': 'blue'.''')
st.text(' ')

# Calculate the Elo probability of your team winning (this part remains unchanged)
elo_probability = calculate_elo_probability(total_score, opponent_total_score)


# Display a comparison table of the total predicted scores
st.write("### Total Predicted Score Comparison")
comparison_df = pd.DataFrame({
    "Team": ["Your Team", "Opponent's Team"],
    "Total Predicted Score": [round(total_score, 1), round(opponent_total_score, 1)],
    "Winning Probability (Your Team)": [f"{round(elo_probability * 100, 2)}%", "-"]
})

st.write(comparison_df)

st.write("The probability of your team winning is computed based on comparing the two teams' predicted scores using an adjusted Elo-like equation.")

# Streamlit User Interface for Optimal Roster
st.title('Western Wolves: NFL Fantasy Team Prediction Dashboard')
st.text('''Enter your team below and the app will automatically calculate the optimal roster based on the highest predicted scores. 
You will be able to input your players for each position (QB, RB, WR, TE) and get the best possible team selection!''')

# Create a section for the user to input players
st.header("Input Your Players")

# Collecting the players from the user
qb_pool = st.multiselect("Select Quarterbacks", df[df['Position'] == 'QB']['Player'].tolist())
rb_pool = st.multiselect("Select Running Backs", df[df['Position'] == 'RB']['Player'].tolist())
wr_pool = st.multiselect("Select Wide Receivers", df[df['Position'] == 'WR']['Player'].tolist())
te_pool = st.multiselect("Select Tight Ends", df[df['Position'] == 'TE']['Player'].tolist())

# Combining all the selected players in one list
selected_players = qb_pool + rb_pool + wr_pool + te_pool

# Function to get the initial optimal roster based on the entire dataset
def get_initial_optimal_roster(df):
    optimal_roster = {}
    
    # Find the best QB (1 QB)
    best_qb = max(df[df['Position'] == 'QB']['Player'], key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0])
    optimal_roster['QB'] = best_qb
    
    # Find the best 2 RBs
    best_rbs = sorted(df[df['Position'] == 'RB']['Player'], key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2]
    optimal_roster['RB1'], optimal_roster['RB2'] = best_rbs
    
    # Find the best 2 WRs
    best_wrs = sorted(df[df['Position'] == 'WR']['Player'], key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2]
    optimal_roster['WR1'], optimal_roster['WR2'] = best_wrs
    
    # Find the best TE (1 TE)
    best_te = max(df[df['Position'] == 'TE']['Player'], key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0])
    optimal_roster['TE'] = best_te
    
    # Combine the remaining players (RB, WR, TE) and pick the top 2 for Flex positions
    remaining_players = (set(df[df['Position'] == 'RB']['Player'].tolist() + df[df['Position'] == 'WR']['Player'].tolist() + df[df['Position'] == 'TE']['Player'].tolist())
                         - set(best_rbs) - set(best_wrs) - set([best_te]))
    
    best_additional_players = sorted(remaining_players, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2]
    optimal_roster['Flex1'], optimal_roster['Flex2'] = best_additional_players
    
    return optimal_roster

# Function to update optimal roster based on selected players
def get_updated_optimal_roster(df, selected_players):
    optimal_roster = {}
    
    # Extract selected positions
    qb_pool = [player for player in selected_players if player in df[df['Position'] == 'QB']['Player'].tolist()]
    rb_pool = [player for player in selected_players if player in df[df['Position'] == 'RB']['Player'].tolist()]
    wr_pool = [player for player in selected_players if player in df[df['Position'] == 'WR']['Player'].tolist()]
    te_pool = [player for player in selected_players if player in df[df['Position'] == 'TE']['Player'].tolist()]
    
    # Get the best QB, RBs, WRs, TE, and Flex based on the selected players
    best_qb = max(qb_pool, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0]) if qb_pool else get_initial_optimal_roster(df)['QB']
    
    best_rbs = sorted(rb_pool, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2] if rb_pool else [get_initial_optimal_roster(df)['RB1'], get_initial_optimal_roster(df)['RB2']]
    
    best_wrs = sorted(wr_pool, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2] if wr_pool else [get_initial_optimal_roster(df)['WR1'], get_initial_optimal_roster(df)['WR2']]
    
    best_te = max(te_pool, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0]) if te_pool else get_initial_optimal_roster(df)['TE']
    
    # Combine RB, WR, TE and pick the top 2 for Flex positions
    remaining_players = (set(rb_pool + wr_pool + te_pool) - set(best_rbs) - set(best_wrs) - set([best_te]))
    
    # Handle the case when no remaining players are available
    best_additional_players = sorted(remaining_players, key=lambda player: df[df['Player'] == player]['Adjusted Median Score'].iloc[0], reverse=True)[:2] if remaining_players else [get_initial_optimal_roster(df)['Flex1'], get_initial_optimal_roster(df)['Flex2']]
    
    optimal_roster = {
        'QB': best_qb,
        'RB1': best_rbs[0], 'RB2': best_rbs[1],
        'WR1': best_wrs[0], 'WR2': best_wrs[1],
        'TE': best_te,
        'Flex1': best_additional_players[0], 'Flex2': best_additional_players[1]
    }
    
    return optimal_roster



# Get the initial optimal roster
initial_optimal_roster = get_initial_optimal_roster(df)

# Get the updated optimal roster based on the user's selections
updated_optimal_roster = get_updated_optimal_roster(df, selected_players)

# Display the optimal roster
st.write("### Optimal Roster (Based on Your Selections)")
st.write(f"**QB:** {updated_optimal_roster['QB']}")
st.write(f"**RB1:** {updated_optimal_roster['RB1']}")
st.write(f"**RB2:** {updated_optimal_roster['RB2']}")
st.write(f"**WR1:** {updated_optimal_roster['WR1']}")
st.write(f"**WR2:** {updated_optimal_roster['WR2']}")
st.write(f"**TE:** {updated_optimal_roster['TE']}")
st.write(f"**Flex1:** {updated_optimal_roster['Flex1']}")
st.write(f"**Flex2:** {updated_optimal_roster['Flex2']}")

