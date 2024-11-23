import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

# Set the layout to wide
st.set_page_config(layout="wide")

# Load the data from an Excel file
df = pd.read_excel('player_median_scores.xlsx')

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

# Function to calculate probability based on normal distributions
def calculate_probability(team_mean, team_stddev, opponent_mean, opponent_stddev):
    # Calculate the difference in scores
    diff_mean = team_mean - opponent_mean
    diff_stddev = np.sqrt(team_stddev**2 + opponent_stddev**2)
    
    # Calculate the probability that team A's score is higher than team B's score
    prob_team_wins = stats.norm.cdf(0, loc=diff_mean, scale=diff_stddev)
    return prob_team_wins

# Function to plot the player's predicted scores with probability ranges
def plot_player_scores(players, team_name=""):
    fig = go.Figure()

    for player in players:
        player_data = df[df['Player'] == player].iloc[0]
        
        # Round the predicted score to 1 decimal place
        predicted_score = round(player_data['Adjusted Median Score'], 1)
        color = player_data['Color']  # Use the color associated with the player's confidence level

        # Plot the dot (representing the score) with the confidence color
        fig.add_trace(go.Scatter(
            x=[player_data['Player']],
            y=[predicted_score],
            mode='markers',  # Only plot the marker (no text here)
            marker=dict(size=12, color=color),  # Use confidence color for the dot
            showlegend=False
        ))

        # Plot the probability range with the confidence color
        fig.add_trace(go.Scatter(
            x=[player_data['Player'], player_data['Player']],
            y=[player_data['Lower Bound'], player_data['Upper Bound']],
            mode='lines',
            line=dict(width=8, color=color),  # Widen the probability bars and color them
            showlegend=False
        ))

        # Add text labels for the predicted score above the probability bars
        fig.add_trace(go.Scatter(
            x=[player_data['Player']],
            y=[player_data['Upper Bound'] + 1],  # Place the score text slightly above the upper bound
            mode='text',
            text=[f"{predicted_score}"],
            textposition="bottom center",  # Position the score text above the bar
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

# Streamlit User Interface
st.title('Western Wolves: NFL Fantasy Team Prediction Dashboard')
st.text('''All you have to do is start typing the name of your player in each slot and then click on it.
For the flex players, just start typing the first one, click on it. Click again in the space, 
then type your second name until you find the player you want to add.
Then click outside the box when you have 2 names in red in the boxes.''')
st.text(' ')
st.text(' ')
st.text('''In the figure below, RED means high confidence in the prediction and zone of probability, BLUE means moderate confidence,
        GREEN means low confidence.''')

# Create two columns for layout
col1, col2 = st.columns(2)

# Left Column - Your Team
with col1:
    st.header("Your Team")

    # Select positions for your team
    qb = st.selectbox("Select Quarterback", df[df['Position'] == 'QB']['Player'].tolist())
    rb1 = st.selectbox("Select Running Back 1", df[df['Position'] == 'RB']['Player'].tolist())
    rb2 = st.selectbox("Select Running Back 2", df[df['Position'] == 'RB']['Player'].tolist())
    wr1 = st.selectbox("Select Wide Receiver 1", df[df['Position'] == 'WR']['Player'].tolist())
    wr2 = st.selectbox("Select Wide Receiver 2", df[df['Position'] == 'WR']['Player'].tolist())
    te = st.selectbox("Select Tight End", df[df['Position'] == 'TE']['Player'].tolist())
    flex = st.multiselect("Select Flex Players", df[(df['Position'] == 'RB') | (df['Position'] == 'WR') | (df['Position'] == 'TE')]['Player'].tolist(), max_selections=2)

    # Combine selected players for your team
    selected_players = [qb, rb1, rb2, wr1, wr2, te] + flex

    # Plot your team's predicted scores with probability ranges
    fig = plot_player_scores(selected_players, team_name="Your Team")
    st.plotly_chart(fig, key="your_team_plot")  # Added unique key for this plot

    # Calculate and display the total predicted score for your team
    your_team_mean, your_team_stddev = calculate_team_stats(selected_players)
    st.write(f"Your Team's Total Predicted Score Mean: {your_team_mean:.1f}")
    st.write(f"Your Team's Total Predicted Score StdDev: {your_team_stddev:.1f}")

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
    opponent_team_mean, opponent_team_stddev = calculate_team_stats(opponent_selected_players)
    st.write(f"Opponent's Total Predicted Score Mean: {opponent_team_mean:.1f}")
    st.write(f"Opponent's Total Predicted Score StdDev: {opponent_team_stddev:.1f}")

# Calculate probability of your team winning
probability = calculate_probability(your_team_mean, your_team_stddev, opponent_team_mean, opponent_team_stddev)

# Display a comparison table of the total predicted scores and the winning probability
st.write("### Total Predicted Score Comparison")
comparison_df = pd.DataFrame({
    "Team": ["Your Team", "Opponent's Team"],
    "Total Predicted Score": [round(your_team_mean, 1), round(opponent_team_mean, 1)],
    "Winning Probability (Your Team)": [f"{round(probability * 100, 2)}%", "-"]
})

st.write(comparison_df)
st.write("The probability of your team winning is computed based on comparing the two teams' predicted scores using a normal distribution.")
