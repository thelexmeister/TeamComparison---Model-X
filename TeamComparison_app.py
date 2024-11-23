import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set the layout to wide
st.set_page_config(layout="wide")

# Load the data from an Excel file
df = pd.read_excel('player_median_scores.xlsx')

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


# Elo Probability Calculation Function
def calculate_elo_probability(team_score, opponent_score):
    # Elo rating system formula
    probability = 1 / (1 + 10 ** ((opponent_score - team_score) / 400))
    return probability

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
    total_score = sum(df[df['Player'] == player]['Adjusted Median Score'].iloc[0] for player in selected_players)
    st.write(f"Total Predicted Score for Your Team: {round(total_score, 1)}")

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

# Calculate the Elo probability of your team winning
elo_probability = calculate_elo_probability(total_score, opponent_total_score)

# Display a comparison table of the total predicted scores
st.write("### Total Predicted Score Comparison")
comparison_df = pd.DataFrame({
    "Team": ["Your Team", "Opponent's Team"],
    "Total Predicted Score": [round(total_score, 1), round(opponent_total_score, 1)],
    "Winning Probability (Your Team)": [f"{round(elo_probability * 100, 2)}%", "-"]
})

st.write(comparison_df)


