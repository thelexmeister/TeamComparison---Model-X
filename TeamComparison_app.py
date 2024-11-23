import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load the data from an Excel file
df = pd.read_excel('player_median_scores.xlsx')

# Calculate the lower and upper bounds based on the probability
df['Lower Bound'] = df['Adjusted Median Score'] - df['Adjusted Median Score'] * ((0.5 - df['Probability']) / 2)
df['Upper Bound'] = df['Adjusted Median Score'] + df['Adjusted Median Score'] * ((0.5 - df['Probability']) / 2)

# Function to plot the player's predicted scores with probability ranges
def plot_player_scores(players):
    fig = go.Figure()

    for player in players:
        player_data = df[df['Player'] == player].iloc[0]
        
        # Round the predicted score to 1 decimal place
        predicted_score = round(player_data['Adjusted Median Score'], 1)

        fig.add_trace(go.Scatter(
            x=[player_data['Player']],
            y=[predicted_score],
            mode='markers+text',
            marker=dict(size=12, color='blue'),
            text=f"{predicted_score}",  # Display the rounded score
            textposition="top center"
        ))

        fig.add_trace(go.Scatter(
            x=[player_data['Player'], player_data['Player']],
            y=[player_data['Lower Bound'], player_data['Upper Bound']],
            mode='lines',
            line=dict(width=2, color='orange'),
            showlegend=False
        ))

    fig.update_layout(
        title="Player Predicted Scores with Probability Ranges",
        xaxis_title="Player",
        yaxis_title="Predicted Score",
        showlegend=False
    )
    return fig

# Streamlit User Interface
st.title('NFL Fantasy Team Prediction Dashboard')

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
    fig = plot_player_scores(selected_players)
    st.plotly_chart(fig)

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
    fig_opponent = plot_player_scores(opponent_selected_players)
    st.plotly_chart(fig_opponent)

    # Calculate and display the total predicted score for the opponent's team
    opponent_total_score = sum(df[df['Player'] == player]['Adjusted Median Score'].iloc[0] for player in opponent_selected_players)
    st.write(f"Total Predicted Score for Opponent's Team: {round(opponent_total_score, 1)}")

# Display a comparison table of the total predicted scores
st.write("### Total Predicted Score Comparison")
comparison_df = pd.DataFrame({
    "Team": ["Your Team", "Opponent's Team"],
    "Total Predicted Score": [round(total_score, 1), round(opponent_total_score, 1)]
})

st.write(comparison_df)



