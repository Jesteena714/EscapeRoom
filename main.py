import streamlit as st
import time
import pandas as pd
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# Function to load the Welcome Page
def welcome_page():
    # Background Image Fix
    page_bg_img ="""
    <style>
    [data-testid="stAppViewContainer"] {
        background: url("https://cdn.gamma.app/to1aya4l718rehv/generated-images/sh8NM6TtzbzSujPkwJ-12.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);  /* Transparent header */
    }
    [data-testid="stToolbar"] {
        right: 2rem;  /* Adjust toolbar position */
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0);  /* Fully transparent background for content */
        padding: 20px;
    }
    .overlay {
        background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent black overlay */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Sidebar - Restart Button
    with st.sidebar:
        st.markdown("### 🎮 Game Controls")
        if st.button("🔄 Restart Game", key="restart_button"):
            st.session_state.current_room = "welcome_page"
            st.session_state.start_time = None
            st.session_state.time_left = 300
            st.session_state.hint_used = False
            st.session_state.completed_rooms = set()
            st.rerun()

    # Title and Introduction with Overlay
    st.markdown(
        """
        <div class="overlay">
            <h1 style='font-size: 45px; color: white;'>🚢 The Great Shipping Escape</h1>
            <p style='font-size: 22px; font-weight: bold; color: white;'>
                🌎 A <strong>rogue AI</strong> has scrambled all logistics data! 📦 Chaos in supply chains!  
                🧠 Can <strong>you</strong> solve the puzzles & save the world?  
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Start Game Button
    if st.button("🚀 Start Your Mission!", key="start_button"):
        st.session_state.current_room = "room_1"
        st.session_state.start_time = time.time()
        st.session_state.time_left = 300
        st.session_state.hint_used = False
        st.rerun()





room1_background = """
    <style>
    .room1-page {
        background-image: url("https://images.unsplash.com/photo-1530533718754-001d2668365a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .room1-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .room1-text {
        font-size: 16px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
"""
room2_background = """
    <style>
    .room2-page {
        background-image: url("https://images.unsplash.com/photo-1530533718754-001d2668365a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .room2-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .room2-text {
        font-size: 16px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
"""


room3_background = """
    <style>
    .room3-page {
        background-image: url("https://images.unsplash.com/photo-1530533718754-001d2668365a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .room3-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .room3-text {
        font-size: 16px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
"""

room4_background = """
    <style>
    .room4-page {
        background-image: url("https://images.unsplash.com/photo-1530533718754-001d2668365a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .room4-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .room4-text {
        font-size: 16px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
"""
# Function for Room 1: Descriptive Statistics - The Manifest Mishap
def room_1():
    st.title("Room 1: Descriptive Statistics – The Manifest Mishap")
    st.markdown(room1_background, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="room1-page">
            <div class="room1-title">🚢 Room 1: Descriptive Statistics – The Manifest Mishap</div>
            <div class="room1-text">
                <h3>📋 Objective</h3>
                <p>
                    The rogue AI, LogiX, has corrupted the cargo manifests, scrambling the weight data. Your task is to analyze the dataset using descriptive statistics, identify anomalies, and unlock the next room.
                </p>
                <p>
                    <strong>LogiX's Message:</strong> "Welcome, human. You think you can outsmart me? Prove your worth by identifying the anomalies in the cargo data. Fail, and the supply chain collapses."
                </p>
                <h3>📋 Challenges</h3>
                <ul>
                    <li>Calculate mean, median, and mode to detect abnormal cargo weights.</li>
                    <li>Use standard deviation to identify outliers in the shipments.</li>
                    <li>Apply percentiles to compare shipments and detect any unusual trends.</li>
                </ul>
                <p>
                    <strong>Reward:</strong> Successfully solving the challenges will grant you a <strong>5-digit passcode</strong> to use in Room 3.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load dataset
    df = pd.read_csv("ship data.csv")

       # Questions pool with fixed syntax
    questions_pool = [
        ("What is the mode weight of the cargo? (Use the 'Weight' column)", df["Weight"].mode()[0]),
        ("What is the mean weight of the cargo (rounded to 2 decimal places)? (Use the 'Weight' column)", round(df["Weight"].mean(), 2)),
        ("How many outliers are there in the dataset? (Use the 'Weight' column)", 
 sum((df["Weight"] < (df["Weight"].quantile(0.25) - 1.5 * (df["Weight"].quantile(0.75) - df["Weight"].quantile(0.25)))) | 
     (df["Weight"] > (df["Weight"].quantile(0.75) + 1.5 * (df["Weight"].quantile(0.75) - df["Weight"].quantile(0.25)))))),
        ("What is the median weight of the cargo? (Use the 'Weight' column)", df["Weight"].median()),
        ("What is the standard deviation of the cargo weight? (Use the 'Weight' column)", round(df["Weight"].std(), 2)),
        ("What is the 25th percentile of cargo weight? (Use the 'Weight' column)", df["Weight"].quantile(0.25)),
        ("What is the 75th percentile of cargo weight? (Use the 'Weight' column)", df["Weight"].quantile(0.75)),
        ("What is the range of cargo weight? (Use the 'Weight' column)", df["Weight"].max() - df["Weight"].min())
    ]

    # Initialize session state for selected questions
    if "selected_questions" not in st.session_state:
        st.session_state.selected_questions = random.sample(questions_pool, 3)

    # Display dataset
    st.write("### 📦 **Ship Data Overview**")
    st.dataframe(df)

    # Display descriptive summary
    st.write("### 📊 **Descriptive Summary of Cargo Weights**")
    descriptive_stats = df["Weight"].describe().to_frame()
    descriptive_stats.loc["Mode"] = df["Weight"].mode()[0]
    descriptive_stats.loc["Range"] = df["Weight"].max() - df["Weight"].min()
    IQR = df["Weight"].quantile(0.75) - df["Weight"].quantile(0.25)
    outliers = sum((df["Weight"] < (df["Weight"].quantile(0.25) - 1.5 * IQR)) | 
                    (df["Weight"] > (df["Weight"].quantile(0.75) + 1.5 * IQR)))
    descriptive_stats.loc["Outliers"] = outliers
    st.write(descriptive_stats)

    # Timer logic
    current_time = time.time()
    time_elapsed = current_time - st.session_state.start_time
    st.session_state.time_left = max(300 - time_elapsed, 0)

    # Display live countdown timer
    timer_placeholder = st.empty()
    with timer_placeholder:
        st.write(f"### ⏳ **Time Left:** {int(st.session_state.time_left // 60)} min {int(st.session_state.time_left % 60)} sec")

    # Add a progress bar for the timer
    progress_bar = st.progress(1.0)  # Initialize progress bar at 100%
    progress = st.session_state.time_left / 300  # Calculate progress (0 to 1)
    progress_bar.progress(progress)

    # Check if time is up
    if st.session_state.time_left <= 0:
        st.error("⏳ **Time's up!** The rogue AI has locked the system. Try again! 🚨")
        st.session_state.selected_questions = random.sample(questions_pool, 3)
        st.session_state.start_time = time.time()
        st.session_state.time_left = 300
        st.rerun()

    # Display questions
    st.write("### 🕵️‍♂️ **Solve These Challenges:**")
    answers = []
    for i, (question, correct_answer) in enumerate(st.session_state.selected_questions):
        # Allow user input without automatic rounding
        answers.append(st.number_input(question, key=f"answer_{i}", step=0.01))  # Unique key for each question

    # Hint system
    st.write("### 💡 **Hint System**")
    if not st.session_state.hint_used:
        st.write("You can use **1 hint** during the game. Choose wisely!")
        hint_question = st.selectbox(
            "Choose a question to get a hint:",
            [q for q, _ in st.session_state.selected_questions],
            index=0,
            key="hint_question"
        )
        if st.button("Get Hint"):
            st.session_state.hint_used = True
            # Find the selected question and provide a hint
            for i, (question, correct_answer) in enumerate(st.session_state.selected_questions):
                if question == hint_question:
                    if "mode" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The mode is the most frequently occurring value in the dataset. Look for the weight that appears the most.")
                    elif "mean" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The mean is the average of all weights. Add up all the weights and divide by the number of cargo items.")
                    elif "outliers" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** Outliers are values that are significantly higher or lower than the rest. Use the IQR method to find them.")
                        st.write(f"💡 **IQR Formula:** IQR = Q3 - Q1")
                        st.write(f"💡 **Q1 (25th percentile):** {df['Weight'].quantile(0.25):.2f}")
                        st.write(f"💡 **Q3 (75th percentile):** {df['Weight'].quantile(0.75):.2f}")
                        st.write(f"💡 **Lower Bound:** Q1 - 1.5 * IQR = {df['Weight'].quantile(0.25) - 1.5 * IQR:.2f}")
                        st.write(f"💡 **Upper Bound:** Q3 + 1.5 * IQR = {df['Weight'].quantile(0.75) + 1.5 * IQR:.2f}")
                    elif "median" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The median is the middle value when the weights are sorted in ascending order.")
                    elif "standard deviation" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** Standard deviation measures how spread out the weights are. A higher value means more variability.")
                    elif "25th percentile" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The 25th percentile (Q1) is the value below which 25% of the weights fall.")
                        st.write(f"💡 **Q1 (25th percentile):** {df['Weight'].quantile(0.25):.2f}")
                    elif "75th percentile" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The 75th percentile (Q3) is the value below which 75% of the weights fall.")
                        st.write(f"💡 **Q3 (75th percentile):** {df['Weight'].quantile(0.75):.2f}")
                    elif "range" in question.lower():
                        st.write(f"💡 **Hint for '{question}':** The range is the difference between the maximum and minimum weights in the dataset.")
                    break
    else:
        st.write("💡 **Hint:** You've already used your hint for this game. No more hints are available.")

    # Submit answers
    if st.button("🚀 **Submit Answers**"):
        correct = all(
            round(float(answers[i]), 2) == round(float(correct_answer), 2)  # Round both for comparison
            for i, (_, correct_answer) in enumerate(st.session_state.selected_questions)
        )
        if correct:
            # Generate a 5-digit passcode
            passcode = random.randint(10000, 99999)
            st.session_state.passcode = passcode  # Store passcode in session state for Room 3

            # Ship sailing animation
            st.markdown("""
                <style>
                @keyframes sail {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
                .sail-animation {
                    animation: sail 3s linear;
                    font-size: 50px;
                    text-align: center;
                }
                </style>
                <div class="sail-animation">🚢</div>
            """, unsafe_allow_html=True)
            st.success(f"🎉 **Correct!** You've identified the anomalies and unlocked the next room. 🗝️")
            st.write(f"🔑 **Your 5-digit passcode for Room 3:** `{passcode}`")
            st.write("⚠️ **Note:** Save this passcode! You'll need it in Room 3 to perform regression analysis.")
            st.session_state.current_room = "room_2"
            st.rerun()
        else:
            st.error("❌ **Incorrect.** Please check your answers and try again. 🧐")

    # Footer (displayed only once)
    if "footer_displayed" not in st.session_state:
        st.session_state.footer_displayed = True
        st.markdown("---")
        st.write("🌟 **Tip:** Use the descriptive statistics above to help you solve the challenges. Good luck, Captain! 🫡")

# Function for Room 2: Data Visualization - The Hidden Shipping Routes
def room_2():
    st.title("Room 2: Data Visualization - The Hidden Shipping Routes")
    st.markdown(room2_background, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="room2-page">
    <div class="room2-title">🚢 Room 2: Data Visualization - The Hidden Shipping Routes</div>
    <div class="room2-text">
        <h3>📋 Objective</h3>
        <p>
            The AI has encrypted key trade pathways, making it impossible for ships to navigate efficiently.
            Use data visualization to uncover hidden patterns in the shipping data and unlock the secret code.
        </p>
        <p>
            <strong>LogiX's Message:</strong> "Prove your analytical prowess by deciphering the shipping anomalies.
            Your success is the only chance to keep the supply lines open. Identify the busiest routes, spot the congestion points, and decode the delays."
        </p>
        <h3>📋 Challenges</h3>
        <ul>
            <li>Create a bar chart to determine the route with the highest trade volume.</li>
            <li>Generate a scatter plot to identify the route suffering from the worst congestion.</li>
            <li>Build a heatmap to decode the shipping delays and reveal the 5-digit passcode.</li>
        </ul>
        <p>
            <strong>Reward:</strong> Successfully solving these challenges will grant you a <strong>5-digit passcode</strong> to be used in Room 3.
        </p>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )
    # Timer logic
    current_time = time.time()
    time_elapsed = current_time - st.session_state.start_time
    st.session_state.time_left = max(300 - time_elapsed, 0)
    
    # Display live countdown timer
    timer_placeholder = st.empty()
    with timer_placeholder:
        st.write(f"### ⏳ **Time Left:** {int(st.session_state.time_left // 60)} min {int(st.session_state.time_left % 60)} sec")


    # Add a progress bar for the timer
    progress_bar = st.progress(1.0)  # Initialize progress bar at 100%
    progress = st.session_state.time_left / 300  # Calculate progress (0 to 1)
    progress_bar.progress(progress)

    # Check if time is up
    if st.session_state.time_left <= 0:
        st.error("⏳ **Time's up!** The rogue AI has locked the system. Try again! 🚨")
        st.session_state.start_time = time.time()
        st.session_state.time_left = 300
        st.rerun()

   # Generate Random Shipping Data
    np.random.seed(42)
    data = pd.DataFrame({
       'Route': [f'Route {i}' for i in range(1, 11)],
       'Trade Volume': np.random.randint(50, 500, 10),
       'Congestion Index': np.random.rand(10) * 100,
       'Delay Hours': np.random.randint(1, 50, 10)
    })

   # Challenge 1: Histogram & Bar Chart
    st.subheader("Challenge 1: Identify the Busiest Trade Route")
    fig, ax = plt.subplots()
    sns.barplot(x=data['Route'], y=data['Trade Volume'], ax=ax, palette='Blues')
    ax.set_xticklabels(data['Route'], rotation=45)
    st.pyplot(fig)

    busiest_route = data.loc[data['Trade Volume'].idxmax(), 'Route']
    user_answer1 = st.text_input("Which route has the highest trade volume? (e.g., Route 3)")

    if user_answer1 == busiest_route:
       st.success("✅ Correct! Proceed to the next challenge.")
       st.session_state.challenge = 2

   # Challenge 2: Scatter Plot for Congestion Points
    if st.session_state.get("challenge", 1) >= 2:
       st.subheader("Challenge 2: Identify the Most Congested Trade Route")
       fig, ax = plt.subplots()
       sns.scatterplot(x=data['Trade Volume'], y=data['Congestion Index'], size=data['Congestion Index'], sizes=(20, 200))
       st.pyplot(fig)

       most_congested_route = data.loc[data['Congestion Index'].idxmax(), 'Route']
       user_answer2 = st.text_input("Which route has the highest congestion index? (e.g., Route 3)")

       if user_answer2:
           if user_answer2 == most_congested_route:
               st.success("✅ Correct! Proceed to the final challenge.")
               st.session_state.challenge = 3
           else:
               st.error("❌ Incorrect answer. Try again!")
               st.info("💡 Hint: Each point represents a trade route. Find the route with the highest congestion index!")

   # Challenge 3: Heatmap for Shipping Delays
    if st.session_state.get("challenge", 1) >= 3:
       st.subheader("Challenge 3: Decode Shipping Delays")
       pivot_data = data.pivot_table(values='Delay Hours', index='Route')
       fig, ax = plt.subplots()
       sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
       st.pyplot(fig)

       # Extracting the correct answer from the heatmap (largest delay value)
       correct_code = str(int(pivot_data.values.max()))  # Convert to string
       user_code = st.text_input("Enter the final code from the heatmap:")

       if user_code:
           if user_code == correct_code:
               st.success(f"✅ You've decoded the shipping delays! Your code to the next room is: **{correct_code}**")

               # Button to proceed to Room 3
               if st.button("🔹 Proceed to Room 3"):
                   st.session_state.current_room = "room_3"
                   st.rerun()

           else:
               st.error("❌ Incorrect code. Try again!")
               # Add Hint after wrong attempts
               st.info("💡 Hint: Look for the **darkest red cell** in the heatmap. The **largest** number is the code!")

        

    # Main App Logic: Handling Room Navigation


# ROOM 3: Regression Analysis
def room_3():
    st.title("Room 3: Regression Analysis – Forecasting Stormy Seas")
    st.markdown(room3_background, unsafe_allow_html=True)
    st.markdown(
    """
    <div class="room3-page">
        <div class="room3-title">🚢 Room 3: Regression Analysis – Forecasting Stormy Seas</div>
        <div class="room3-text">
            <h3>📋 Objective</h3>
            <p>
                The AI has manipulated shipping logs and weather conditions, distorting real delay times. Your mission is to harness the power of regression analysis to forecast shipping delays accurately.
            </p>
            <p>
                <strong>LogiX's Message:</strong> "Predict the stormy delays, if you dare. Use your analytical skills to foresee the disruptions in the supply chain. Your success ensures that the ships sail smoothly; your failure could spell chaos."
            </p>
            <h3>📋 Challenges</h3>
            <ul>
                <li>Examine a simulated dataset containing shipping distances, weather severity, and port congestion.</li>
                <li>Train a linear regression model to predict shipping delays.</li>
                <li>Input realistic conditions and compare your prediction to the model's output.</li>
            </ul>
            <p>
                <strong>Reward:</strong> Master the regression model and secure a <strong>5-digit passcode</strong> to unlock Room 4.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    # Simulated dataset
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    data = {
        'Distance (km)': np.random.randint(100, 10000, 50),
        'Weather Severity (1-10)': np.random.randint(1, 11, 50),
        'Port Congestion (1-10)': np.random.randint(1, 11, 50),
        'Shipping Delay (hours)': np.random.randint(5, 100, 50)
    }
    df = pd.DataFrame(data)

    # Train a regression model
    X = df[['Distance (km)', 'Weather Severity (1-10)', 'Port Congestion (1-10)']]
    y = df['Shipping Delay (hours)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Streamlit UI
    #st.title("🚢 Room 3: Regression Analysis – Forecasting Stormy Seas")
    #st.write("Predict shipping delays using regression models.")

    # Show dataset
    st.subheader("Shipping Logs & Weather Conditions")
    st.dataframe(df)

    # User input
    st.subheader("Enter Conditions to Predict Delay Time")
    distance = st.number_input("Distance (km)", min_value=100, max_value=10000, value=1000, step=100)
    weather = st.slider("Weather Severity (1-10)", min_value=1, max_value=10, value=5)
    congestion = st.slider("Port Congestion (1-10)", min_value=1, max_value=10, value=5)

    # Predict button
    if st.button("Predict Shipping Delay"):
        prediction = model.predict([[distance, weather, congestion]])[0]
        st.session_state["prediction"] = prediction  # Store prediction in session state
        st.success(f"Predicted Shipping Delay: {prediction:.2f} hours")

    # User enters their predicted answer
    if "prediction" in st.session_state:
        st.subheader("Enter Your Answer")
        user_answer = st.number_input("What is the predicted shipping delay time?", min_value=0.0, step=0.1)

        if st.button("Submit Answer"):
            if abs(user_answer - st.session_state["prediction"]) < 1.0:  # Allowing small error margin
                st.success("Correct! You may proceed to the next room.")
                st.session_state.current_room = "room_4"  # Change room
                st.rerun()  # Refresh the app to reflect the new room
            else:
                st.error("Incorrect! Try again.")

def room_4():
    #st.title("Room 5: Markov Chain – Cracking the AI’s Code")
    st.title("Room 4: Markov Chain – Cracking the AI’s Code")
    st.markdown(room4_background, unsafe_allow_html=True)
    st.markdown(
    """
    <div class="room4-page">
        <div class="room4-title">🚢 Room 4: Markov Chain – Cracking the AI’s Code</div>
        <div class="room4-text">
            <h3>📋 Objective</h3>
            <p>
                Welcome to the high-tech command center, where the rogue AI is making real-time decisions based on probability states. 
    Your objective is to predict the AI's next move and override its logic using Markov Chains. In the command center, the rogue AI is making decisions using complex probability states. Your task is to decode its logic by predicting its next move using Markov Chains and overriding its decision.
            </p>
            <p>
                <strong>LogiX's Message:</strong> "I control the probabilities, but can you control fate? Decode my transition matrix and take control of the next state. Only then will you break my algorithmic hold on the supply chain."
            </p>
            <h3>📋 Challenges</h3>
            <ul>
                <li>Understand the current AI state and the provided transition matrix.</li>
                <li>Predict the AI's next state based on given probabilities.</li>
                <li>Override the AI's decision to progress further.</li>
            </ul>
            <p>
                <strong>Reward:</strong> Successfully predicting and overriding the AI’s move will allow you to proceed and receive the final code to end the game.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    

    # Timer logic
    current_time = time.time()
    time_elapsed = current_time - st.session_state.start_time
    st.session_state.time_left = max(300 - time_elapsed, 0)

    # Display live countdown timer
    timer_placeholder = st.empty()
    with timer_placeholder:
        st.write(f"### ⏳ **Time Left:** {int(st.session_state.time_left // 60)} min {int(st.session_state.time_left % 60)} sec")

    # Add a progress bar for the timer
    progress_bar = st.progress(1.0)  # Initialize progress bar at 100%
    progress = st.session_state.time_left / 300  # Calculate progress (0 to 1)
    progress_bar.progress(progress)

    # Define the States and Transition Matrix for the Markov Chain
    states = ['Idle', 'Analyzing', 'Executing', 'Alerting', 'Overriding']
    transition_matrix = np.array([
     [0.1, 0.4, 0.3, 0.1, 0.1], # Idle -> [Idle, Analyzing, Executing, Alerting, Overriding]
    [0.2, 0.2, 0.5, 0.05, 0.05], # Analyzing -> [Idle, Analyzing, Executing, Alerting, Overriding]
    [0.1, 0.1, 0.7, 0.05, 0.05], # Executing -> [Idle, Analyzing, Executing, Alerting, Overriding]
    [0.15, 0.15, 0.15, 0.4, 0.15], # Alerting -> [Idle, Analyzing, Executing, Alerting, Overriding]
    [0.2, 0.2, 0.2, 0.2, 0.2], # Overriding -> [Idle, Analyzing, Executing, Alerting, Overriding]
    ])

    # Display the current state and transition matrix
    current_state = st.session_state.get("current_state", "Idle")
    st.write(f"Current State: **{current_state}**")
 
    st.subheader("Transition Matrix:")
    transition_df = {
        'Idle': [0.1, 0.4, 0.3, 0.1, 0.1],
        'Analyzing': [0.2, 0.2, 0.5, 0.05, 0.05],
        'Executing': [0.1, 0.1, 0.7, 0.05, 0.05],
        'Alerting': [0.15, 0.15, 0.15, 0.4, 0.15],
        'Overriding': [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    st.write(pd.DataFrame(transition_df, index=states))

    # Predict the next state (based on the transition probabilities)
    st.subheader("Challenge: Predict the AI's next move!")
    st.write("The AI is currently in a state, and it will transition to another state based on probabilities.")
 
    next_state_probabilities = transition_matrix[states.index(current_state)]
    st.write(f"Next State Probabilities: {next_state_probabilities}")
 

    # Simulate the AI’s decision using the Markov Chain (by sampling based on probabilities)
    predicted_state = st.selectbox("Which state do you think the AI will move to?", states)

    ai_next_state = np.random.choice(states, p=next_state_probabilities)
 
    if predicted_state == ai_next_state:
        st.success(f"✅ Correct! The AI moved to **{ai_next_state}**.")
        st.session_state.current_state = ai_next_state
    else:
        st.error(f"❌ Incorrect. The AI actually moved to **{ai_next_state}**.")
 
    # Option to override AI’s decision using Markov Chain
    st.subheader("Override the AI's Decision!")
    if st.button("Override AI's Decision"):
    # Override the decision with a random choice from the current state’s transition probabilities
        override_state = np.random.choice(states, p=next_state_probabilities)
        st.success(f"✅ You've overridden the AI's decision! The new state is **{override_state}**.")
        st.session_state.current_state = override_state
    # Button to proceed to next room
    if st.session_state.get("current_state") == "Overriding":
        st.write("You have successfully cracked the AI's code and can move to the next room!")
        if st.button("🔹 Proceed to the Next Room"):
            st.session_state.current_room = "room_5"
            st.rerun()
            # Proceed to the next room
def room_5():
    # Set the background image using custom CSS
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("https://cdn.gamma.app/to1aya4l718rehv/generated-images/zqHA2DoM7kC6vdy8uKqeJ.jpg") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);  /* Transparent header */
        }}
        [data-testid="stToolbar"] {{
            right: 2rem;  /* Adjust toolbar position */
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0);  /* Fully transparent background for content */
            padding: 20px;
        }}
        .overlay {{
            background-color: rgba(0, 0, 0, 0.8);  /* Darker semi-transparent black overlay */
            padding: 40px;  /* Increased padding for better spacing */
            border-radius: 15px;  /* Rounded corners */
            color: white;  /* White text for better contrast */
            text-align: center;
            margin: 0 auto;  /* Center the overlay */
            max-width: 800px;  /* Limit width for better readability */
        }}
        .overlay h1 {{
            font-size: 45px;  /* Larger heading */
            margin-bottom: 20px;  /* Spacing below heading */
        }}
        .overlay p {{
            font-size: 22px;  /* Larger paragraph text */
            line-height: 1.6;  /* Improved line spacing */
            margin-bottom: 20px;  /* Spacing below paragraphs */
        }}
        .overlay h2 {{
            font-size: 36px;  /* Larger subheading */
            margin-top: 30px;  /* Spacing above subheading */
            margin-bottom: 20px;  /* Spacing below subheading */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Victory message with overlay
    st.markdown(
        """
        <div class="overlay">
            <h1>🎉 Congratulations, Commander!</h1>
            <p>
                <strong>Mission Accomplished!</strong><br><br>
                You have successfully navigated through the layers of the rogue AI's defenses. By decoding shipping routes, forecasting stormy seas with regression analysis, and cracking the AI’s Markov Chain, you have proven your analytical prowess.<br><br>
                <strong>LogiX's Final Message:</strong><br>
                "You have outsmarted my algorithms and disrupted my operations. The supply lines remain open and the future is secured—thanks to your brilliance. This is not the end, but the beginning of a new era of human ingenuity."<br><br>
                Thank you for playing. Your journey in overcoming the AI's challenges has come to a triumphant close.
            </p>
            <h2>🚀 <strong>GAME OVER</strong></h2>
            <p>
                🚀 You have completed all the rooms! Congratulations!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Restart button
    if st.button("🔄 Restart Game", key="restart_button_final"):
        st.session_state.current_room = "welcome_page"
        st.session_state.start_time = None
        st.session_state.time_left = 300
        st.session_state.hint_used = False
        st.session_state.completed_rooms = set()
        st.rerun()
        
def main():
    if "current_room" not in st.session_state:
        st.session_state.current_room = "welcome_page"

    if st.session_state.current_room == "welcome_page":
        welcome_page()
    elif st.session_state.current_room == "room_1":
        room_1()
    elif st.session_state.current_room == "room_2":
        room_2()
    elif st.session_state.current_room == "room_3":
        room_3()
    elif st.session_state.current_room == "room_4":
        room_4()
    else:
        st.title("🎉 Congratulations, Commander!")
        st.write("""
    **Mission Accomplished!**
    
    You have successfully navigated through the layers of the rogue AI's defenses. By decoding shipping routes, forecasting stormy seas with regression analysis, and cracking the AI’s Markov Chain, you have proven your analytical prowess.
    
    **LogiX's Final Message:**
    "You have outsmarted my algorithms and disrupted my operations. The supply lines remain open and the future is secured—thanks to your brilliance. This is not the end, but the beginning of a new era of human ingenuity."
    
    Thank you for playing. Your journey in overcoming the AI's challenges has come to a triumphant close.
    """)
        st.write("🚀 **GAME OVER**")
        st.write("🚀 You have completed all the rooms! Congratulations!")


# Run the main function to start the game
if __name__ == "__main__":
    main()
