import streamlit as st
try:
    import torch
except:
    import os
    os.system("pip install torch --index-url https://download.pytorch.org/whl/cpu")
    import torch
import torch.nn as nn

heroes_correspondances = {
    0: "darkstar",
    1: "xesha",
    2: "tempus",
    3: "judge",
    4: "tristan",
    5: "corvus",
    6: "galaad",
    7: "mushy",
    8: "helios",
    9: "martha",
    10: "phobos",
    11: "kai",
    12: "andvari",
    13: "kayla",
    14: "mojo",
    15: "krista",
    16: "jhu",
    17: "aidan",
    18: "iris",
    19: "julius",
    20: "jorgen",
    21: "cornelius",
    22: "yasmine",
    23: "astaroth",
    24: "luther",
    25: "faceless",
    26: "celeste",
    27: "dante",
    28: "ziri",
    29: "chabba",
    30: "oya",
    31: "octavia",
    32: "morrigan",
    33: "cleaver",
    34: "qingmao",
    35: "maya",
    36: "keira",
    37: "nebula",
    38: "fox",
    39: "marcus",
    40: "karkh",
    41: "orion",
    42: "jet",
    43: "lian",
    44: "sun",
    45: "aurora",
    46: "rufus",
    47: "lilith",
    48: "fafnir",
    49: "heidi",
    50: "amira",
    51: "ginger",
    52: "lars",
    53: "peppy",
    54: "satori",
    55: "elmir",
    56: "alvanor",
    57: "astridlucas",
    58: "sebastian",
    59: "daredevil",
    60: "arachne",
    61: "cascade",
    62: "ginger",
    63: "thea",
    64: "ishmael",
    65: "isaac",
    66: "folio",
    67: "artemis",
    68: "dorian",
}
from bidict import bidict

correspondances = bidict(
    {
        hero_name: i
        for i, hero_name in enumerate(
            sorted(list(set(heroes_correspondances.values())))
        )
    }
)

num_heroes = len(correspondances)


from torch import nn


class SimpleFightPredictor(nn.Module):
    def __init__(self, num_heroes, hidden_dim, dropout_rate=0.5):
        super(SimpleFightPredictor, self).__init__()
        self.dropout_rate = dropout_rate
        # Encoding layers for each team
        self.team_encoder = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_heroes, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
        )

        # Final fully connected layers for combining team representations
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, left_team, right_team):
        # Encode each team separately
        left_team_encoded = self.team_encoder(
            left_team
        )  # Shape: (batch_size, hidden_dim)
        right_team_encoded = self.team_encoder(
            right_team
        )  # Shape: (batch_size, hidden_dim)
        # Concatenate team encodings and pass through final layers
        combined_rep = torch.cat(
            [left_team_encoded, right_team_encoded], dim=1
        )  # Shape: (batch_size, hidden_dim * 2)
        return self.final_fc(combined_rep)  # Shape: (batch_size, 1)


best_params = {
    "hidden_dim": 1904,
    "batch_size": 8,
    "optim": "SGD",
    "decay": 0.0003158359706882851,
    "lr": 0.11364436878365329,
    "dropout_rate": 0.12202705797845492,
}
hidden_dim = best_params["hidden_dim"]
batch_size = best_params["batch_size"]
optim = best_params["optim"]
decay = best_params["decay"]
lr = best_params["lr"]
dropout_rate = best_params["dropout_rate"]



# Load your model
@st.cache_resource
def load_model():
    model = SimpleFightPredictor(
        num_heroes=num_heroes, hidden_dim=hidden_dim, dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.eval()
    return model

model = load_model()

# Streamlit app title
st.title("🎈 Hero wars Legends Draft outcome prediction")

# Define dropdown options
options = sorted(list(heroes_correspondances.values()))
level_options = [3, 4, 5, 6]
# User input section with dropdowns
st.write("Please select the values for prediction. The order doesn't matter.")

# Define columns for Attack and Defense inputs
col1, col2 = st.columns(2)

# Attack inputs (left column)
with col1:
    st.subheader("Attack Team")
    attack_hero_inputs = []
    attack_level_inputs = []
    for i in range(5):
        row = st.columns(2)
        with row[0]:
            hero_input = st.selectbox(f"Hero", options, index=0, key=f"input_{i}_hero_attack")
        with row[1]:
            level_input = st.selectbox(f"Star level", level_options, index=0, key=f"input_{i}_level_attack")
        attack_hero_inputs.append(hero_input)
        attack_level_inputs.append(level_input)

# Defense inputs (right column)
with col2:
    st.subheader("Defense Team")
    defense_hero_inputs = []
    defense_level_inputs = []
    for i in range(5):
        row = st.columns(2)
        with row[0]:
            hero_input = st.selectbox(f"Hero", options, index=0, key=f"input_{i}_hero_defense")
        with row[1]:
            level_input = st.selectbox(f"Star level", level_options, index=0, key=f"input_{i}_level_defense")
        defense_hero_inputs.append(hero_input)
        defense_level_inputs.append(level_input)

# Create a button to make predictions
if st.button("Predict"):
    # Prepare the input tensor (modify based on your model’s expected input format)
    team_tensor = torch.zeros((1, num_heroes))
    for hero_name, hero_level in zip(attack_hero_inputs, attack_level_inputs):
        team_tensor[0, correspondances[hero_name]] = (hero_level - 2.0) / 4.0
    opposing_team = torch.zeros((1, num_heroes))
    for hero_name, hero_level in zip(defense_hero_inputs, defense_level_inputs):
        opposing_team[0, correspondances[hero_name]] = (hero_level - 2.0) / 4.0


    # Make prediction
    with torch.no_grad():
        prediction = model(team_tensor, opposing_team)[0].item()  # Convert tensor to scalar
    
    # Display prediction
    st.write(f"Attack team has a {round(100*prediction, 2)}% chance to win")

import numpy as np
# Create a button to make predictions
if st.button("Advise"):
    # Prepare the input tensor (modify based on your model’s expected input format)
    attack_heroes_ids = [correspondances[hero_name] for hero_name in attack_hero_inputs]
    team_tensor = torch.zeros((1, num_heroes))
    for hero_name, hero_level in zip(attack_hero_inputs, attack_level_inputs):
        team_tensor[0, correspondances[hero_name]] = (hero_level - 2.0) / 4.0
    opposing_team = torch.zeros((1, num_heroes))
    for hero_name, hero_level in zip(defense_hero_inputs, defense_level_inputs):
        opposing_team[0, correspondances[hero_name]] = (hero_level - 2.0) / 4.0
    # Make prediction
    with torch.no_grad():
        current_win_probability = model(team_tensor, opposing_team)[0].item()
    heroes_to_swap = torch.nonzero(team_tensor)
    heroes_swap_options = {}
    for hero_index_to_swap in range(5):
        hero_alternate_teams = torch.clone(team_tensor).repeat(num_heroes, 1)
        hero_to_swap = heroes_to_swap[hero_index_to_swap][1].item()
        hero_level = team_tensor[0, hero_to_swap]
        hero_swap_options = {} 
        for hero_id in range(num_heroes):
            hero_alternate_teams[hero_id, hero_to_swap] = 0.
            hero_alternate_teams[hero_id, hero_id] = hero_level
        with torch.no_grad():
            prediction = model(hero_alternate_teams, opposing_team.repeat(num_heroes, 1)).numpy().ravel()
        best_heroes = np.argsort(prediction)
        for hero_id in best_heroes[::-1]:
            if hero_id not in attack_heroes_ids:
                win_probability = prediction[hero_id]
                best_hero_to_swap_for = hero_id
                break
        heroes_swap_options[correspondances.inverse[hero_to_swap]] = [correspondances.inverse[best_hero_to_swap_for], win_probability]
    hero_to_swap, (chosen_hero, better_win_probability) = max(heroes_swap_options.items(), key=lambda x:x[1][1])
    if better_win_probability > current_win_probability:
        better_win_probability = round(100 * better_win_probability, 2)
        # Display prediction
        st.write(f"Swap out {hero_to_swap} for {chosen_hero} to increase your win probability to {better_win_probability:.2f}%")
    else:
        st.write(f"According to me, there is no better attack team than this one to beat the current defense team.")
