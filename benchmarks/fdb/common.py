"""Full-Duplex-Bench source revision and v1.0 tasks."""

FDB_COMMIT = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"

FDB_CATEGORIES = {
    "candor_turn_taking": 119,
    "synthetic_user_interruption": 200,
    "candor_pause_handling": 216,
}


TASKS = {
    "smooth_turn_taking": "candor_turn_taking",
    "user_interruption": "synthetic_user_interruption",
    "pause_handling": "candor_pause_handling",
}
