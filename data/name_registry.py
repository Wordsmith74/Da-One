import logging

# Configure logging to see the missing alias warnings in your console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Dictionaries ---
# (Ensure your existing NBA_TEAM and WNBA_TEAM dictionaries are defined here as they were in 1000006787.png)

def get_team(team_name):
    """
    Looks up a team key based on full name or aliases.
    Logs a warning if the team_name is not found.
    """
    # Combine NBA and WNBA teams for a unified search
    all_teams = list(NBA_TEAM.items()) + list(WNBA_TEAM.items())
    
    for key, data in all_teams:
        # Normalize input and stored data for case-insensitive matching
        target = team_name.lower()
        full_name = data['full'].lower()
        aliases = [a.lower() for a in data.get('aliases', [])]
        
        if target == full_name or target in aliases:
            return key
            
    # If the loop finishes, the alias is missing
    logger.warning(f"DEBUG: Missing alias found for team: {team_name}")
    return None

# --- Rest of your existing functions (e.g., get_park, etc.) ---
