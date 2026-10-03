import sqlite3

def setup_db():
    """Creates the database and the UserSettings table if they don't exist."""
    conn = sqlite3.connect('flowsync.db')
    cursor = conn.cursor()
    
    # Added UNIQUE constraint to prevent duplicate gesture mappings for the same user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UserSettings (
            user_name TEXT,
            gesture_name TEXT,
            action_mapped TEXT,
            UNIQUE(user_name, gesture_name)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database and Table are ready!")

def save_setting(user, gesture, action):
    """Saves a new setting or updates an existing one for the user."""
    conn = sqlite3.connect('flowsync.db')
    cursor = conn.cursor()
    
    # INSERT OR REPLACE ensures old settings are overwritten if updated
    cursor.execute('''
        INSERT OR REPLACE INTO UserSettings (user_name, gesture_name, action_mapped) 
        VALUES (?, ?, ?)
    ''', (user, gesture, action))
    
    conn.commit()
    conn.close()

def get_action(user, gesture):
    """Retrieves the mapped action for a specific user and gesture."""
    conn = sqlite3.connect('flowsync.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT action_mapped FROM UserSettings 
        WHERE user_name=? AND gesture_name=?
    ''', (user, gesture))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0] # Returns the action string (e.g., 'Left Click')
    
    return None