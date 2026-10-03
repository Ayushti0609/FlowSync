import pyautogui

# Automatically get the screen size
screen_width, screen_height = pyautogui.size()

# Remove PyAutoGUI's default delay for faster mouse movement
pyautogui.PAUSE = 0 

def move_cursor(x_ratio, y_ratio):
    """
    The AI team will send X and Y as ratios between 0.0 and 1.0.
    Convert these to actual screen pixels.
    """
    actual_x = int(x_ratio * screen_width)
    actual_y = int(y_ratio * screen_height)
    
    # Move the mouse to the calculated coordinates
    pyautogui.moveTo(actual_x, actual_y)

# ---------------------------------------------------------
# CLICK & SCROLL FUNCTIONS
# ---------------------------------------------------------

def left_click():
    """Performs a standard left mouse click."""
    pyautogui.click()
    print("Action Executed: Left Click")

def right_click():
    """Performs a right mouse click."""
    pyautogui.click(button='right')
    print("Action Executed: Right Click")

def double_click():
    """Performs a double mouse click."""
    pyautogui.doubleClick()
    print("Action Executed: Double Click")

def scroll_up():
    """Scrolls the screen up."""
    pyautogui.scroll(300)  # 300 is the scroll speed/amount
    print("Action Executed: Scroll Up")

def scroll_down():
    """Scrolls the screen down."""
    pyautogui.scroll(-300)
    print("Action Executed: Scroll Down")

def take_screenshot():
    """Takes a screenshot and saves it directly (Windows + PrintScreen)."""
    pyautogui.hotkey('win', 'printscreen')
    print("Action Executed: Take Screenshot")

# ---------------------------------------------------------
# MEDIA FUNCTIONS
# ---------------------------------------------------------

def volume_up():
    """Increases the system volume."""
    pyautogui.press('volumeup')
    print("Action Executed: Volume Up")

def volume_down():
    """Decreases the system volume."""
    pyautogui.press('volumedown')
    print("Action Executed: Volume Down")

# ---------------------------------------------------------
# ACTION DISPATCHER
# ---------------------------------------------------------

def execute_action(action_name):
    """
    Executes the specific PyAutoGUI command based on the mapped action 
    retrieved from the SQLite database.
    """
    if action_name == 'Left Click':
        left_click()
    elif action_name == 'Right Click':
        right_click()
    elif action_name == 'Double Click':
        double_click()
    elif action_name == 'Scroll Up':
        scroll_up()
    elif action_name == 'Scroll Down':
        scroll_down()
    elif action_name == 'Take Screenshot':
        take_screenshot()
    elif action_name == 'Volume Up':
        volume_up()
    elif action_name == 'Volume Down':
        volume_down()
    elif action_name == 'None' or action_name is None:
        pass # Do nothing if no action is mapped
    else:
        print(f"Unknown Action: {action_name}")