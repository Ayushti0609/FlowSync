import tkinter as tk
from tkinter import messagebox

# --- Colors ---
ACCENT = "#0C447C"
ACCENT_LIGHT = "#E6F1FB"
BG = "#F1EFE8"
CARD_BG = "#FFFFFF"
TEXT_SECONDARY = "#5F5E5A"
SUCCESS_BG = "#EAF3DE"
SUCCESS_TEXT = "#3B6D11"

root = tk.Tk()
root.title("FlowSync")
root.geometry("380x520")
root.configure(bg=BG)
root.resizable(False, False)

container = tk.Frame(root, bg=BG)
container.place(relx=0.5, rely=0.5, anchor="center", width=320, height=460)

welcome_frame = tk.Frame(container, bg=CARD_BG)
calibration_frame = tk.Frame(container, bg=CARD_BG)
main_frame = tk.Frame(container, bg=CARD_BG)

for frame in (welcome_frame, calibration_frame, main_frame):
    frame.place(relx=0, rely=0, relwidth=1, relheight=1)

def show_frame(frame):
    frame.tkraise()

# =========================================================
# SCREEN 1: Welcome
# =========================================================
inner1 = tk.Frame(welcome_frame, bg=CARD_BG, padx=30, pady=30)
inner1.pack(fill="both", expand=True)

icon_box = tk.Frame(inner1, bg=ACCENT_LIGHT, width=48, height=48)
icon_box.pack(pady=(0, 12))
icon_box.pack_propagate(False)
tk.Label(icon_box, text="~", bg=ACCENT_LIGHT, fg=ACCENT, font=("Arial", 20, "bold")).pack(expand=True)

tk.Label(inner1, text="FlowSync", bg=CARD_BG, font=("Arial", 20, "bold")).pack()
tk.Label(inner1, text="Gesture-based cursor control", bg=CARD_BG, fg=TEXT_SECONDARY, font=("Arial", 10)).pack(pady=(0, 24))

tk.Label(inner1, text="Your name", bg=CARD_BG, font=("Arial", 10), anchor="w").pack(fill="x")
name_entry = tk.Entry(inner1, font=("Arial", 12), relief="solid", bd=1)
name_entry.pack(fill="x", pady=(4, 20), ipady=6)

def go_to_calibration():
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("FlowSync", "Please enter your name first.")
        return
    user_name_label.config(text=f"Hi {name}, let's calibrate")
    profile_value_label.config(text=name)   # naam Main screen ke stat card mein bhi dikhega
    show_frame(calibration_frame)

tk.Button(inner1, text="Create new profile", command=go_to_calibration,
          bg=ACCENT, fg="white", font=("Arial", 11, "bold"), relief="flat", cursor="hand2"
          ).pack(fill="x", ipady=8, pady=(0, 8))

tk.Button(inner1, text="Load existing profile",
          bg=CARD_BG, fg="black", font=("Arial", 11), relief="solid", bd=1, cursor="hand2"
          ).pack(fill="x", ipady=8)

# =========================================================
# SCREEN 2: Calibration
# =========================================================
inner2 = tk.Frame(calibration_frame, bg=CARD_BG, padx=30, pady=30)
inner2.pack(fill="both", expand=True)

user_name_label = tk.Label(inner2, text="Let's calibrate", bg=CARD_BG, font=("Arial", 15, "bold"))
user_name_label.pack(anchor="w")

step_label = tk.Label(inner2, text="1 / 4", bg=CARD_BG, fg=TEXT_SECONDARY, font=("Arial", 10))
step_label.pack(anchor="e", pady=(0, 10))

instruction_label = tk.Label(
    inner2, text="Move your hand to the top-left corner and hold.",
    bg=CARD_BG, fg=TEXT_SECONDARY, font=("Arial", 10), wraplength=260, justify="left"
)
instruction_label.pack(anchor="w", pady=(0, 16))

preview_box2 = tk.Frame(inner2, bg=BG, height=140)
preview_box2.pack(fill="x", pady=(0, 20))
tk.Label(preview_box2, text="camera preview yahan aayega", bg=BG, fg=TEXT_SECONDARY, font=("Arial", 9)).pack(expand=True)

corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
current_step = {"index": 0}

def go_to_main():
    show_frame(main_frame)

def capture_corner():
    idx = current_step["index"]
    print(f"Corner captured: {corners[idx]}")

    idx += 1
    if idx >= len(corners):
        current_step["index"] = 0
        step_label.config(text="1 / 4")
        instruction_label.config(text="Move your hand to the top-left corner and hold.")
        messagebox.showinfo("FlowSync", "Calibration complete!")
        go_to_main()
        return

    current_step["index"] = idx
    step_label.config(text=f"{idx + 1} / 4")
    instruction_label.config(text=f"Move your hand to the {corners[idx]} corner and hold.")

tk.Button(inner2, text="Hold to capture", command=capture_corner,
          bg=ACCENT, fg="white", font=("Arial", 11, "bold"), relief="flat", cursor="hand2"
          ).pack(fill="x", ipady=8)

# =========================================================
# SCREEN 3: Main control
# =========================================================
inner3 = tk.Frame(main_frame, bg=CARD_BG, padx=25, pady=25)
inner3.pack(fill="both", expand=True)

# --- Header row: logo + status badge ---
header_row = tk.Frame(inner3, bg=CARD_BG)
header_row.pack(fill="x", pady=(0, 16))

logo_box = tk.Frame(header_row, bg=ACCENT_LIGHT, width=28, height=28)
logo_box.pack(side="left")
logo_box.pack_propagate(False)
tk.Label(logo_box, text="~", bg=ACCENT_LIGHT, fg=ACCENT, font=("Arial", 12, "bold")).pack(expand=True)

tk.Label(header_row, text=" FlowSync", bg=CARD_BG, font=("Arial", 13, "bold")).pack(side="left")

status_badge = tk.Label(header_row, text="● Idle", bg="#F1EFE8", fg=TEXT_SECONDARY,
                         font=("Arial", 9, "bold"), padx=10, pady=3)
status_badge.pack(side="right")

# --- Webcam preview placeholder ---
preview_box3 = tk.Frame(inner3, bg=BG, height=130)
preview_box3.pack(fill="x", pady=(0, 14))
tk.Label(preview_box3, text="webcam preview", bg=BG, fg=TEXT_SECONDARY, font=("Arial", 9)).pack(expand=True)

# --- Stat cards row ---
stats_row = tk.Frame(inner3, bg=CARD_BG)
stats_row.pack(fill="x", pady=(0, 16))

def make_stat_card(parent, label_text, value_text):
    card = tk.Frame(parent, bg=BG, padx=8, pady=8)
    tk.Label(card, text=label_text, bg=BG, fg=TEXT_SECONDARY, font=("Arial", 8)).pack()
    value_label = tk.Label(card, text=value_text, bg=BG, font=("Arial", 11, "bold"))
    value_label.pack()
    return card, value_label

speed_card, speed_value_label = make_stat_card(stats_row, "Speed", "Medium")
speed_card.pack(side="left", expand=True, fill="x", padx=(0, 4))

profile_card, profile_value_label = make_stat_card(stats_row, "Profile", "-")
profile_card.pack(side="left", expand=True, fill="x", padx=4)

fps_card, fps_value_label = make_stat_card(stats_row, "FPS", "0")
fps_card.pack(side="left", expand=True, fill="x", padx=(4, 0))

# --- Start/Stop/Settings buttons ---
btn_row = tk.Frame(inner3, bg=CARD_BG)
btn_row.pack(fill="x")

tracking_state = {"active": False}

def toggle_tracking():
    tracking_state["active"] = not tracking_state["active"]
    if tracking_state["active"]:
        status_badge.config(text="● Active", bg=SUCCESS_BG, fg=SUCCESS_TEXT)
        toggle_btn.config(text="Stop", bg="#E24B4A")
        print("Tracking started - yahan hand-tracking module call hoga")
    else:
        status_badge.config(text="● Idle", bg="#F1EFE8", fg=TEXT_SECONDARY)
        toggle_btn.config(text="Start", bg=ACCENT)
        print("Tracking stopped")

def open_settings():
    messagebox.showinfo("FlowSync", "Settings screen abhi baaki hai.")

toggle_btn = tk.Button(btn_row, text="Start", command=toggle_tracking,
                        bg=ACCENT, fg="white", font=("Arial", 11, "bold"), relief="flat", cursor="hand2")
toggle_btn.pack(side="left", expand=True, fill="x", ipady=8, padx=(0, 6))

tk.Button(btn_row, text="Settings", command=open_settings,
          bg=CARD_BG, fg="black", font=("Arial", 11), relief="solid", bd=1, cursor="hand2"
          ).pack(side="left", expand=True, fill="x", ipady=8, padx=(6, 0))

# --- Shuru mein Welcome screen dikhao ---
show_frame(welcome_frame)

root.mainloop()