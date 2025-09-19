GestureStar: Cosmic Command
Project Description
GestureStar: Cosmic Command is an immersive space shooter game where you pilot a spaceship using hand gestures and head movements, powered by MediaPipe and Pygame. Dodge asteroids, unleash powerful weapons like lasers and force waves, and monitor your ship’s status via a sleek dashboard with a dynamic steering gauge. Experience intuitive controls through real-time gesture detection and head-tilt navigation in a thrilling cosmic adventure.
Features

Gesture-Based Controls: Use hand gestures (Point, Open Palm, Fist, Peace Sign, Rock Sign, OK Sign) to fire weapons like lasers, twin cannons, or shields.
Head-Tilt Steering: Control your spaceship’s direction by tilting your head, visualized through an interactive circular gauge.
Real-Time Camera Input: Leverages MediaPipe for accurate hand and face tracking via webcam.
Dynamic Dashboard: Displays ship health, score, lives, steering status, throttle, boost, and weapon cooldowns.
Engaging Gameplay: Battle asteroids, manage resources, and survive with strategic gesture combinations.
Sci-Fi Aesthetic: Features a starry background, particle effects, and a futuristic dashboard with semi-transparent panels.

Prerequisites

Python 3.8+ (tested with Python 3.12)
Webcam for gesture and head tracking
Dependencies:pip install pygame opencv-python mediapipe numpy


A compatible operating system (Windows, macOS, or Linux)
Optional: A terminal other than PowerShell (e.g., Command Prompt) to avoid potential console buffer errors (e.g., System.ArgumentOutOfRangeException)

Installation

Clone the Repository:git clone https://github.com/your-username/gesturestar-cosmic-command.git
cd gesturestar-cosmic-command


Install Dependencies:pip install -r requirements.txt

Or manually:pip install pygame opencv-python mediapipe numpy


Download the Game File:Ensure the main game file (game.py) is in the repository root. This file contains the complete game code, including gesture detection and rendering logic.
Optional Heart Icon:The game attempts to load a heart_icon.png for the lives display. If not found, it defaults to a drawn heart. To use a custom icon, place a 35x35 PNG file named heart_icon.png in the same directory as game.py.

How to Play

Run the Game:python game.py


Calibration:
On startup, the game calibrates head position. Keep your head centered and still until the calibration bar fills (takes about 2 seconds).


Controls:
Steering: Tilt your head left or right to steer the spaceship (visualized on the dashboard’s steering gauge).
Throttle: Lean your head forward to accelerate or backward to decelerate.
Boost: Open your mouth to activate a speed boost (shown as “🚀 ACTIVE” on the dashboard).
Weapons:
👆 Point: Fire a precision laser (0.3s cooldown).
🖐️ Open Palm: Unleash a radial force wave (1.2s cooldown).
✊ Fist: Hold to charge a powerful shot, release to fire.
✌️ Peace Sign: Fire twin cannons (0.8s cooldown).
🤘 Rock Sign: Activate rapid-fire bullets.
👌 OK Sign: Deploy a protective shield (3.0s cooldown).


Quit: Press ESC to exit.
Restart: Press R to reset after a game over.


Objective:
Destroy asteroids to earn points while avoiding collisions.
Monitor your ship’s health (shield) and lives on the dashboard.
Survive as long as possible to maximize your score.



Dashboard Overview
The dashboard (right side of the screen) provides real-time feedback:

Camera Feed: Shows your webcam input for gesture tracking.
Shield: Displays ship health as a percentage.
Score & Lives: Tracks your score and remaining lives (heart icons).
Ship Systems:
Steering: Circular gauge with a needle showing head tilt (LEFT, CENTER, RIGHT).
Throttle: Indicates ACCELERATE, NEUTRAL, or DECELERATE based on head lean.
Boost: Shows when the mouth-activated boost is active.


Weapons Array: Lists all gestures with their weapons, highlighting the active gesture and showing cooldown/charge bars.

Troubleshooting

No Camera Feed: Ensure your webcam is connected and accessible. The game scans for cameras (indices 0-2) and sets 640x480 resolution.
Game Crashes: Check for errors in the terminal. Common issues:
Missing dependencies: Re-run pip install commands.
Console errors (e.g., System.ArgumentOutOfRangeException): Try running in Command Prompt instead of PowerShell to avoid terminal buffer issues.


Dashboard Not Displaying: Ensure you’re using the latest game.py with the fixed border_radius issue in the steering gauge.
Performance: Lower your screen resolution or reduce the number of asteroids (self.asteroids limit in spawn_asteroids) if lag occurs.

Notes

The game uses MediaPipe for robust gesture and face tracking, requiring a decent CPU/GPU for smooth performance.
The heart_icon.png is optional; the game will render a fallback heart if not provided.
If you encounter MediaPipe warnings (e.g., Feedback manager requires a single signature inference), these are benign and do not affect gameplay.

Contributing
Feel free to fork the repository, submit pull requests, or open issues for bugs or feature suggestions. Ideas for new gestures, weapons, or visual effects are welcome!
License
This project is licensed under the MIT License. See the LICENSE file for details.
