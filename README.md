
# 🌌 GestureStar: Cosmic Command

Pilot a spaceship using **hand gestures** and **head movements**!
Dodge asteroids, fire weapons, and manage your ship via a futuristic dashboard – powered by **MediaPipe** + **Pygame**.

---

## ⚡ TL;DR (Quick Start)

```bash
# Clone repo


# Install dependencies
pip install -r requirements.txt

# Run game
python game.py
```

👉 Requires **Python 3.8+**, **webcam**, and `pygame opencv-python mediapipe numpy`.

---

## 🎮 Controls

* **Steer** → Tilt head (left/right)
* **Throttle** → Lean forward/backward
* **Boost** → Open mouth
* **Weapons** →

  * 👆 Point → Laser
  * 🖐️ Palm → Force wave
  * ✊ Fist → Charge shot
  * ✌️ Peace → Twin cannons
  * 🤘 Rock → Rapid fire
  * 👌 OK → Shield

`ESC` = Quit • `R` = Restart

---

## 📊 Dashboard

* Camera feed (gesture tracking)
* Health (shield %) & lives ❤️
* Score tracker
* Steering gauge + throttle + boost
* Weapon cooldowns

---

## 🛠️ Troubleshooting

* **No camera** → Check webcam (indices 0–2).
* **Console crash** (PowerShell bug) → Use **Command Prompt** instead.
* **Laggy** → Reduce asteroid count (`spawn_asteroids`).

---

## 🤝 Contribute

Ideas for new gestures, weapons, or effects?

* Fork, PR, or open an issue 🚀

---

## 📜 License

MIT – see [LICENSE](LICENSE).

---

