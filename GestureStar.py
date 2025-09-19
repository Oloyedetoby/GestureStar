import cv2
import mediapipe as mp
import pygame
import math
import random
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

# --- CORE DATA STRUCTURES (From your original code) ---

class GestureType(Enum):
    """Enumeration of supported hand gestures."""
    POINT = "point"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PEACE_SIGN = "peace_sign"
    ROCK_SIGN = "rock_sign"
    OK_SIGN = "ok_sign"
    SNAP = "snap"
    NONE = "none"

@dataclass
class HandGesture:
    """Data class for hand gesture detection results."""
    gesture: GestureType
    confidence: float
    position: Tuple[float, float]
    landmarks: Optional[List]

@dataclass
class FacialState:
    """Data class for facial expression states."""
    mouth_open: bool
    mouth_open_ratio: float
    head_pose: Tuple[float, float, float, float]  # x, y, z, tilt
    is_calibrated: bool

# --- DETECTION SYSTEM (Your original, reliable class) ---

class GestureDetector:
    """Professional gesture detection system using MediaPipe."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_hand_gesture(self, landmarks) -> GestureType:
        if not landmarks:
            return GestureType.NONE
        
        thumb_tip, index_tip = landmarks[4], landmarks[8]
        thumb_mcp, index_pip = landmarks[2], landmarks[6]
        middle_tip, middle_pip = landmarks[12], landmarks[10]
        ring_tip, ring_pip = landmarks[16], landmarks[14]
        pinky_tip, pinky_pip = landmarks[20], landmarks[18]
        
        def is_finger_extended(tip, pip): return tip.y < pip.y
        
        fingers_up = [
            thumb_tip.x > thumb_mcp.x, # Simple check for right hand
            is_finger_extended(index_tip, index_pip),
            is_finger_extended(middle_tip, middle_pip),
            is_finger_extended(ring_tip, ring_pip),
            is_finger_extended(pinky_tip, pinky_pip)
        ]
        total_fingers = sum(fingers_up)
        
        if total_fingers == 1 and fingers_up[1]: return GestureType.POINT
        if total_fingers == 5: return GestureType.OPEN_PALM
        if total_fingers == 0: return GestureType.FIST
        if total_fingers == 2 and fingers_up[1] and fingers_up[2]: return GestureType.PEACE_SIGN
        if total_fingers == 2 and fingers_up[1] and fingers_up[4]: return GestureType.ROCK_SIGN
        if math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) < 0.05: return GestureType.OK_SIGN
        
        return GestureType.NONE
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandGesture], FacialState]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        hand_gestures = []
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture_type = self.detect_hand_gesture(hand_landmarks.landmark)
                center_x = sum(lm.x for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                center_y = sum(lm.y for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                hand_gestures.append(HandGesture(
                    gesture=gesture_type, confidence=0.8,
                    position=(center_x, center_y), landmarks=hand_landmarks.landmark
                ))
        
        face_results = self.face_mesh.process(rgb_frame)
        facial_state = FacialState(
            mouth_open=False, mouth_open_ratio=0.0,
            head_pose=(0.5, 0.5, 0.0, 0.0), is_calibrated=False
        )
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            upper_lip, lower_lip = face_landmarks.landmark[13], face_landmarks.landmark[14]
            mouth_height = abs(upper_lip.y - lower_lip.y)
            facial_state.mouth_open = mouth_height > 0.015
            facial_state.mouth_open_ratio = min(mouth_height / 0.03, 1.0)
            
            nose_tip, chin = face_landmarks.landmark[1], face_landmarks.landmark[175]
            left_eye, right_eye = face_landmarks.landmark[33], face_landmarks.landmark[362]
            head_center_x = (left_eye.x + right_eye.x) / 2
            head_center_y = (nose_tip.y + chin.y) / 2
            eye_angle = math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
            facial_state.head_pose = (head_center_x, head_center_y, nose_tip.z, math.degrees(eye_angle))
        
        return hand_gestures, facial_state

# --- GAME SYSTEMS ---

class WeaponSystem:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width, self.screen_height = screen_width, screen_height
        self.projectiles, self.shield_particles = [], []
        self.weapon_cooldowns = {gesture: 0.0 for gesture in GestureType}
        self.charge_start_time, self.max_charge_time = None, 2.0
    
    def update(self, dt: float):
        for weapon in self.weapon_cooldowns:
            if self.weapon_cooldowns[weapon] > 0: self.weapon_cooldowns[weapon] -= dt
        for p in self.projectiles[:]:
            p['x'] += p['vel_x'] * dt; p['y'] += p['vel_y'] * dt; p['life'] -= dt
            if p['life'] <= 0: self.projectiles.remove(p)
        for p in self.shield_particles[:]:
            p['life'] -= 1
            if p['life'] <= 0: self.shield_particles.remove(p)
    
    def can_fire_weapon(self, weapon_type: GestureType): return self.weapon_cooldowns.get(weapon_type, 0) <= 0
    
    def fire_precision_laser(self, ship_pos, target_pos):
        if not self.can_fire_weapon(GestureType.POINT): return
        dx, dy = target_pos[0]-ship_pos[0], target_pos[1]-ship_pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            dx, dy = dx/dist, dy/dist
            self.projectiles.append({'type':'laser','x':ship_pos[0],'y':ship_pos[1],'vel_x':dx*800,'vel_y':dy*800,'life':1.0,'damage':25,'color':(255,255,100)})
            self.weapon_cooldowns[GestureType.POINT] = 0.3
    
    def fire_twin_cannons(self, ship_pos, ship_angle):
        if not self.can_fire_weapon(GestureType.PEACE_SIGN): return
        angle_rad = math.radians(ship_angle-90)
        for offset in [-15, 15]:
            offset_angle = math.radians(ship_angle+offset-90)
            start_x, start_y = ship_pos[0]+math.cos(offset_angle)*20, ship_pos[1]+math.sin(offset_angle)*20
            self.projectiles.append({'type':'missile','x':start_x,'y':start_y,'vel_x':math.cos(angle_rad)*600,'vel_y':math.sin(angle_rad)*600,'life':1.5,'damage':35,'color':(255,100,255)})
        self.weapon_cooldowns[GestureType.PEACE_SIGN] = 0.8
    
    def activate_rapid_fire(self, ship_pos, ship_angle):
        for _ in range(3):
            spread_rad = math.radians(ship_angle+random.uniform(-10,10)-90)
            self.projectiles.append({'type':'bullet','x':ship_pos[0],'y':ship_pos[1],'vel_x':math.cos(spread_rad)*700,'vel_y':math.sin(spread_rad)*700,'life':1.0,'damage':15,'color':(255,200,0)})
    
    def create_force_wave(self, ship_pos):
        if not self.can_fire_weapon(GestureType.OPEN_PALM): return
        for angle in range(0,360,15):
            angle_rad = math.radians(angle)
            self.projectiles.append({'type':'wave','x':ship_pos[0],'y':ship_pos[1],'vel_x':math.cos(angle_rad)*400,'vel_y':math.sin(angle_rad)*400,'life':0.8,'damage':20,'color':(100,255,200)})
        self.weapon_cooldowns[GestureType.OPEN_PALM] = 1.2
    
    def charge_fist_weapon(self, ship_pos, is_charging):
        if is_charging:
            if self.charge_start_time is None: self.charge_start_time=time.time()
        elif self.charge_start_time is not None:
            charge_ratio = min((time.time()-self.charge_start_time)/self.max_charge_time, 1.0)
            self.fire_charged_shot(ship_pos, charge_ratio)
            self.charge_start_time = None
    
    def fire_charged_shot(self, ship_pos, charge_ratio):
        damage, size, speed = 50+charge_ratio*100, 10+charge_ratio*20, 500+charge_ratio*300
        for _ in range(1+int(charge_ratio*4)):
            angle_rad = math.radians(random.uniform(-30,30)-90)
            self.projectiles.append({'type':'charged','x':ship_pos[0],'y':ship_pos[1],'vel_x':math.cos(angle_rad)*speed,'vel_y':math.sin(angle_rad)*speed,'life':2.0,'damage':damage,'size':size,'color':(255,int(100+charge_ratio*155),100)})
    
    def activate_shield(self, ship_pos):
        if not self.can_fire_weapon(GestureType.OK_SIGN): return
        for angle in range(0,360,10):
            angle_rad = math.radians(angle)
            self.shield_particles.append({'x':ship_pos[0]+math.cos(angle_rad)*40,'y':ship_pos[1]+math.sin(angle_rad)*40,'life':180,'color':(100,255,100)})
        self.weapon_cooldowns[GestureType.OK_SIGN] = 3.0

class EffectManager:
    def __init__(self):
        self.particles, self.shockwaves = [], []
        self.screen_shake_duration, self.screen_shake_intensity = 0.0, 0
    def trigger_explosion(self, x, y, size):
        for _ in range(int(size/2)):
            angle, speed = random.uniform(0, 2*math.pi), random.uniform(50,200)
            self.particles.append({'x':x,'y':y,'vel_x':math.cos(angle)*speed,'vel_y':math.sin(angle)*speed,'life':random.uniform(0.5,1.2),'color':random.choice([(255,255,100),(255,150,0),(200,200,200)]),'size':random.randint(2,5)})
        self.shockwaves.append({'x':x,'y':y,'radius':10,'max_radius':size*2.5,'life':0.5,'max_life':0.5,'width':10})
        self.screen_shake_duration, self.screen_shake_intensity = 0.3, int(size/5)
    def trigger_player_hit(self): self.screen_shake_duration, self.screen_shake_intensity = 0.5, 10
    def update(self, dt):
        for p in self.particles[:]:
            p['x']+=p['vel_x']*dt; p['y']+=p['vel_y']*dt; p['life']-=dt
            if p['life']<=0: self.particles.remove(p)
        for sw in self.shockwaves[:]:
            sw['radius']+=sw['max_radius']/sw['max_life']*dt; sw['life']-=dt
            if sw['life']<=0: self.shockwaves.remove(sw)
        if self.screen_shake_duration>0: self.screen_shake_duration-=dt
        else: self.screen_shake_intensity=0
    def get_shake_offset(self):
        if self.screen_shake_intensity>0: return(random.randint(-self.screen_shake_intensity,self.screen_shake_intensity),random.randint(-self.screen_shake_intensity,self.screen_shake_intensity))
        return(0,0)
    def render(self, surface):
        for p in self.particles:
            alpha = max(0,min(255,int(p['life']/1.2*255))); s=pygame.Surface((p['size']*2,p['size']*2),pygame.SRCALPHA); pygame.draw.circle(s, p['color']+(alpha,),(p['size'],p['size']),p['size']); surface.blit(s, (int(p['x']-p['size']), int(p['y']-p['size'])))
        for sw in self.shockwaves:
            alpha = max(0,min(255,int(sw['life']/sw['max_life']*255))); width = int(sw['width']*sw['life']/sw['max_life'])
            if width>0: s=pygame.Surface((sw['radius']*2,sw['radius']*2),pygame.SRCALPHA); pygame.draw.circle(s, (255,255,255,alpha), (int(sw['radius']),int(sw['radius'])), int(sw['radius']), width); surface.blit(s, (int(sw['x']-sw['radius']),int(sw['y']-sw['radius'])))

class SpaceshipController:
    def __init__(self, game_width: int, game_height: int):
        self.game_width, self.game_height = game_width, game_height
        self.max_health, self.max_lives = 100, 3
        self.reset(full_reset=True)
        self.head_center, self.is_calibrated = [0.5,0.5,0.0], False
        self.calibration_samples, self.movement_history, self.max_history = [], [], 5
        # ### NEW: Attributes to store control status for the dashboard ###
        self.steering_status = "CENTER"
        self.speed_status = "NEUTRAL"
    def reset(self, full_reset=True):
        self.position = [self.game_width//2, self.game_height//2]
        self.velocity, self.angle, self.boost_active = [0.0,0.0], 0.0, False
        self.health = self.max_health
        self.invulnerability_timer = 3.0
        if full_reset: self.lives = self.max_lives
    def lose_life(self):
        self.lives -= 1
        self.reset(full_reset=False)
        return self.lives > 0
    def take_damage(self, amount):
        if self.invulnerability_timer<=0:
            self.health -= amount
            self.invulnerability_timer = 2.0
            return "damaged" if self.health > 0 else "lost_life"
        return "invulnerable"
    def update(self, dt):
        if self.invulnerability_timer>0: self.invulnerability_timer -= dt
    def calibrate_head_center(self, head_pose):
        self.calibration_samples.append([head_pose[0], head_pose[1], head_pose[2]])
        if len(self.calibration_samples) >= 60:
            self.head_center = np.median(np.array(self.calibration_samples), axis=0).tolist()
            self.is_calibrated = True
    def update_movement(self, head_pose, mouth_boost, dt):
        if not self.is_calibrated: return
        head_dx, head_dy = (head_pose[0]-self.head_center[0])*6, (head_pose[1]-self.head_center[1])*4
        head_dz = (head_pose[2]-self.head_center[2])*3
        
        # ### NEW: Update dashboard status ###
        tilt_angle = head_pose[3]
        if tilt_angle > 7: self.steering_status = "RIGHT"
        elif tilt_angle < -7: self.steering_status = "LEFT"
        else: self.steering_status = "CENTER"

        if head_dz > 0.3: self.speed_status = "DECELERATE"
        elif head_dz < -0.3: self.speed_status = "ACCELERATE"
        else: self.speed_status = "NEUTRAL"
        
        self.movement_history.append([head_dx, head_dy, head_dz])
        if len(self.movement_history)>self.max_history: self.movement_history.pop(0)
        smooth_dx, smooth_dy, smooth_dz = np.mean(self.movement_history, axis=0) if len(self.movement_history)>=3 else (head_dx,head_dy,head_dz)
        
        boost = 1.0 + mouth_boost*1.5; self.boost_active = mouth_boost>0.3
        self.velocity[0]+=smooth_dx*400*boost*dt; self.velocity[1]+=smooth_dy*400*boost*dt
        speed_mod = max(0.5, min(2.0, 1.0 - smooth_dz*0.5)) # Inverted dz for intuitive lean
        drag = 0.92 if self.boost_active else 0.88; self.velocity[0]*=drag; self.velocity[1]*=drag
        self.position[0]+=self.velocity[0]*speed_mod*dt; self.position[1]+=self.velocity[1]*speed_mod*dt
        self.angle += (tilt_angle*0.8 - self.angle)*0.15
        self.position[0] = max(50, min(self.game_width-50, self.position[0]))
        self.position[1] = max(50, min(self.game_height-50, self.position[1]))

class CosmicVoyageGame:
    GESTURE_EMOJIS = {
        GestureType.POINT: "👆", GestureType.OPEN_PALM: "🖐️",
        GestureType.FIST: "✊", GestureType.PEACE_SIGN: "✌️",
        GestureType.ROCK_SIGN: "🤘", GestureType.OK_SIGN: "👌",
        GestureType.SNAP: "🫰", GestureType.NONE: ""
    }
    WEAPON_DESCRIPTIONS = {
        GestureType.POINT: "Laser", GestureType.OPEN_PALM: "Force Wave",
        GestureType.FIST: "Charge Shot", GestureType.PEACE_SIGN: "Twin Cannons",
        GestureType.ROCK_SIGN: "Rapid Fire", GestureType.OK_SIGN: "Shield",
        GestureType.SNAP: "Explosion"
    }
    COOLDOWN_DURATIONS = {
        GestureType.POINT: 0.3,
        GestureType.OPEN_PALM: 1.2,
        GestureType.FIST: 0.0,  # No cooldown, as it's charge-based
        GestureType.PEACE_SIGN: 0.8,
        GestureType.ROCK_SIGN: 0.0,  # No cooldown, as it's instant
        GestureType.OK_SIGN: 3.0,
        GestureType.SNAP: 0.0
    }

    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h
        self.dashboard_width = int(self.screen_width * 0.22)
        self.game_width = self.screen_width - self.dashboard_width
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN | pygame.SCALED)
        
        self.game_surface = pygame.Surface((self.game_width, self.screen_height))
        pygame.display.set_caption("Cosmic Voyage: Gesture Control")
        self.clock = pygame.time.Clock()
        
        self.cap = self.initialize_camera()
        if not self.cap: raise RuntimeError("Failed to initialize camera")
        
        self.gesture_detector = GestureDetector()
        self.weapon_system = WeaponSystem(self.game_width, self.screen_height)
        self.spaceship = SpaceshipController(self.game_width, self.screen_height)
        self.effect_manager = EffectManager()
        
        self.running, self.game_over, self.score = True, False, 0
        self.stars = self.generate_starfield(400)
        self.asteroids = []
        
        self.flash_message, self.flash_message_timer = None, 0.0
        
        self.font_sm = pygame.font.Font(None, 28)
        self.font_md = pygame.font.Font(None, 42)
        self.font_lg = pygame.font.Font(None, 80)
        try:
            self.font_emoji = pygame.font.SysFont("Segoe UI Emoji", 32)
        except pygame.error:
            self.font_emoji = pygame.font.Font(None, 40)
        # NEW: Smaller font for gesture text to save space
        self.font_xs = pygame.font.Font(None, 24)

        try:
            self.heart_icon = pygame.transform.scale(pygame.image.load('heart_icon.png').convert_alpha(),(35,35))
        except FileNotFoundError:
            self.heart_icon=pygame.Surface((35,35),pygame.SRCALPHA); pygame.draw.polygon(self.heart_icon,(255,20,20),[(17,35),(0,12),(17,20),(35,12)])

    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        print("🎥 Scanning for cameras...")
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            for index in range(3):
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.mean() > 10:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
                            print(f"✅ Camera {index} with backend {backend} initialized successfully!")
                            return cap
                    cap.release()
                except Exception: continue
        print("❌ Error: No suitable camera found.")
        return None

    def reset_game(self):
        self.score, self.game_over = 0, False
        self.asteroids.clear(); self.weapon_system.projectiles.clear()
        self.effect_manager.particles.clear(); self.effect_manager.shockwaves.clear()
        self.spaceship.reset(full_reset=True)
    
    def generate_starfield(self, count):
        return [{'x':random.randint(0,self.game_width),'y':random.randint(0,self.screen_height),'speed':random.uniform(0.5,2.0)} for _ in range(count)]
    
    def spawn_asteroids(self):
        if len(self.asteroids) > 25 or random.randint(0, 40) != 0: return
        size=random.randint(30,70); side=random.choice(['top','bottom','left','right'])
        if side=='top': x,y,vx,vy=random.randint(0,self.game_width),-70,random.uniform(-1,1),random.uniform(1,3)
        elif side=='bottom': x,y,vx,vy=random.randint(0,self.game_width),self.screen_height+70,random.uniform(-1,1),random.uniform(-3,-1)
        elif side=='left': x,y,vx,vy=-70,random.randint(0,self.screen_height),random.uniform(1,3),random.uniform(-1,1)
        else: x,y,vx,vy=self.game_width+70,random.randint(0,self.screen_height),random.uniform(-3,-1),random.uniform(-1,1)
        shape=[( (size+random.uniform(-size*0.3,size*0.3))*math.cos(i/12*2*math.pi), (size+random.uniform(-size*0.3,size*0.3))*math.sin(i/12*2*math.pi) ) for i in range(12)]
        self.asteroids.append({'x':x,'y':y,'vel_x':vx*30,'vel_y':vy*30,'size':size,'rotation':0,'rotation_speed':random.uniform(-50,50),'health':size*2,'points':int(80-size),'hit_timer':0,'shape':shape})
    
    def handle_collisions(self):
        for proj in self.weapon_system.projectiles[:]:
            for asteroid in self.asteroids[:]:
                if math.hypot(proj['x']-asteroid['x'], proj['y']-asteroid['y']) < asteroid['size']:
                    asteroid['health']-=proj['damage']; asteroid['hit_timer']=5
                    if proj in self.weapon_system.projectiles: self.weapon_system.projectiles.remove(proj)
                    if asteroid['health']<=0:
                        self.score+=asteroid['points']; self.effect_manager.trigger_explosion(asteroid['x'],asteroid['y'],asteroid['size'])
                        if asteroid in self.asteroids: self.asteroids.remove(asteroid)
                    break
        for asteroid in self.asteroids[:]:
            if math.hypot(self.spaceship.position[0]-asteroid['x'], self.spaceship.position[1]-asteroid['y']) < asteroid['size']+15:
                damage_result = self.spaceship.take_damage(int(asteroid['size']/2))
                if damage_result != "invulnerable":
                    self.effect_manager.trigger_player_hit(); self.effect_manager.trigger_explosion(asteroid['x'],asteroid['y'],asteroid['size'])
                    if asteroid in self.asteroids: self.asteroids.remove(asteroid)
                    if damage_result=="lost_life":
                        if self.spaceship.lose_life():
                            lives = self.spaceship.lives; plural = "S" if lives!=1 else ""; self.trigger_flash_message(f"{lives} LIFE{plural} REMAINING!")
                        else: self.game_over=True
                    break
    
    def process_gestures(self, hand_gestures):
        ship_pos, ship_angle = tuple(self.spaceship.position), self.spaceship.angle
        is_fist_charging = False
        for hand in hand_gestures:
            gesture = hand.gesture
            if gesture == GestureType.POINT: self.weapon_system.fire_precision_laser(ship_pos, (hand.position[0]*self.game_width, hand.position[1]*self.screen_height))
            elif gesture == GestureType.PEACE_SIGN: self.weapon_system.fire_twin_cannons(ship_pos, ship_angle)
            elif gesture == GestureType.ROCK_SIGN: self.weapon_system.activate_rapid_fire(ship_pos, ship_angle)
            elif gesture == GestureType.OPEN_PALM: self.weapon_system.create_force_wave(ship_pos)
            elif gesture == GestureType.FIST: is_fist_charging = True
            elif gesture == GestureType.OK_SIGN: self.weapon_system.activate_shield(ship_pos)
        self.weapon_system.charge_fist_weapon(ship_pos, is_fist_charging)
    
    def update_game_objects(self, dt):
        if self.flash_message_timer > 0: self.flash_message_timer -= dt
        else: self.flash_message = None
        self.spaceship.update(dt); self.effect_manager.update(dt)
        for star in self.stars: star['y'] = (star['y']+star['speed']*50*dt)%self.screen_height
        for asteroid in self.asteroids:
            asteroid['x']+=asteroid['vel_x']*dt; asteroid['y']+=asteroid['vel_y']*dt; asteroid['rotation']=(asteroid['rotation']+asteroid['rotation_speed']*dt)%360
            if asteroid['hit_timer']>0: asteroid['hit_timer']-=1
        self.spawn_asteroids(); self.weapon_system.update(dt)
    
    def run(self):
        while self.running:
            dt = self.clock.tick(60)/1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.running = False
                    if event.key == pygame.K_r: self.reset_game()
            
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            hand_gestures, facial_state = self.gesture_detector.process_frame(frame)
            
            if not self.game_over:
                if not self.spaceship.is_calibrated:
                    self.spaceship.calibrate_head_center(facial_state.head_pose)
                else:
                    self.spaceship.update_movement(facial_state.head_pose, facial_state.mouth_open_ratio, dt)
                    self.process_gestures(hand_gestures)
                    self.update_game_objects(dt)
                    self.handle_collisions()
            
            self.render(frame, hand_gestures)
        self.cleanup()
    
    def trigger_flash_message(self, text, duration=3.0):
        self.flash_message = text
        self.flash_message_timer = duration

    def render(self, frame, hand_gestures):
        self.game_surface.fill((5,5,15))
        for star in self.stars: pygame.draw.circle(self.game_surface, (200,200,200), (int(star['x']),int(star['y'])), 1)
        for p in self.weapon_system.projectiles: pygame.draw.circle(self.game_surface, p['color'],(int(p['x']),int(p['y'])), p.get('size',8))
        self.effect_manager.render(self.game_surface)
        for asteroid in self.asteroids:
            angle_rad=math.radians(asteroid['rotation']); points=[(asteroid['x']+p[0]*math.cos(angle_rad)-p[1]*math.sin(angle_rad), asteroid['y']+p[0]*math.sin(angle_rad)+p[1]*math.cos(angle_rad)) for p in asteroid['shape']]
            color=(255,255,255) if asteroid['hit_timer']>0 else (150,100,70)
            pygame.draw.polygon(self.game_surface, color, points)
            if asteroid['hit_timer']==0: pygame.draw.polygon(self.game_surface, (200,150,100), points, 3)
        
        if not (self.spaceship.invulnerability_timer>0 and int(self.spaceship.invulnerability_timer*10)%2==0):
            pos, angle = self.spaceship.position, self.spaceship.angle; angle_rad = math.radians(angle)
            points = [(pos[0]+p[0]*math.cos(angle_rad)-p[1]*math.sin(angle_rad), pos[1]+p[0]*math.sin(angle_rad)+p[1]*math.cos(angle_rad)) for p in [(0,-20),(-12,15),(0,8),(12,15)]]
            pygame.draw.polygon(self.game_surface, (100,255,255) if self.spaceship.boost_active else (100,200,255), points)
            pygame.draw.polygon(self.game_surface, (255,255,255), points, 2)
        
        if self.flash_message and self.flash_message_timer > 0:
            alpha = 255 if self.flash_message_timer > 1.0 else int(self.flash_message_timer * 255)
            flash_surf = self.font_lg.render(self.flash_message, True, (255,255,100)); flash_surf.set_alpha(alpha)
            self.game_surface.blit(flash_surf, flash_surf.get_rect(center=(self.game_width/2, self.screen_height/2)))
        
        self.screen.blit(self.game_surface, self.effect_manager.get_shake_offset())
        self.render_dashboard(frame, hand_gestures)

        if self.game_over:
            s=pygame.Surface((self.screen_width,self.screen_height),pygame.SRCALPHA); s.fill((0,0,0,180)); self.screen.blit(s,(0,0))
            go_text=self.font_lg.render("GAME OVER",True,(255,50,50)); self.screen.blit(go_text,go_text.get_rect(center=(self.screen_width/2,self.screen_height/2-80)))
            score_text=self.font_md.render(f"Final Score: {self.score}",True,(255,255,100)); self.screen.blit(score_text,score_text.get_rect(center=(self.screen_width/2,self.screen_height/2+10)))
            restart_text=self.font_md.render("Press 'R' to Restart",True,(200,200,200)); self.screen.blit(restart_text,restart_text.get_rect(center=(self.screen_width/2,self.screen_height/2+70)))
        elif not self.spaceship.is_calibrated:
            self.render_calibration_screen()

        pygame.display.flip()

    def render_calibration_screen(self):
        s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA); s.fill((0,0,0,180)); self.screen.blit(s,(0,0))
        title_text = self.font_lg.render("AWAITING CALIBRATION", True, (255,255,255)); self.screen.blit(title_text, title_text.get_rect(center=(self.screen_width/2, self.screen_height/2-100)))
        instr_text = self.font_md.render("Keep Your Head Centered and Still", True, (200,200,200)); self.screen.blit(instr_text, instr_text.get_rect(center=(self.screen_width/2, self.screen_height/2-20)))
        progress = len(self.spaceship.calibration_samples) / 60.0; bar_width = self.screen_width*0.4; bar_height = 40; bar_x = self.screen_width/2-bar_width/2; bar_y = self.screen_height/2+50
        pygame.draw.rect(self.screen,(50,50,50),(bar_x,bar_y,bar_width,bar_height),border_radius=10)
        pygame.draw.rect(self.screen,(0,255,100),(bar_x,bar_y,bar_width*progress,bar_height),border_radius=10)

    def render_dashboard(self, frame, hand_gestures):
        dash_x = self.game_width
        # Darker background with a subtle gradient effect
        bg_surface = pygame.Surface((self.dashboard_width, self.screen_height), pygame.SRCALPHA)
        for i in range(self.screen_height):
            alpha = 255 - int(i / self.screen_height * 50)
            pygame.draw.line(bg_surface, (10, 20, 30, alpha), (0, i), (self.dashboard_width, i))
        self.screen.blit(bg_surface, (dash_x, 0))
        pygame.draw.line(self.screen, (100, 100, 120), (dash_x, 0), (dash_x, self.screen_height), 3)

        # Camera feed
        original_h, original_w, _ = frame.shape
        cam_aspect_ratio = original_h / original_w
        cam_display_w = self.dashboard_width - 20
        cam_display_h = int(cam_display_w * cam_aspect_ratio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        cam_surface_scaled = pygame.transform.scale(cam_surface, (cam_display_w, cam_display_h))
        self.screen.blit(cam_surface_scaled, (dash_x + 10, 10))
        y_offset = cam_display_h + 20  # Reduced spacing

        # Health bar
        health_pct = self.spaceship.health / self.spaceship.max_health if self.spaceship.max_health > 0 else 0
        health_color = (int(255 * (1 - health_pct)), int(255 * health_pct), 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (dash_x + 10, y_offset, self.dashboard_width - 20, 20), border_radius=5)
        pygame.draw.rect(self.screen, health_color, (dash_x + 10, y_offset, (self.dashboard_width - 20) * health_pct, 20), border_radius=5)
        health_text = self.font_xs.render(f"SHIELD: {int(health_pct * 100)}%", True, (200, 200, 200))
        self.screen.blit(health_text, (dash_x + 10, y_offset - 20))
        y_offset += 30  # Reduced spacing

        # Score and lives
        score_text = self.font_md.render(f"SCORE: {self.score}", True, (255, 255, 100))
        self.screen.blit(score_text, (dash_x + 10, y_offset))
        for i in range(self.spaceship.lives):
            self.screen.blit(self.heart_icon, (dash_x + 10 + i * 40, y_offset + 40))
        y_offset += 80  # Reduced spacing

        # Ship Systems Section
        systems_title = self.font_md.render("SHIP SYSTEMS", True, (100, 200, 255))
        self.screen.blit(systems_title, (dash_x + 10, y_offset))
        y_offset += 35  # Reduced spacing

        # Steering with visual indicator
        steer_color = (0, 255, 150) if self.spaceship.steering_status != "CENTER" else (100, 100, 100)
        steer_text = self.font_xs.render(f"STEERING: {self.spaceship.steering_status}", True, steer_color)
        self.screen.blit(steer_text, (dash_x + 20, y_offset))
        steer_bar_w = self.dashboard_width - 40
        pygame.draw.rect(self.screen, (50, 50, 50), (dash_x + 20, y_offset + 20, steer_bar_w, 8), border_radius=3)
        if self.spaceship.steering_status == "LEFT":
            pygame.draw.rect(self.screen, steer_color, (dash_x + 20, y_offset + 20, steer_bar_w // 2, 8), border_radius=3)
        elif self.spaceship.steering_status == "RIGHT":
            pygame.draw.rect(self.screen, steer_color, (dash_x + 20 + steer_bar_w // 2, y_offset + 20, steer_bar_w // 2, 8), border_radius=3)
        y_offset += 35  # Reduced spacing

        # Throttle with progress bar
        speed_color = (0, 255, 150) if self.spaceship.speed_status != "NEUTRAL" else (100, 100, 100)
        speed_text = self.font_xs.render(f"THROTTLE: {self.spaceship.speed_status}", True, speed_color)
        self.screen.blit(speed_text, (dash_x + 20, y_offset))
        pygame.draw.rect(self.screen, (50, 50, 50), (dash_x + 20, y_offset + 20, steer_bar_w, 8), border_radius=3)
        if self.spaceship.speed_status == "ACCELERATE":
            pygame.draw.rect(self.screen, speed_color, (dash_x + 20, y_offset + 20, steer_bar_w, 8), border_radius=3)
        elif self.spaceship.speed_status == "DECELERATE":
            pygame.draw.rect(self.screen, speed_color, (dash_x + 20, y_offset + 20, steer_bar_w // 2, 8), border_radius=3)
        y_offset += 35  # Reduced spacing

        # Boost with emoji indicator
        boost_color = (255, 150, 0) if self.spaceship.boost_active else (100, 100, 100)
        boost_text = self.font_xs.render(f"BOOST: {'🚀 ACTIVE' if self.spaceship.boost_active else 'INACTIVE'}", True, boost_color)
        self.screen.blit(boost_text, (dash_x + 20, y_offset))
        y_offset += 45  # Reduced spacing

        # Weapons Array Section
        weapons_title = self.font_md.render("WEAPONS ARRAY", True, (255, 100, 100))
        self.screen.blit(weapons_title, (dash_x + 10, y_offset))
        y_offset += 35  # Reduced spacing

        active_gesture = hand_gestures[0].gesture if hand_gestures else GestureType.NONE
        for gesture_enum in [g for g in GestureType if g != GestureType.NONE]:
            # Determine color based on gesture activation
            is_active = gesture_enum == active_gesture
            color = (0, 255, 150) if is_active else (100, 100, 100)
            
            # Get gesture details
            emoji_char = self.GESTURE_EMOJIS.get(gesture_enum, "❔")
            weapon_desc = self.WEAPON_DESCRIPTIONS.get(gesture_enum, "Unknown")
            display_text = f"{weapon_desc}"  # Simplified to just weapon name to save space
            
            # Render gesture emoji and text
            emoji_surf = self.font_emoji.render(emoji_char, True, color)
            text_surf = self.font_xs.render(display_text, True, color)
            self.screen.blit(emoji_surf, (dash_x + 20, y_offset))
            self.screen.blit(text_surf, (dash_x + 50, y_offset + 5))
            
            # Cooldown bar for gestures with cooldowns
            cooldown = self.weapon_system.weapon_cooldowns.get(gesture_enum, 0)
            max_cooldown = self.COOLDOWN_DURATIONS.get(gesture_enum, 0)
            if max_cooldown > 0:
                cooldown_pct = max(0, cooldown / max_cooldown)
                bar_width = self.dashboard_width - 80
                pygame.draw.rect(self.screen, (50, 50, 50), (dash_x + 50, y_offset + 25, bar_width, 6), border_radius=2)
                if cooldown_pct > 0:
                    pygame.draw.rect(self.screen, (255, 50, 50), (dash_x + 50, y_offset + 25, bar_width * (1 - cooldown_pct), 6), border_radius=2)
            
            # Special case for FIST (charge-based)
            if gesture_enum == GestureType.FIST and is_active and self.weapon_system.charge_start_time is not None:
                charge_ratio = min((time.time() - self.weapon_system.charge_start_time) / self.weapon_system.max_charge_time, 1.0)
                bar_width = self.dashboard_width - 80
                pygame.draw.rect(self.screen, (50, 50, 50), (dash_x + 50, y_offset + 25, bar_width, 6), border_radius=2)
                pygame.draw.rect(self.screen, (255, 255, 100), (dash_x + 50, y_offset + 25, bar_width * charge_ratio, 6), border_radius=2)
            
            y_offset += 35  # Reduced spacing to fit all gestures
    
    def cleanup(self):
        self.cap.release(); pygame.quit()

def main():
    try:
        game = CosmicVoyageGame()
        game.run()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback; traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()