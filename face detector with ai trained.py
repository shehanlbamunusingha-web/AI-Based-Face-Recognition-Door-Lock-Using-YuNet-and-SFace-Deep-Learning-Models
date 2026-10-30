import os
import sys
import csv
import time
import random
import cv2
import numpy as np
from collections import deque, Counter

# ==========================
# CONFIG
# ==========================
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
DB_FILE = "face_db.npz"          # saved faces (kept after closing the program)
LOG_FILE = "access_log.csv"      # every grant / deny / lockout is logged here

# ---- detection / saving ----
DETECT_SCORE = 0.9
SAMPLES_PER_POSE = 5             # frames captured for every position
SAMPLE_GAP = 0.25                # seconds between samples
COUNTDOWN = 2                    # seconds before guided capture starts
PAUSE_BETWEEN_POSES = 0.8
BAD_FRAMES_ALLOWED = 3           # tolerate short jitter before restarting a pose
YAW_TURN = 0.18                  # turn needed for LEFT / RIGHT (about 20 degrees)
YAW_MAX = 0.70                   # too far turned
PITCH_TURN = 0.06                # tilt needed for UP / DOWN (about 10 degrees)
PITCH_MAX = 0.35                 # too far tilted
PITCH_SIGN = 1                   # set to -1 if UP / DOWN work the wrong way round
HOLD_FACTOR = 0.7                # once the pose is reached you only need 70% to keep it
RELAX_EVERY = 3.0                # seconds: if a pose is not reached, thresholds relax
SELF_TEST_SECONDS = 4.0          # quick test right after saving a face

# ---- image quality gates (saving AND identifying) ----
MIN_SHARPNESS = 20.0             # lower = accepts blurrier images
BRIGHT_RANGE = (50, 210)         # accepted average brightness (0-255)
MIN_FACE_FRAC = 0.17             # face width must be >= 17% of frame width
SIZE_RANGE = {                   # allowed face width / frame width per pose
    "default": (MIN_FACE_FRAC, 0.55),
    "FAR":     (MIN_FACE_FRAC, 0.30),
    "CLOSE":   (0.36, 0.62),
}

# ---- identification accuracy ----
MATCH_THRESHOLD = 0.45           # stricter than SFace default 0.363
TOP_K = 3                        # score = mean of the best K templates
MARGIN = 0.05                    # best person must beat 2nd person by this
MAX_MATCH_YAW = 0.30             # only match when face is nearly frontal
WINDOW_N = 10                    # look at the last N valid frames ...
REQUIRED = 8                     # ... and need this many to agree

# ---- liveness (random head-movement challenge) ----
ENABLE_LIVENESS = False         # False = automatic unlock, no left/right prompts
                                 # True  = extra head-movement challenge
LIVENESS_STEPS = 2               # number of random actions (LEFT/RIGHT/UP/DOWN)
LIVE_YAW = 0.16                  # turn needed for LEFT / RIGHT
LIVE_PITCH = 0.05                # tilt needed for UP / DOWN
CHALLENGE_TIMEOUT = 6            # seconds allowed per action
LIVENESS_RETRIES = 1             # new random challenge if you are too slow
LIVE_DETECT_SCORE = 0.6          # easier detection while the head is turned
FACE_LOST_GRACE = 1.0            # seconds a lost face is tolerated
LIVE_LOST_GRACE = 2.0            # same, during the liveness challenge

# ---- door behaviour ----
UNLOCK_SECONDS = 5               # how long the door stays unlocked
COOLDOWN = 3                     # seconds before next attempt after a grant
DENY_FRAMES = 20                 # unknown for this many frames = failed attempt
MAX_FAILS = 5                    # failed attempts before lockout
LOCKOUT_SECONDS = 60
MSG_HOLD = 0.5                   # seconds a status text stays before it may change

CAM_INDEX = 0
WINDOW = "Face Recognition"
GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
COSINE = cv2.FaceRecognizerSF_FR_COSINE

POSES = [
    ("STRAIGHT", "Look STRAIGHT at the camera"),
    ("LEFT",     "<<< Turn your face to the LEFT"),
    ("RIGHT",    "Turn your face to the RIGHT >>>"),
    ("UP",       "^^^ Tilt your face UP a little"),
    ("DOWN",     "vvv Tilt your face DOWN a little"),
    ("FAR",      "Step BACK a little (look straight)"),
    ("CLOSE",    "Move CLOSER (look straight)"),
]
TURN_KEYS = ("LEFT", "RIGHT")
TILT_KEYS = ("UP", "DOWN")
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}

CHALLENGE_TEXT = {
    "LEFT":  "<<< TURN your face LEFT",
    "RIGHT": "TURN your face RIGHT >>>",
    "UP":    "^^^ LOOK UP a little",
    "DOWN":  "vvv LOOK DOWN a little",
}

# ==========================
# LOAD MODELS + CAMERA
# ==========================
for path in (YUNET_MODEL, SFACE_MODEL):
    if not os.path.exists(path):
        sys.exit(f"Model file not found: {path}")

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (320, 320), DETECT_SCORE, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    sys.exit("Cannot open camera")


# ==========================
# DOOR HARDWARE  (edit this part)
# ==========================
def unlock_door(name):
    """Called once when access is granted.
    Replace the print with your hardware code. Do NOT block with
    time.sleep() here (it would freeze the video); use a Timer.

    Raspberry Pi example:
        from gpiozero import OutputDevice
        relay = OutputDevice(17)                 # create once at top of file
        relay.on()
        threading.Timer(UNLOCK_SECONDS, relay.off).start()

    Arduino (serial) example:
        ser.write(b"U")                          # Arduino unlocks and re-locks
    """
    print(f"[DOOR] UNLOCK for {name} ({UNLOCK_SECONDS}s)")


def log_event(event, name="", score=0.0, note=""):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "name", "score", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, name,
                    f"{score:.3f}", note])


# ==========================
# DATABASE  (several templates per person)
# ==========================
def load_db():
    db = {}
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=False)
        for n, f in zip(data["names"], data["features"]):
            db.setdefault(str(n), []).append(f.astype(np.float32))
    return {n: np.vstack(v) for n, v in db.items()}   # name -> (k, 128)


def save_db(db):
    names, feats = [], []
    for n, arr in db.items():
        for row in arr:
            names.append(n)
            feats.append(row)
    np.savez(DB_FILE, names=np.array(names),
             features=np.vstack(feats).astype(np.float32))


# ==========================
# HELPERS
# ==========================
def read_frame():
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.flip(frame, 1)  # mirror view (like a real mirror)


def detect_faces(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return [] if faces is None else list(faces)


def get_feature(frame, face):
    aligned = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned)


def head_pose(face):
    """Yaw / pitch estimate from YuNet's 5 landmarks.
    yaw < 0 : face turned to your LEFT (mirror view)
    pitch   : (with PITCH_SIGN = 1) smaller = looking up, bigger = looking down"""
    lm = face[4:14].reshape(5, 2)
    r_eye, l_eye, nose, r_mouth, l_mouth = lm
    eye_mid = (r_eye + l_eye) / 2
    mouth_mid = (r_mouth + l_mouth) / 2
    eye_dist = max(np.linalg.norm(r_eye - l_eye), 1e-6)
    yaw = (nose[0] - eye_mid[0]) / eye_dist
    pitch = (nose[1] - eye_mid[1]) / max(mouth_mid[1] - eye_mid[1], 1e-6)
    return float(yaw), float(PITCH_SIGN * pitch)


def smooth(prev, new, a=0.5):
    return new if prev is None else a * new + (1 - a) * prev


def face_quality(frame, face, sharp_scale=1.0):
    """Reject blurry / too dark / too bright faces."""
    H, W = frame.shape[:2]
    x, y, w, h = map(int, face[:4])
    crop = frame[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
    if crop.size == 0:
        return False, "Face not clear"
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (112, 112))
    bright = float(gray.mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if bright < BRIGHT_RANGE[0]:
        return False, "Too dark - add more light"
    if bright > BRIGHT_RANGE[1]:
        return False, "Too bright - reduce light"
    if sharp < MIN_SHARPNESS * sharp_scale:
        return False, "Image blurry - hold still"
    return True, ""


def check_size(face, W):
    if face[2] < MIN_FACE_FRAC * W:
        return False, "Move CLOSER to the camera"
    return True, ""


def check_position(face, W, H, key="default"):
    """For saving: face near the centre oval with the size this pose needs.
    Turning / tilting moves the face, so those poses allow more drift."""
    lo, hi = SIZE_RANGE.get(key, SIZE_RANGE["default"])
    tol = 0.28 if key in TURN_KEYS + TILT_KEYS else 0.18
    x, y, w, h = face[:4]
    cx, cy = x + w / 2, y + h / 2
    if abs(cx - W / 2) > tol * W or abs(cy - H / 2) > tol * H:
        return False, "Move your face to the CENTER oval"
    if w < lo * W:
        return False, "Move CLOSER to the camera"
    if w > hi * W:
        return False, "Step BACK a little more" if key == "FAR" \
            else "Move BACK a little"
    return True, ""


def pose_progress(key, dy, dp):
    """How far the head has moved towards the wanted direction (+ = good)."""
    if key == "LEFT":
        return -dy
    if key == "RIGHT":
        return dy
    if key == "UP":
        return -dp
    if key == "DOWN":
        return dp
    return 0.0


def pose_threshold(key, f=1.0):
    if key in TURN_KEYS:
        return max(YAW_TURN * f, 0.08)
    if key in TILT_KEYS:
        return max(PITCH_TURN * f, 0.025)
    return 0.0


def check_pose(key, yaw, pitch, base, f=1.0):
    """f < 1 makes the pose easier (hold hysteresis / relax over time)."""
    dy = yaw - base["yaw"]
    dp = pitch - base["pitch"]

    if key == "STRAIGHT":
        if abs(yaw) > 0.20:
            return False, "Look straight at the camera"
        return True, ""
    if key in ("FAR", "CLOSE"):
        if abs(dy) > 0.20 or abs(dp) > 0.20:
            return False, "Look straight at the camera"
        return True, ""

    prog = pose_progress(key, dy, dp)
    thr = pose_threshold(key, f)
    if prog < thr:
        if prog < -0.8 * thr:
            return False, f"That is {OPPOSITE[key]} - go {key} instead"
        if key in TURN_KEYS:
            return False, f"Turn a little more to the {key}"
        return False, f"Tilt a little more {key}"
    if key in TURN_KEYS:
        if prog > YAW_MAX:
            return False, "Too far - turn back a little"
        if abs(dp) > 0.25:
            return False, "Keep your head level"
    else:
        if prog > PITCH_MAX:
            return False, "Too far - tilt back a little"
        if abs(dy) > 0.25:
            return False, "Don't turn sideways"
    return True, ""


def challenge_done(key, dy, dp):
    if key == "LEFT":
        return dy <= -LIVE_YAW
    if key == "RIGHT":
        return dy >= LIVE_YAW
    if key == "UP":
        return dp <= -LIVE_PITCH
    if key == "DOWN":
        return dp >= LIVE_PITCH
    return False


def person_score(templates, feat):
    """Mean of the best TOP_K template scores (more stable than 1 best)."""
    scores = sorted(
        (recognizer.match(t.reshape(1, -1), feat, COSINE) for t in templates),
        reverse=True)
    k = min(TOP_K, len(scores))
    return float(np.mean(scores[:k]))


def best_match(db, feat):
    """Returns (name, score, accepted). Needs threshold AND margin."""
    ranked = sorted(((n, person_score(t, feat)) for n, t in db.items()),
                    key=lambda kv: kv[1], reverse=True)
    name, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    accepted = best >= MATCH_THRESHOLD and (best - second) >= MARGIN
    return name, best, accepted


def enrollment_consistency(feats):
    """Mean cosine between all saved samples (low = bad enrollment)."""
    s = [recognizer.match(feats[i].reshape(1, -1), feats[j].reshape(1, -1), COSINE)
         for i in range(len(feats)) for j in range(i + 1, len(feats))]
    return float(np.mean(s)) if s else 0.0


# ---------- drawing ----------
def draw_box(frame, face, color, label=None):
    x, y, w, h = map(int, face[:4])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def put_text(frame, text, pos, color=WHITE, scale=0.6, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick)


class StableMsg:
    """Keeps the status text (and its colour) steady so it does not flicker."""

    def __init__(self, hold=0.5):
        self.hold = hold
        self.msg, self.color, self.t = "", WHITE, 0.0

    def update(self, msg, color, now, force=False):
        if (msg, color) != (self.msg, self.color) and \
                (force or now - self.t >= self.hold):
            self.msg, self.color, self.t = msg, color, now
        return self.msg, self.color


def draw_guide(frame, ok):
    H, W = frame.shape[:2]
    color = GREEN if ok else WHITE
    cv2.ellipse(frame, (W // 2, H // 2), (int(W * 0.17), int(H * 0.38)),
                0, 0, 360, color, 2)


def draw_progress(frame, done, total):
    H, W = frame.shape[:2]
    x0, y0, bw, bh = 10, H - 45, W - 20, 10
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bw * done / total), y0 + bh),
                  GREEN, -1)


def draw_gauge(frame, key, dy, dp, thr_yaw, thr_pitch):
    """Shows how far you have moved: the dot must reach the green zone."""
    H, W = frame.shape[:2]
    if key in TURN_KEYS:
        L = 300
        x0, cx, y = W // 2 - L // 2, W // 2, H - 75
        scale = (L / 2) / 0.8
        sign = -1 if key == "LEFT" else 1
        cv2.line(frame, (x0, y), (x0 + L, y), (120, 120, 120), 6)
        z0, z1 = cx + sign * thr_yaw * scale, cx + sign * 0.8 * scale
        cv2.line(frame, (int(min(z0, z1)), y), (int(max(z0, z1)), y),
                 (0, 180, 0), 6)
        cv2.line(frame, (cx, y - 8), (cx, y + 8), WHITE, 2)
        mx = int(np.clip(cx + dy * scale, x0, x0 + L))
        cv2.circle(frame, (mx, y), 9,
                   GREEN if sign * dy >= thr_yaw else YELLOW, -1)
    elif key in TILT_KEYS:
        L = 180
        x, cy = W - 30, H // 2 + 40
        scale = (L / 2) / 0.3
        sign = -1 if key == "UP" else 1
        cv2.line(frame, (x, cy - L // 2), (x, cy + L // 2), (120, 120, 120), 6)
        z0, z1 = cy + sign * thr_pitch * scale, cy + sign * 0.3 * scale
        cv2.line(frame, (x, int(min(z0, z1))), (x, int(max(z0, z1))),
                 (0, 180, 0), 6)
        cv2.line(frame, (x - 8, cy), (x + 8, cy), WHITE, 2)
        my = int(np.clip(cy + dp * scale, cy - L // 2, cy + L // 2))
        cv2.circle(frame, (x, my), 9,
                   GREEN if sign * dp >= thr_pitch else YELLOW, -1)


def draw_checklist(frame, idx):
    H, W = frame.shape[:2]
    for i, (k, _) in enumerate(POSES):
        mark = "[x]" if i < idx else (">" if i == idx else "[ ]")
        col = GREEN if i < idx else (YELLOW if i == idx else WHITE)
        put_text(frame, f"{mark} {k}", (W - 130, 25 + 20 * i), col, 0.5, 1)


def ask_yes_no(question):
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def close_windows():
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


# ==========================
# STEP 2: SAVE FACE (guided)
# ==========================
def save_face(name):
    base = {"yaw": 0.0, "pitch": PITCH_SIGN * 0.55}
    start = time.time()
    pause_until = 0.0
    pose_t0 = start + COUNTDOWN       # when the current pose became active
    last_sample = 0.0
    idx = 0
    bad = 0
    entered = False                   # pose reached once -> easier to keep
    ys = ps = None
    pose_feats, pose_metrics = [], []
    all_feats = []
    total = len(POSES) * SAMPLES_PER_POSE
    sm = StableMsg(MSG_HOLD)

    def finish_pose(now):
        nonlocal idx, bad, entered, pose_feats, pose_metrics, pause_until, pose_t0
        idx += 1
        bad = 0
        entered = False
        pose_feats, pose_metrics = [], []
        pause_until = now + PAUSE_BETWEEN_POSES
        pose_t0 = pause_until

    while True:
        frame = read_frame()
        if frame is None:
            print("Camera frame error")
            return None

        H, W = frame.shape[:2]
        faces = detect_faces(frame)
        now = time.time()
        guide_ok = False
        box_color = YELLOW
        key, instr = POSES[min(idx, len(POSES) - 1)]
        gauge, info = None, ""
        can_skip = False

        if now - start < COUNTDOWN:
            title = f"Get ready... {int(COUNTDOWN - (now - start)) + 1}"
            msg = "Put your face inside the oval"
            instr = ""
        elif now < pause_until:
            title = "Good!"
            msg = f"Next: {POSES[idx][1]}" if idx < len(POSES) else ""
            instr = ""
        else:
            title = f"Step {idx + 1}/{len(POSES)}"
            can_skip = idx > 0
            # thresholds relax the longer a pose takes; holding is easier
            relax = max(0.4, 1.0 - 0.2 * int(max(now - pose_t0, 0) / RELAX_EVERY))
            f = relax * (HOLD_FACTOR if entered else 1.0)

            if len(faces) == 0:
                msg = "No face detected - face the camera"
                pose_feats.clear(); pose_metrics.clear(); bad = 0
            elif len(faces) > 1:
                msg = "Only ONE person please"
                pose_feats.clear(); pose_metrics.clear(); bad = 0
            else:
                face = faces[0]
                yaw, pitch = head_pose(face)
                ys, ps = smooth(ys, yaw), smooth(ps, pitch)
                dy, dp = ys - base["yaw"], ps - base["pitch"]

                pose_ok, pose_msg = check_pose(key, ys, ps, base, f)
                pos_ok, pos_msg = check_position(face, W, H, key)
                q_ok, q_msg = face_quality(
                    frame, face, 0.6 if key == "FAR" else 1.0)

                gauge = (key, dy, dp,
                         pose_threshold("LEFT", f), pose_threshold("UP", f))
                if key in TURN_KEYS + TILT_KEYS:
                    info = (f"movement {pose_progress(key, dy, dp):+.2f}"
                            f" / need {pose_threshold(key, f):.2f}")
                    if relax < 1.0:
                        info += "  (easier mode)"

                if pose_ok:
                    entered = True
                if pose_ok and pos_ok and q_ok:
                    bad = 0
                    guide_ok = True
                    box_color = GREEN
                    msg = "Hold still... capturing"
                    if now - last_sample >= SAMPLE_GAP:
                        pose_feats.append(get_feature(frame, face))
                        pose_metrics.append((yaw, pitch))
                        last_sample = now
                else:
                    # size is the main point of FAR / CLOSE, pose for the rest
                    order = ([(pos_ok, pos_msg), (pose_ok, pose_msg), (q_ok, q_msg)]
                             if key in ("FAR", "CLOSE") else
                             [(pose_ok, pose_msg), (pos_ok, pos_msg), (q_ok, q_msg)])
                    msg = next(m for ok, m in order if not ok)
                    bad += 1
                    if bad >= BAD_FRAMES_ALLOWED:
                        pose_feats.clear(); pose_metrics.clear()

            if len(pose_feats) >= SAMPLES_PER_POSE:
                all_feats.extend(pose_feats)
                if key == "STRAIGHT":
                    m = np.mean(np.array(pose_metrics), axis=0)
                    base["yaw"], base["pitch"] = float(m[0]), float(m[1])
                finish_pose(now)
                if idx >= len(POSES):
                    return np.vstack(all_feats).astype(np.float32)

        shown, shown_color = sm.update(msg, GREEN if guide_ok else WHITE, now)
        draw_guide(frame, shown_color == GREEN)
        if len(faces) == 1:
            draw_box(frame, faces[0], shown_color)
        if gauge is not None:
            draw_gauge(frame, *gauge)

        put_text(frame, f"Saving face for: {name}", (10, 25), YELLOW)
        put_text(frame, title, (10, 55), WHITE)
        if instr:
            put_text(frame, instr, (10, 95), YELLOW, 0.8, 2)
        put_text(frame, shown, (10, 130), shown_color, 0.65)
        if info:
            put_text(frame, info, (10, 158), WHITE, 0.55, 1)
        draw_checklist(frame, idx)
        draw_progress(frame, len(all_feats) + len(pose_feats), total)
        put_text(frame, "Q = cancel   N = skip this step" if can_skip
                 else "Q = cancel", (10, H - 15))
        cv2.imshow(WINDOW, frame)

        kp = cv2.waitKey(1) & 0xFF
        if kp == ord("q"):
            return None
        if kp == ord("n") and can_skip:
            print(f"[SAVE] skipped step: {key}")
            finish_pose(now)
            if idx >= len(POSES):
                return np.vstack(all_feats).astype(np.float32)


def self_test(templates):
    """After saving: look straight and see that the new data really matches you.
    Returns the average score, or None if it could not run / was cancelled."""
    start = time.time()
    scores = []
    while time.time() - start < SELF_TEST_SECONDS:
        frame = read_frame()
        if frame is None:
            return None
        H, W = frame.shape[:2]
        faces = detect_faces(frame)
        msg, color = "Self-test: look straight at the camera", YELLOW
        if len(faces) == 1:
            face = faces[0]
            yaw, _ = head_pose(face)
            q_ok, q_msg = face_quality(frame, face)
            if not q_ok:
                msg = q_msg
            elif abs(yaw) > MAX_MATCH_YAW:
                msg = "Look straight at the camera"
            else:
                s = person_score(templates, get_feature(frame, face))
                scores.append(s)
                msg, color = f"Self-test score: {s:.2f}", GREEN
            draw_box(frame, face, color)
        put_text(frame, msg, (10, 30), color, 0.7)
        put_text(frame, "Q = skip", (10, H - 15))
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    return float(np.mean(scores)) if len(scores) >= 5 else None


# ==========================
# STEP 3: IDENTIFY + UNLOCK
# ==========================
def identify(db):
    mode = "head-movement check ON" if ENABLE_LIVENESS else "automatic unlock"
    print("\nIdentification started. Press Q in the video window to quit.")
    print(f"Threshold={MATCH_THRESHOLD}, need {REQUIRED}/{WINDOW_N} matching "
          f"frames, mode: {mode}")

    rng = random.SystemRandom()
    sm = StableMsg(MSG_HOLD)
    S = {
        "state": "VERIFY",       # VERIFY -> (TURN -> RETURN -> VERIFY2 ->) grant
        "hist": deque(maxlen=WINDOW_N),
        "pose_hist": deque(maxlen=10),
        "deny": 0, "fails": 0, "tries": 0,
        "who": None, "ch": None, "last_first": None,
        "ys": None, "ps": None,
        "last_seen": 0.0, "last_score": 0.0,
        "lockout": 0.0, "hold_until": 0.0,
        "hold_msg": "", "hold_color": WHITE,
    }

    def reset():
        S["state"] = "VERIFY"
        S["hist"].clear()
        S["pose_hist"].clear()
        S["deny"] = 0
        S["tries"] = 0
        S["who"] = None
        S["ch"] = None
        S["ys"] = S["ps"] = None

    def grant(name, score):
        print(f"[ACCESS GRANTED] {name} ({score:.2f})")
        log_event("GRANTED", name, score)
        unlock_door(name)
        S["fails"] = 0
        S["hold_until"] = time.time() + UNLOCK_SECONDS + COOLDOWN
        S["hold_msg"] = f"ACCESS GRANTED: {name}"
        S["hold_color"] = GREEN
        reset()

    def deny(reason, score):
        now = time.time()
        S["fails"] += 1
        print(f"[ACCESS DENIED] {reason} (fail {S['fails']}/{MAX_FAILS})")
        log_event("DENIED", "", score, reason)
        if S["fails"] >= MAX_FAILS:
            S["lockout"] = now + LOCKOUT_SECONDS
            S["fails"] = 0
            log_event("LOCKOUT", "", 0.0, f"{LOCKOUT_SECONDS}s")
            print(f"[LOCKOUT] {LOCKOUT_SECONDS}s")
        S["hold_until"] = now + 2
        S["hold_msg"] = f"ACCESS DENIED: {reason}"
        S["hold_color"] = RED
        reset()

    def start_challenge(base_yaw, base_pitch, now):
        """Random actions; the first one never repeats the previous attempt's."""
        options = ["LEFT", "RIGHT", "UP", "DOWN"]
        while True:
            steps = rng.sample(options, LIVENESS_STEPS)
            if steps[0] != S["last_first"]:
                break
        S["last_first"] = steps[0]
        S["ch"] = {"steps": steps, "i": 0, "by": base_yaw, "bp": base_pitch,
                   "deadline": now + CHALLENGE_TIMEOUT}
        S["state"] = "TURN"
        S["hist"].clear()
        print(f"[LIVENESS] challenge: {' then '.join(steps)}")

    def challenge_timeout(now, score=0.0):
        S["tries"] += 1
        if S["tries"] <= LIVENESS_RETRIES:
            ch = S["ch"]
            start_challenge(ch["by"], ch["bp"], now)
        else:
            deny("liveness timeout", score)

    prev = time.time()
    while True:
        frame = read_frame()
        if frame is None:
            print("Camera frame error")
            break

        H, W = frame.shape[:2]
        in_challenge = S["state"] in ("TURN", "RETURN")
        # turned faces get a lower detection score so they are not lost
        detector.setScoreThreshold(LIVE_DETECT_SCORE if in_challenge else DETECT_SCORE)
        faces = detect_faces(frame)
        now = time.time()
        msg, color, big, small = "", WHITE, "", ""
        forced = False
        box_face, box_label, box_color, gauge = None, None, YELLOW, None

        if now < S["lockout"]:
            msg = f"LOCKED OUT - wait {int(S['lockout'] - now) + 1}s"
            color, forced = RED, True
        elif now < S["hold_until"]:
            msg, color, forced = S["hold_msg"], S["hold_color"], True
            if len(faces) == 1:
                box_face, box_color = faces[0], S["hold_color"]
        elif len(faces) == 0:
            msg = "Face lost - look at the camera" if in_challenge \
                else "Show your face to unlock"
            grace = LIVE_LOST_GRACE if in_challenge else FACE_LOST_GRACE
            if now - S["last_seen"] > grace:
                reset()
        elif len(faces) > 1:
            msg = "Only one person at a time"
            reset()
        else:
            face = faces[0]
            box_face = face
            S["last_seen"] = now
            yaw, pitch = head_pose(face)
            S["ys"] = smooth(S["ys"], yaw)
            S["ps"] = smooth(S["ps"], pitch)
            state = S["state"]

            if state in ("VERIFY", "VERIFY2"):
                size_ok, size_msg = check_size(face, W)
                q_ok, q_msg = face_quality(frame, face)
                if not size_ok:
                    msg = size_msg
                elif not q_ok:
                    msg = q_msg
                elif abs(yaw) > MAX_MATCH_YAW:
                    msg = "Look straight at the camera"
                else:
                    if state == "VERIFY":
                        S["pose_hist"].append((yaw, pitch))
                    feat = get_feature(frame, face)
                    name, score, ok = best_match(db, feat)
                    S["last_score"] = score
                    S["hist"].append(name if ok else None)
                    S["deny"] = 0 if ok else S["deny"] + 1

                    cnt = Counter(n for n in S["hist"] if n is not None)
                    who, votes = cnt.most_common(1)[0] if cnt else (None, 0)

                    msg = "Verifying..." if state == "VERIFY2" \
                        else "Scanning your face..."
                    color = YELLOW
                    box_label = f"Scanning {min(votes, REQUIRED)}/{REQUIRED}"
                    small = f"match score {score:.2f} (need {MATCH_THRESHOLD})"

                    if len(S["hist"]) == WINDOW_N and votes >= REQUIRED:
                        if state == "VERIFY" and ENABLE_LIVENESS:
                            if len(S["pose_hist"]) >= 3:
                                by, bp = np.mean(np.array(S["pose_hist"]), axis=0)
                            else:
                                by, bp = yaw, pitch
                            S["who"] = who
                            start_challenge(float(by), float(bp), now)
                        elif state == "VERIFY2" and who != S["who"]:
                            deny("identity changed during liveness", score)
                        else:
                            grant(who, S["last_score"])
                    elif S["deny"] >= DENY_FRAMES:
                        deny("face not recognised", score)

            elif state == "TURN":
                ch = S["ch"]
                key = ch["steps"][ch["i"]]
                dy, dp = S["ys"] - ch["by"], S["ps"] - ch["bp"]
                remaining = ch["deadline"] - now
                thr = LIVE_YAW if key in TURN_KEYS else LIVE_PITCH
                prog = pose_progress(key, dy, dp)
                big = CHALLENGE_TEXT[key]
                msg = (f"LIVENESS {ch['i'] + 1}/{len(ch['steps'])}  "
                       f"{max(int(remaining) + 1, 0)}s left")
                small = f"move {prog:+.2f} / {thr:.2f}"
                if prog < -0.8 * thr:
                    small += f"  (that is {OPPOSITE[key]}!)"
                color = YELLOW
                gauge = (key, dy, dp, LIVE_YAW, LIVE_PITCH)
                if remaining <= 0:
                    challenge_timeout(now)
                elif challenge_done(key, dy, dp):
                    S["state"] = "RETURN"
                    ch["deadline"] = now + CHALLENGE_TIMEOUT

            elif state == "RETURN":
                ch = S["ch"]
                dy, dp = S["ys"] - ch["by"], S["ps"] - ch["bp"]
                big = "Good! Now look STRAIGHT at the camera"
                msg = f"{max(int(ch['deadline'] - now) + 1, 0)}s left"
                color = YELLOW
                if now > ch["deadline"]:
                    challenge_timeout(now)
                elif abs(dy) < 0.15 and abs(dp) < 0.12:
                    if ch["i"] + 1 < len(ch["steps"]):
                        ch["i"] += 1
                        S["state"] = "TURN"
                        ch["deadline"] = now + CHALLENGE_TIMEOUT
                    else:
                        S["state"] = "VERIFY2"
                        S["hist"].clear()

        # ---- draw (colours stay steady: white = info, yellow = scanning,
        #      green = granted, red = denied / locked) ----
        shown, shown_color = sm.update(msg, color, now, force=forced)
        if box_face is not None:
            draw_box(frame, box_face, box_color, box_label)
        if gauge is not None:
            draw_gauge(frame, *gauge)
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        put_text(frame, f"Door lock | FPS {fps:.0f}", (10, 25))
        if big:
            put_text(frame, big, (10, 65), YELLOW, 0.9, 2)
            put_text(frame, shown, (10, 100), shown_color, 0.6)
        else:
            put_text(frame, shown, (10, 60), shown_color, 0.8)
        if small:
            put_text(frame, small, (10, H - 40), WHITE, 0.5, 1)
        put_text(frame, f"Fails: {S['fails']}/{MAX_FAILS}   Q = quit",
                 (10, H - 15))
        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ==========================
# MAIN FLOW
# ==========================
def main():
    db = load_db()

    # ---------- STEP 1: ASK ----------
    if db:
        print(f"Saved faces found: {', '.join(db.keys())}")

    while True:
        if db:
            want_save = ask_yes_no("Do you want to save a new face?")
        else:
            print("No saved face found. You must save at least one face.")
            want_save = True

        if not want_save:
            break

        # ---------- STEP 2: SAVE ----------
        name = input("Enter name for this face: ").strip()
        if not name:
            name = f"Person{len(db) + 1}"

        if name in db and not ask_yes_no(f"'{name}' already exists. Replace?"):
            continue

        print("Sit in front of the camera with good, even light and follow "
              "the on-screen instructions (straight, left, right, up, down, "
              "far, close). Press N to skip a hard step.")
        feats = save_face(name)
        cons, test = 0.0, None
        if feats is not None:
            cons = enrollment_consistency(feats)
            test = self_test(feats)
        close_windows()

        if feats is None:
            print("Saving cancelled.")
            continue

        print(f"Enrollment consistency: {cons:.2f}")
        if test is None:
            print("Self-test: skipped / not enough good frames.")
        else:
            print(f"Self-test score: {test:.2f} "
                  f"(door opens above {MATCH_THRESHOLD})")

        weak = cons < 0.40 or (test is not None and test < MATCH_THRESHOLD + 0.05)
        if weak:
            print("WARNING: this face data is weak (bad light, blur or "
                  "movement). Saving it again is recommended.")
            if ask_yes_no("Save it again now?"):
                continue

        db[name] = feats
        save_db(db)
        print(f"Face saved successfully for '{name}' ({len(feats)} samples).")

        if not ask_yes_no("Save another face?"):
            break

    # ---------- STEP 3: IDENTIFY ----------
    if db:
        identify(db)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
