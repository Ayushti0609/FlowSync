# This file smooths hand landmark coordinates
# to reduce frame-to-frame jitter.

class LandmarkSmoother:

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_landmarks = None

    def smooth(self, landmarks):

        if landmarks is None:
            return None

        # First frame: no previous data available
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return landmarks

        smoothed_landmarks = {}

        for name, current_point in landmarks.items():

            # If current landmark is missing
            if current_point is None:

                # Use previous point if available
                if self.previous_landmarks.get(name) is not None:
                    smoothed_landmarks[name] = self.previous_landmarks[name]
                else:
                    smoothed_landmarks[name] = None

                continue

            previous_point = self.previous_landmarks.get(name)

            # If previous point is missing,
            # use the current point directly
            if previous_point is None:

                smoothed_landmarks[name] = current_point
                continue

            # Exponential Moving Average
            smoothed_landmarks[name] = {
                "x": (
                    self.alpha * current_point["x"]
                    + (1 - self.alpha) * previous_point["x"]
                ),

                "y": (
                    self.alpha * current_point["y"]
                    + (1 - self.alpha) * previous_point["y"]
                ),

                "z": (
                    self.alpha * current_point["z"]
                    + (1 - self.alpha) * previous_point["z"]
                )
            }

        # Store current smoothed landmarks
        # for the next frame
        self.previous_landmarks = smoothed_landmarks

        return smoothed_landmarks

    def reset(self):
        self.previous_landmarks = None