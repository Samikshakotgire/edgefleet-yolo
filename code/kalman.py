from filterpy.kalman import KalmanFilter

class BallTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)  # [x,vx,y,vy,ax,ay]
        # Predict future ball position (parabolic arc)
    
    def predict_next_frame(self, current_pos):
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[2])  # Show PREDICTED position
