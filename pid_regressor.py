from capymoa.base import Regressor
from capymoa.instance import RegressionInstance

class PIDRegressor(Regressor):
    count = 0
    last_value = 0
    previous_error = 0
    
    integral = 0
    derivative = 0
    
    def __init__(self,schema=None, Kp=1, Ki=0, Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        super.__init__(schema)
    
    def train(self, instance:RegressionInstance):
        # calculate error
        error = instance.y_value - self.last_value
        # calculate integral
        self.integral += error # * dt, but we aren't worrying about that, assuming evenly spaced instances
        # calculate derivative
        self.derivative = error - self.previous_error
        
        # update the last values
        self.last_value = instance.y_value
        self.count += 1
        self.previous_error = error
        pass
    
    def predict(self, instance:RegressionInstance) -> int:
        P = self.Kp * instance.y_value
        I = self.Ki * self.integral
        D = self.Kd * self.derivative
        return P + I + D


class LastGuessRegressor(Regressor):
    last_guess = 0
    
    def train(self, instance:RegressionInstance):
        self.last_guess = instance.y_value
        return
    
    def predict(self, instance:RegressionInstance) -> int:
        return self.last_guess