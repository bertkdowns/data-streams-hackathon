from capymoa.base import Regressor
from capymoa.instance import RegressionInstance
import random
class PIDRegressor(Regressor):
    count = 0
    last_value = 0
    previous_error = 0
    
    integral = 0
    derivative = 0
    
    def __init__(self,schema=None, Kp=1, Ki=0, Kd=0):
        super().__init__(schema=schema)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
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
        P = self.Kp * self.last_value
        I = self.Ki * self.integral
        D = self.Kd * self.derivative
        return P + I + D
    
    def __str__(self):
        """Return a string representation of the PIDRegressor. This is needed for evaluation scripts"""
        return f"PID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"


class LastGuessRegressor(Regressor):
    last_guess = 0
    
    def train(self, instance:RegressionInstance):
        self.last_guess = instance.y_value
        return
    
    def predict(self, instance:RegressionInstance) -> int:
        return self.last_guess
    
    def __str__(self):
        """Return a string representation of the LastGuessRegressor. This is needed for evaluation scripts"""
        return f"LastGuessRegressor"


class RandomRegressor(Regressor):
    min = 99999999999
    max = -999999999
    
    def train(self, instance:RegressionInstance):
        """Update the min and max values seen during training."""
        if instance.y_value < self.min:
            self.min = instance.y_value
        if instance.y_value > self.max:
            self.max = instance.y_value
        return
    
    def predict(self, instance:RegressionInstance) -> int:
        """Return a random integer between the min and max values seen during training."""
        if self.min == 99999999999 or self.max == -999999999:
            return 0 # initial guess
        return random.uniform(self.min, self.max)
    
    def __str__(self):
        """Return a string representation of the RandomRegressor."""
        return f"RandomRegressor"

class GaussianRandomRegressor(Regressor):
    mean = 0
    std_dev = 0
    
    def train(self, instance:RegressionInstance):
        """Update the mean and standard deviation values seen during training."""
        self.mean = (self.mean + instance.y_value) / 2
        self.std_dev = (self.std_dev + abs(instance.y_value - self.mean)) / 2
        return
    
    def predict(self, instance:RegressionInstance) -> int:
        """Return a random integer between the min and max values seen during training."""
        return random.gauss(self.mean, self.std_dev)
    
    def __str__(self):
        """Return a string representation of the GaussianRandomRegressor."""
        return f"GaussianRandomRegressor"