class ExponentialMovingAverage:
    """An extension for calculating EMA of live metrics."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._ema = None

    def update(self, value: float) -> float:
        if self._ema is None:
            self._ema = value
        else:
            self._ema = (self.alpha * value) + ((1.0 - self.alpha) * self._ema)
        return self._ema
