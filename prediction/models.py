from django.db import models


class Prediction(models.Model):
    """Stores a single Parkinson's disease prediction request and result."""
    created_at = models.DateTimeField(auto_now_add=True)
    entry_name = models.CharField(max_length=64, default='')
    prediction_result = models.CharField(max_length=32)
    probability = models.FloatField(null=True, blank=True)

    # Input features from the Parkinson's dataset.
    fo = models.FloatField()
    fhi = models.FloatField()
    flo = models.FloatField()
    jitter_percent = models.FloatField()
    jitter_abs = models.FloatField()
    rap = models.FloatField()
    ppq = models.FloatField()
    ddp = models.FloatField()
    shimmer = models.FloatField()
    shimmer_db = models.FloatField()
    apq3 = models.FloatField()
    apq5 = models.FloatField()
    apq = models.FloatField()
    dda = models.FloatField()
    nhr = models.FloatField()
    hnr = models.FloatField()
    rpde = models.FloatField()
    dfa = models.FloatField()
    spread1 = models.FloatField()
    spread2 = models.FloatField()
    d2 = models.FloatField()
    ppe = models.FloatField()

    def __str__(self):
        return f"Prediction {self.id} - {self.prediction_result}"
