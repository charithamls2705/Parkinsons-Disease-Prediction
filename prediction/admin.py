from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'prediction_result', 'probability', 'created_at')
    readonly_fields = ('created_at',)
