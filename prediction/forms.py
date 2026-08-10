from pathlib import Path
import json
from django import forms

FEATURES_PATH = Path(__file__).resolve().parents[1] / 'models' / 'feature_names.json'


def load_feature_names():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature names file not found: {FEATURES_PATH}")
    with FEATURES_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


class PredictionForm(forms.Form):
    """Django form for Parkinson's disease voice feature input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['entry_name'] = forms.CharField(
            label='Entry Name',
            required=True,
            max_length=64,
            widget=forms.TextInput(
                attrs={
                    'placeholder': 'Patient_001 or Test_001',
                }
            ),
            error_messages={
                'required': 'An entry name is required.',
                'max_length': 'Entry Name must be 64 characters or fewer.',
            },
        )
        feature_names = load_feature_names()
        for feature in feature_names:
            self.fields[feature] = forms.FloatField(
                label=feature,
                required=True,
                error_messages={
                    'required': 'This field is required.',
                    'invalid': 'Enter a valid number.',
                },
                widget=forms.NumberInput(
                    attrs={
                        'step': 'any',
                        'placeholder': 'Enter numeric value',
                    }
                ),
            )
