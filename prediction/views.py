import json
from pathlib import Path
from django.shortcuts import render, get_object_or_404
from django.db import DatabaseError
from joblib import load

from .forms import PredictionForm
from .models import Prediction

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'parkinsons_knn_pipeline.joblib'
FEATURES_PATH = Path(__file__).resolve().parents[1] / 'models' / 'feature_names.json'

FEATURE_NAME_MAP = {
    'MDVP:Fo(Hz)': 'fo',
    'MDVP:Fhi(Hz)': 'fhi',
    'MDVP:Flo(Hz)': 'flo',
    'MDVP:Jitter(%)': 'jitter_percent',
    'MDVP:Jitter(Abs)': 'jitter_abs',
    'MDVP:RAP': 'rap',
    'MDVP:PPQ': 'ppq',
    'Jitter:DDP': 'ddp',
    'MDVP:Shimmer': 'shimmer',
    'MDVP:Shimmer(dB)': 'shimmer_db',
    'Shimmer:APQ3': 'apq3',
    'Shimmer:APQ5': 'apq5',
    'MDVP:APQ': 'apq',
    'Shimmer:DDA': 'dda',
    'NHR': 'nhr',
    'HNR': 'hnr',
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'spread1': 'spread1',
    'spread2': 'spread2',
    'D2': 'd2',
    'PPE': 'ppe',
}

FEATURE_GROUPS = [
    {
        'title': 'Frequency Features',
        'description': 'Basic pitch measurements from voice analysis.',
        'fields': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'],
    },
    {
        'title': 'Jitter Measures',
        'description': 'Microvariations in frequency used to assess voice stability.',
        'fields': ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP'],
    },
    {
        'title': 'Shimmer Measures',
        'description': 'Microvariations in amplitude used to assess vocal tremor and breathiness.',
        'fields': ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'],
    },
    {
        'title': 'Noise and Complexity',
        'description': 'Nonlinear measurements that capture voice signal irregularity.',
        'fields': ['NHR', 'HNR', 'RPDE', 'DFA'],
    },
    {
        'title': 'Spectral Spread and Pitch',
        'description': 'Additional spectral and nonlinear voice features.',
        'fields': ['spread1', 'spread2', 'D2', 'PPE'],
    },
]

FEATURE_DESCRIPTIONS = {
    'MDVP:Fo(Hz)': 'Average fundamental frequency in Hz.',
    'MDVP:Fhi(Hz)': 'Maximum fundamental frequency in Hz.',
    'MDVP:Flo(Hz)': 'Minimum fundamental frequency in Hz.',
    'MDVP:Jitter(%)': 'Relative variation in pitch.',
    'MDVP:Jitter(Abs)': 'Absolute variation in pitch.',
    'MDVP:RAP': 'Relative amplitude perturbation.',
    'MDVP:PPQ': 'Five-point period perturbation.',
    'Jitter:DDP': 'Difference of differences of periods.',
    'MDVP:Shimmer': 'Relative variation in amplitude.',
    'MDVP:Shimmer(dB)': 'Amplitude variation in decibels.',
    'Shimmer:APQ3': 'Three-point amplitude perturbation.',
    'Shimmer:APQ5': 'Five-point amplitude perturbation.',
    'MDVP:APQ': 'Average amplitude perturbation.',
    'Shimmer:DDA': 'Difference of differences of amplitude.',
    'NHR': 'Noise-to-harmonic ratio.',
    'HNR': 'Harmonics-to-noise ratio.',
    'RPDE': 'Recurrence period density entropy.',
    'DFA': 'Detrended fluctuation analysis value.',
    'spread1': 'Nonlinear vocal spread measure.',
    'spread2': 'Spectral spread feature.',
    'D2': 'Correlation dimension approximation.',
    'PPE': 'Pitch period entropy.',
}


def load_feature_names():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature names file not found: {FEATURES_PATH}")
    with FEATURES_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model file not found: {MODEL_PATH}")
    return load(MODEL_PATH)


def home(request):
    return render(request, 'prediction/home.html', {
        'title': 'ParkinsonAI',
    })


def predict(request):
    form = PredictionForm(request.POST or None)
    prediction = None
    probability = None
    save_error = None

    if request.method == 'POST' and form.is_valid():
        feature_names = load_feature_names()
        input_vector = [[form.cleaned_data[name] for name in feature_names]]
        model = load_model()
        prediction_value = model.predict(input_vector)[0]
        prediction = 'Parkinson\'s Disease' if int(prediction_value) == 1 else 'No Parkinson\'s Disease'
        entry_name = form.cleaned_data['entry_name']

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_vector)[0]
            probability = round(float(max(proba)) * 100, 2)

        try:
            prediction_record = Prediction(
                entry_name=entry_name,
                prediction_result=prediction,
                probability=probability,
                **{
                    FEATURE_NAME_MAP[name]: form.cleaned_data[name]
                    for name in feature_names
                }
            )
            prediction_record.save()
        except DatabaseError as error:
            save_error = 'Unable to save history right now. Please try again later.'
            print(f"Prediction save error: {error}")

    return render(request, 'prediction/predict.html', {
        'title': 'Predict',
        'form': form,
        'prediction': prediction,
        'probability': probability,
        'save_error': save_error,
        'field_groups': FEATURE_GROUPS,
        'feature_descriptions': FEATURE_DESCRIPTIONS,
    })


def history(request):
    save_error = None
    predictions = []
    total_predictions = 0
    parkinson_predictions = 0
    non_parkinson_predictions = 0

    try:
        predictions = Prediction.objects.order_by('-created_at')[:50]
        total_predictions = Prediction.objects.count()
        parkinson_predictions = Prediction.objects.filter(prediction_result__icontains='Parkinson').count()
        non_parkinson_predictions = total_predictions - parkinson_predictions
    except DatabaseError as error:
        save_error = 'Unable to load prediction history right now.'
        print(f"History load error: {error}")

    return render(request, 'prediction/history.html', {
        'title': 'History',
        'predictions': predictions,
        'total_predictions': total_predictions,
        'parkinson_predictions': parkinson_predictions,
        'non_parkinson_predictions': non_parkinson_predictions,
        'error': save_error,
    })


def history_detail(request, pk):
    prediction_record = get_object_or_404(Prediction, pk=pk)
    feature_names = load_feature_names()
    feature_data = [
        (name, getattr(prediction_record, FEATURE_NAME_MAP[name]))
        for name in feature_names
    ]
    return render(request, 'prediction/history_detail.html', {
        'title': 'History Detail',
        'record': prediction_record,
        'feature_data': feature_data,
    })


def about(request):
    return render(request, 'prediction/about.html', {
        'title': 'About Model',
    })
