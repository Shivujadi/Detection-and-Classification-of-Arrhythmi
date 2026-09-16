"""
Flask Application Test Suite for Phase 6.
Tests routes, template rendering, JSON prediction API, invalid input handling,
and model loading using Flask test client.
"""

import sys
import os
import json
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app
from src.data_loader import AAMI_CLASS_NAMES


class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_home_page(self):
        """Test GET / route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Detection and Classification of Arrhythmia", response.data)

    def test_about_page(self):
        """Test GET /about route."""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MIT-BIH Arrhythmia Database", response.data)

    def test_models_page(self):
        """Test GET /models route."""
        response = self.client.get('/models')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Weighted KNN", response.data)

    def test_visualizations_page(self):
        """Test GET /visualizations route."""
        response = self.client.get('/visualizations')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"raw_ecg_waveform.png", response.data)

    def test_predict_page_get(self):
        """Test GET /predict route."""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ECG Heartbeat Arrhythmia Classification Form", response.data)

    def test_sample_beats_api(self):
        """Test GET /api/sample-beats route."""
        response = self.client.get('/api/sample-beats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('N', data['samples'])
        self.assertEqual(len(data['samples']['N']['signal']), 180)

    def test_valid_prediction_api(self):
        """Test POST /predict with valid 180-sample vector."""
        sample_vector = [0.5] * 180
        response = self.client.post(
            '/predict',
            data=json.dumps({'signal_vector': sample_vector, 'model_name': 'best_model.pkl'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn(data['result']['predicted_class'], ['N', 'S', 'V', 'F', 'Q'])

    def test_invalid_length_prediction_api(self):
        """Test POST /predict with invalid sample vector length."""
        invalid_vector = [0.5] * 50  # Only 50 samples
        response = self.client.post(
            '/predict',
            data=json.dumps({'signal_vector': invalid_vector}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn('180 sample points', data['message'])


if __name__ == '__main__':
    print("=" * 60)
    print("        RUNNING FLASK APPLICATION TEST SUITE")
    print("=" * 60)
    unittest.main()
