"""
Fake News Detector - Flask Web Application
Professional web interface with explainability
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from config import Config
from utils.model_utils import MultiDomainDetector
import os

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db = SQLAlchemy(app)

# Initialize multi-domain detector
detector = MultiDomainDetector()

# Database Models
class Prediction(db.Model):
    """Store prediction history"""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    domain = db.Column(db.String(20), default='general')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text[:100] + '...' if len(self.text) > 100 else self.text,
            'prediction': self.prediction,
            'confidence': round(self.confidence, 2),
            'domain': self.domain,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }


# Routes
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        domain = data.get('domain', 'auto')  # 'auto', 'general', 'political', 'health'
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) > app.config['MAX_TEXT_LENGTH']:
            return jsonify({'error': 'Text too long'}), 400
        
        # Make prediction with specified or auto-detected domain
        result = detector.predict(text, domain=domain)
        
        if result is None:
            return jsonify({'error': 'Prediction failed. Model not loaded.'}), 500
        
        # Save to database
        prediction = Prediction(
            text=text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            domain=result['domain']
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """Generate explanation for prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        domain = data.get('domain', 'auto')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate explanation
        explanation = detector.explain_prediction(text, domain=domain, num_features=10)
        
        if explanation is None:
            return jsonify({'error': 'Explanation generation failed'}), 500
        
        # Generate highlighted text
        highlighted_text = detector.get_word_highlighting(
            text, 
            explanation['important_words']
        )
        explanation['highlighted_text'] = highlighted_text
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """View prediction history"""
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).paginate(
        page=page, 
        per_page=app.config['PREDICTIONS_PER_PAGE'],
        error_out=False
    )
    
    return render_template('history.html', predictions=predictions)


@app.route('/statistics')
def statistics():
    """Show statistics dashboard"""
    total_predictions = Prediction.query.count()
    fake_count = Prediction.query.filter_by(prediction='Fake').count()
    real_count = Prediction.query.filter_by(prediction='Real').count()
    
    # Domain breakdown
    political_count = Prediction.query.filter_by(domain='political').count()
    health_count = Prediction.query.filter_by(domain='health').count()
    general_count = Prediction.query.filter_by(domain='general').count()
    
    # Average confidence
    predictions = Prediction.query.all()
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
    
    stats = {
        'total': total_predictions,
        'fake': fake_count,
        'real': real_count,
        'fake_percentage': (fake_count / total_predictions * 100) if total_predictions > 0 else 0,
        'real_percentage': (real_count / total_predictions * 100) if total_predictions > 0 else 0,
        'political': political_count,
        'health': health_count,
        'general': general_count,
        'avg_confidence': round(avg_confidence, 2)
    }
    
    return render_template('statistics.html', stats=stats)


# Add new route for model comparison
@app.route('/compare', methods=['POST'])
def compare():
    """Compare predictions across all domain models"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get predictions from all models
        results = detector.predict_all_domains(text)
        
        if not results:
            return jsonify({'error': 'No models available'}), 500
        
        return jsonify({
            'success': True,
            'results': results,
            'text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/api/health')
def api_health():
    """API health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': detector.loaded,
        'timestamp': datetime.utcnow().isoformat()
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500



# Initialize database and load models
def initialize():
    """Initialize database and load models"""
    with app.app_context():
        db.create_all()
    
    # Load all domain-specific models
    loaded = detector.load_all_models()
    
    if loaded:
        print(f"✓ Loaded models: {', '.join(loaded)}")
    else:
        print("⚠ No models loaded. Please train models first.")
        print("   Run: python train_domain_models.py")

# Call initialization
initialize()


if __name__ == '__main__':
    # Run app
    print("="*60)
    print("FAKE NEWS DETECTOR - WEB APPLICATION")
    print("="*60)
    print("\nStarting server...")
    print("✓ Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)