"""
Database models for the HousePrice Prediction System
"""
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    password    = db.Column(db.String(256), nullable=False)
    is_admin    = db.Column(db.Boolean, default=False)
    is_blocked  = db.Column(db.Boolean, default=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def set_password(self, raw):
        self.password = generate_password_hash(raw)

    def check_password(self, raw):
        return check_password_hash(self.password, raw)


class Property(db.Model):
    __tablename__ = "properties"
    id            = db.Column(db.Integer, primary_key=True)
    location      = db.Column(db.String(100), nullable=False)
    area_sqft     = db.Column(db.Float, nullable=False)
    bedrooms      = db.Column(db.Integer, nullable=False)
    bathrooms     = db.Column(db.Integer, nullable=False)
    property_type = db.Column(db.String(50), nullable=False)
    age_years     = db.Column(db.Integer, default=0)
    floor         = db.Column(db.Integer, default=0)
    parking       = db.Column(db.Integer, default=0)  # 0/1
    furnished     = db.Column(db.String(30), nullable=False)
    price_lakhs   = db.Column(db.Float, nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    __tablename__ = "predictions"
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    location        = db.Column(db.String(100))
    area_sqft       = db.Column(db.Float)
    bedrooms        = db.Column(db.Integer)
    bathrooms       = db.Column(db.Integer)
    property_type   = db.Column(db.String(50))
    age_years       = db.Column(db.Integer)
    floor           = db.Column(db.Integer)
    parking         = db.Column(db.Integer)
    furnished       = db.Column(db.String(30))
    predicted_price = db.Column(db.Float)
    model_used      = db.Column(db.String(50))
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


class MLModel(db.Model):
    __tablename__ = "ml_models"
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    file_path   = db.Column(db.String(200), nullable=False)
    mae         = db.Column(db.Float)
    mse         = db.Column(db.Float)
    rmse        = db.Column(db.Float)
    r2          = db.Column(db.Float)
    is_active   = db.Column(db.Boolean, default=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class AppLog(db.Model):
    __tablename__ = "app_logs"
    id         = db.Column(db.Integer, primary_key=True)
    level      = db.Column(db.String(20))   # INFO / WARNING / ERROR
    message    = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
