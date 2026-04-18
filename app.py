"""
HousePrice Prediction – Main Flask Application
"""
import os
import json
import csv
import io
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file)

from database import db, User, Property, Prediction, MLModel, AppLog
import ml_service

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "houseprice_secret_2024_xK9p"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///houseprice.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ─── Auth Helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("user_login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin_id" not in session:
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


def log_action(level, message):
    try:
        entry = AppLog(level=level, message=message)
        db.session.add(entry)
        db.session.commit()
    except Exception:
        pass


# ─── DB Initialise ────────────────────────────────────────────────────────────

def init_db():
    """Create tables, seed admin, load CSV, register models in DB."""
    db.create_all()

    # Create admin if not exists
    if not User.query.filter_by(email="admin@houseprice.com").first():
        admin = User(name="Admin", email="admin@houseprice.com", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()

    # Seed properties from CSV (only if table empty)
    if Property.query.count() == 0:
        csv_path = "data/houseprice_properties.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                p = Property(
                    location=row["location"],
                    area_sqft=float(row["area_sqft"]),
                    bedrooms=int(row["bedrooms"]),
                    bathrooms=int(row["bathrooms"]),
                    property_type=row["property_type"],
                    age_years=int(row["age_years"]),
                    floor=int(row["floor"]),
                    parking=int(row["parking"]),
                    furnished=row["furnished"],
                    price_lakhs=float(row["price_lakhs"]),
                )
                db.session.add(p)
            db.session.commit()

    # Register trained models if not in DB
    model_files = {
        "Linear Regression":  "models/linear_regression.pkl",
        "Random Forest":      "models/random_forest.pkl",
        "Gradient Boosting":  "models/gradient_boosting.pkl",
        "XGBoost":            "models/xgboost.pkl",
    }
    metrics_map = {m["name"]: m for m in ml_service.get_all_metrics()}

    for name, path in model_files.items():
        if os.path.exists(path) and not MLModel.query.filter_by(name=name).first():
            m = metrics_map.get(name, {})
            mlm = MLModel(
                name=name, file_path=path,
                mae=m.get("mae"), mse=m.get("mse"),
                rmse=m.get("rmse"), r2=m.get("r2"),
                is_active=False,
            )
            db.session.add(mlm)
    db.session.commit()

    # Activate best model
    if not MLModel.query.filter_by(is_active=True).first():
        best_info = ml_service.get_best_model_info()
        if best_info:
            best = MLModel.query.filter_by(name=best_info["name"]).first()
            if best:
                best.is_active = True
                db.session.commit()

    # Load active model into ml_service
    active = MLModel.query.filter_by(is_active=True).first()
    if active and os.path.exists(active.file_path):
        ml_service.load_model(active.file_path, active.name)


# ─────────────────────────────────────────────────────────────────────────────
#  USER ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = {
        "properties": Property.query.count(),
        "predictions": Prediction.query.count(),
        "users": User.query.filter_by(is_admin=False).count(),
    }
    active_model = MLModel.query.filter_by(is_active=True).first()
    return render_template("user/index.html", stats=stats, active_model=active_model)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")

        if not name or not email or not pwd:
            flash("All fields are required.", "danger")
            return render_template("user/register.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("user/register.html")

        user = User(name=name, email=email)
        user.set_password(pwd)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("user_login"))

    return render_template("user/register.html")


@app.route("/login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        user  = User.query.filter_by(email=email, is_admin=False).first()

        if user and user.check_password(pwd):
            if user.is_blocked:
                flash("Your account is blocked. Contact admin.", "danger")
                return render_template("user/login.html")
            session["user_id"]   = user.id
            session["user_name"] = user.name
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for("predict"))
        flash("Invalid credentials.", "danger")

    return render_template("user/login.html")


@app.route("/logout")
def user_logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "location":      request.form.get("location"),
                "area_sqft":     float(request.form.get("area_sqft", 0)),
                "bedrooms":      int(request.form.get("bedrooms", 1)),
                "bathrooms":     int(request.form.get("bathrooms", 1)),
                "property_type": request.form.get("property_type"),
                "age_years":     int(request.form.get("age_years", 0)),
                "floor":         int(request.form.get("floor", 0)),
                "parking":       int(request.form.get("parking", 0)),
                "furnished":     request.form.get("furnished"),
            }

            price = ml_service.predict(form_data)
            model_used = ml_service.get_active_model_name()

            pred = Prediction(
                user_id=session.get("user_id"),
                predicted_price=price,
                model_used=model_used,
                **{k: form_data[k] for k in form_data},
            )
            db.session.add(pred)
            db.session.commit()

            # Find similar properties
            similar = Property.query.filter(
                Property.location == form_data["location"],
                Property.bedrooms == form_data["bedrooms"],
            ).limit(3).all()

            result = {
                "price": price,
                "model": model_used,
                "similar": similar,
            }
            log_action("INFO", f"Prediction: {form_data['location']} {form_data['bedrooms']}BHK → ₹{price}L")

        except Exception as e:
            flash(f"Prediction failed: {str(e)}", "danger")
            log_action("ERROR", str(e))

    return render_template(
        "user/predict.html",
        locations=ml_service.LOCATIONS,
        property_types=ml_service.PROPERTY_TYPES,
        furnished_opts=ml_service.FURNISHED_OPTS,
        result=result,
        form_data=form_data,
    )


@app.route("/history")
@login_required
def history():
    preds = (Prediction.query
             .filter_by(user_id=session["user_id"])
             .order_by(Prediction.created_at.desc())
             .limit(50).all())
    return render_template("user/history.html", predictions=preds)


# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        admin = User.query.filter_by(email=email, is_admin=True).first()
        if admin and admin.check_password(pwd):
            session["admin_id"]   = admin.id
            session["admin_name"] = admin.name
            log_action("INFO", f"Admin login: {email}")
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "danger")

    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_id", None)
    session.pop("admin_name", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    total_users       = User.query.filter_by(is_admin=False).count()
    total_properties  = Property.query.count()
    total_predictions = Prediction.query.count()
    avg_price         = db.session.query(db.func.avg(Property.price_lakhs)).scalar() or 0
    active_model      = MLModel.query.filter_by(is_active=True).first()

    # Chart: predictions per day (last 7 days)
    today = datetime.utcnow().date()
    daily_labels, daily_data = [], []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Prediction.query.filter(
            db.func.date(Prediction.created_at) == str(day)
        ).count()
        daily_labels.append(day.strftime("%d %b"))
        daily_data.append(count)

    # Chart: properties per location
    loc_rows = (db.session.query(Property.location,
                                  db.func.count(Property.id))
                .group_by(Property.location).all())
    loc_labels = [r[0] for r in loc_rows]
    loc_data   = [r[1] for r in loc_rows]

    # Recent logs
    logs = AppLog.query.order_by(AppLog.created_at.desc()).limit(10).all()

    return render_template("admin/dashboard.html",
        total_users=total_users, total_properties=total_properties,
        total_predictions=total_predictions,
        avg_price=round(avg_price, 2),
        active_model=active_model,
        daily_labels=json.dumps(daily_labels),
        daily_data=json.dumps(daily_data),
        loc_labels=json.dumps(loc_labels),
        loc_data=json.dumps(loc_data),
        logs=logs,
    )


# ── Properties ────────────────────────────────────────────────────────────────

@app.route("/admin/properties")
@admin_required
def admin_properties():
    page     = request.args.get("page", 1, type=int)
    search   = request.args.get("search", "")
    query    = Property.query
    if search:
        query = query.filter(Property.location.ilike(f"%{search}%"))
    props    = query.order_by(Property.id.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template("admin/properties.html", props=props, search=search,
                           locations=ml_service.LOCATIONS,
                           property_types=ml_service.PROPERTY_TYPES,
                           furnished_opts=ml_service.FURNISHED_OPTS)


@app.route("/admin/properties/add", methods=["POST"])
@admin_required
def admin_add_property():
    try:
        p = Property(
            location=request.form["location"],
            area_sqft=float(request.form["area_sqft"]),
            bedrooms=int(request.form["bedrooms"]),
            bathrooms=int(request.form["bathrooms"]),
            property_type=request.form["property_type"],
            age_years=int(request.form.get("age_years", 0)),
            floor=int(request.form.get("floor", 0)),
            parking=int(request.form.get("parking", 0)),
            furnished=request.form["furnished"],
            price_lakhs=float(request.form["price_lakhs"]),
        )
        db.session.add(p)
        db.session.commit()
        flash("Property added successfully.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("admin_properties"))


@app.route("/admin/properties/delete/<int:pid>", methods=["POST"])
@admin_required
def admin_delete_property(pid):
    p = Property.query.get_or_404(pid)
    db.session.delete(p)
    db.session.commit()
    flash("Property deleted.", "info")
    return redirect(url_for("admin_properties"))


@app.route("/admin/properties/upload", methods=["POST"])
@admin_required
def admin_upload_csv():
    f = request.files.get("csv_file")
    if not f or not f.filename.endswith(".csv"):
        flash("Please upload a valid .csv file.", "danger")
        return redirect(url_for("admin_properties"))
    try:
        df = pd.read_csv(f)
        required = {"location","area_sqft","bedrooms","bathrooms",
                    "property_type","age_years","floor","parking","furnished","price_lakhs"}
        if not required.issubset(df.columns):
            flash(f"CSV must have columns: {required}", "danger")
            return redirect(url_for("admin_properties"))
        count = 0
        for _, row in df.iterrows():
            p = Property(**{col: row[col] for col in required})
            db.session.add(p)
            count += 1
        db.session.commit()
        flash(f"Uploaded {count} properties.", "success")
    except Exception as e:
        flash(f"Upload failed: {e}", "danger")
    return redirect(url_for("admin_properties"))


@app.route("/admin/properties/export")
@admin_required
def admin_export_csv():
    props = Property.query.all()
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["id","location","area_sqft","bedrooms","bathrooms",
                     "property_type","age_years","floor","parking","furnished","price_lakhs"])
    for p in props:
        writer.writerow([p.id, p.location, p.area_sqft, p.bedrooms, p.bathrooms,
                         p.property_type, p.age_years, p.floor, p.parking,
                         p.furnished, p.price_lakhs])
    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv",
                     download_name="houseprice_properties.csv", as_attachment=True)


# ── Users ─────────────────────────────────────────────────────────────────────

@app.route("/admin/users")
@admin_required
def admin_users():
    users = User.query.filter_by(is_admin=False).order_by(User.created_at.desc()).all()
    for u in users:
        u.pred_count = Prediction.query.filter_by(user_id=u.id).count()
    return render_template("admin/users.html", users=users)


@app.route("/admin/users/toggle/<int:uid>", methods=["POST"])
@admin_required
def admin_toggle_user(uid):
    u = User.query.get_or_404(uid)
    u.is_blocked = not u.is_blocked
    db.session.commit()
    status = "blocked" if u.is_blocked else "unblocked"
    flash(f"User {u.name} {status}.", "info")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/delete/<int:uid>", methods=["POST"])
@admin_required
def admin_delete_user(uid):
    u = User.query.get_or_404(uid)
    Prediction.query.filter_by(user_id=uid).update({"user_id": None})
    db.session.delete(u)
    db.session.commit()
    flash("User deleted.", "info")
    return redirect(url_for("admin_users"))


# ── ML Models ─────────────────────────────────────────────────────────────────

@app.route("/admin/models")
@admin_required
def admin_models():
    models = MLModel.query.order_by(MLModel.r2.desc()).all()
    logs   = AppLog.query.order_by(AppLog.created_at.desc()).limit(20).all()
    return render_template("admin/models.html", models=models, logs=logs)


@app.route("/admin/models/activate/<int:mid>", methods=["POST"])
@admin_required
def admin_activate_model(mid):
    MLModel.query.update({"is_active": False})
    m = MLModel.query.get_or_404(mid)
    m.is_active = True
    db.session.commit()
    if os.path.exists(m.file_path):
        ml_service.load_model(m.file_path, m.name)
        flash(f"'{m.name}' is now active.", "success")
        log_action("INFO", f"Switched active model to: {m.name}")
    else:
        flash("Model file not found.", "danger")
    return redirect(url_for("admin_models"))


# ── Analytics API ─────────────────────────────────────────────────────────────

@app.route("/admin/analytics")
@admin_required
def admin_analytics():
    # Avg price per location
    loc_price = (db.session.query(Property.location,
                                   db.func.avg(Property.price_lakhs))
                 .group_by(Property.location)
                 .order_by(db.func.avg(Property.price_lakhs).desc()).all())

    # BHK distribution
    bhk_dist = (db.session.query(Property.bedrooms,
                                   db.func.count(Property.id))
                .group_by(Property.bedrooms).all())

    # Property type split
    type_dist = (db.session.query(Property.property_type,
                                    db.func.count(Property.id))
                 .group_by(Property.property_type).all())

    return render_template("admin/analytics.html",
        loc_labels=json.dumps([r[0] for r in loc_price]),
        loc_avg=json.dumps([round(r[1], 2) for r in loc_price]),
        bhk_labels=json.dumps([f"{r[0]} BHK" for r in bhk_dist]),
        bhk_data=json.dumps([r[1] for r in bhk_dist]),
        type_labels=json.dumps([r[0] for r in type_dist]),
        type_data=json.dumps([r[1] for r in type_dist]),
    )


# ── API endpoint for prediction history chart ─────────────────────────────────

@app.route("/api/recent-predictions")
@admin_required
def api_recent_predictions():
    preds = (Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all())
    data  = [{"price": p.predicted_price, "location": p.location,
              "date": p.created_at.strftime("%d %b")} for p in preds]
    return jsonify(data)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
