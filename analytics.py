import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.cluster import KMeans
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("/home/claude/bug_tracker/data/bugs.csv")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['resolved_at'] = pd.to_datetime(df['resolved_at'])
    return df

def severity_prediction_model(df):
    """Predict bug severity using Random Forest"""
    le_comp = LabelEncoder()
    le_env = LabelEncoder()
    le_assign = LabelEncoder()
    
    features = df.copy()
    features['component_enc'] = le_comp.fit_transform(features['component'])
    features['env_enc'] = le_env.fit_transform(features['environment'])
    features['assignee_enc'] = le_assign.fit_transform(features['assignee'])
    features['severity_enc'] = LabelEncoder().fit_transform(features['severity'])
    
    X = features[['component_enc', 'env_enc', 'assignee_enc', 'votes', 'comments', 'is_regression']]
    y = features['severity_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    feature_importances = dict(zip(
        ['Component', 'Environment', 'Assignee', 'Votes', 'Comments', 'Is Regression'],
        model.feature_importances_.tolist()
    ))
    
    return {
        "model": "Random Forest Classifier",
        "accuracy": round(accuracy * 100, 2),
        "feature_importances": feature_importances,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

def resolution_time_model(df):
    """Predict resolution time using Gradient Boosting"""
    resolved = df.dropna(subset=['resolution_days']).copy()
    
    le = LabelEncoder()
    resolved['severity_enc'] = le.fit_transform(resolved['severity'])
    resolved['component_enc'] = le.fit_transform(resolved['component'])
    resolved['env_enc'] = le.fit_transform(resolved['environment'])
    
    X = resolved[['severity_enc', 'component_enc', 'env_enc', 'votes', 'comments', 'is_regression']]
    y = resolved['resolution_days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        "model": "Gradient Boosting Regressor",
        "mae_days": round(mae, 2),
        "avg_resolution_days": round(resolved['resolution_days'].mean(), 2),
        "by_severity": resolved.groupby('severity')['resolution_days'].mean().round(2).to_dict()
    }

def bug_clustering(df):
    """Cluster bugs by behavior patterns using KMeans"""
    le = LabelEncoder()
    features = pd.DataFrame({
        'severity_enc': le.fit_transform(df['severity']),
        'component_enc': le.fit_transform(df['component']),
        'votes': df['votes'],
        'comments': df['comments'],
        'is_regression': df['is_regression'].astype(int)
    })
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    df = df.copy()
    df['cluster'] = clusters
    
    cluster_labels = ["Performance Issues", "Security Bugs", "UI/UX Defects", "Data Integrity"]
    cluster_counts = df['cluster'].value_counts().to_dict()
    
    return {
        "model": "KMeans Clustering",
        "n_clusters": 4,
        "cluster_labels": cluster_labels,
        "cluster_distribution": {cluster_labels[k]: v for k, v in cluster_counts.items()}
    }

def get_analytics_summary(df):
    total = len(df)
    open_bugs = len(df[df['status'] == 'Open'])
    critical = len(df[df['severity'] == 'Critical'])
    resolved_dates = df['resolved_at'].dropna()
    max_date = resolved_dates.dt.date.max() if len(resolved_dates) > 0 else None
    resolved_today = len(df[(df['status'] == 'Resolved') & (df['resolved_at'].dt.date == max_date)]) if max_date else 0
    
    monthly = df.groupby(df['created_at'].dt.to_period('M')).size().reset_index()
    monthly.columns = ['month', 'count']
    monthly['month'] = monthly['month'].astype(str)
    
    by_component = df.groupby('component').size().reset_index(name='count').sort_values('count', ascending=False)
    by_severity = df.groupby('severity').size().reset_index(name='count')
    by_status = df.groupby('status').size().reset_index(name='count')
    by_assignee = df.groupby('assignee').size().reset_index(name='count').sort_values('count', ascending=False)
    
    resolution_rate = len(df[df['status'].isin(['Resolved', 'Closed'])]) / total * 100
    
    return {
        "summary": {
            "total": total, "open": open_bugs, "critical": critical,
            "resolved_today": resolved_today,
            "resolution_rate": round(resolution_rate, 1),
            "avg_resolution_days": round(df['resolution_days'].dropna().mean(), 1)
        },
        "monthly_trend": monthly.to_dict(orient='records'),
        "by_component": by_component.to_dict(orient='records'),
        "by_severity": by_severity.to_dict(orient='records'),
        "by_status": by_status.to_dict(orient='records'),
        "by_assignee": by_assignee.to_dict(orient='records'),
        "recent_bugs": df.sort_values('created_at', ascending=False).head(10)[
            ['id', 'title', 'component', 'severity', 'status', 'assignee', 'created_at']
        ].to_dict(orient='records')
    }

def run_all_analytics():
    df = load_data()
    analytics = get_analytics_summary(df)
    analytics['ml_severity'] = severity_prediction_model(df)
    analytics['ml_resolution'] = resolution_time_model(df)
    analytics['ml_clustering'] = bug_clustering(df)
    return analytics

if __name__ == "__main__":
    results = run_all_analytics()
    with open("/home/claude/bug_tracker/data/analytics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Analytics generated successfully!")
    print(f"Total bugs: {results['summary']['total']}")
    print(f"ML Accuracy: {results['ml_severity']['accuracy']}%")
    print(f"Avg Resolution: {results['ml_resolution']['avg_resolution_days']} days")
