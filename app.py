from flask import Flask, request, jsonify, render_template, send_from_directory, session
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import chardet
import base64
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
from werkzeug.utils import secure_filename
warnings.filterwarnings('ignore')

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define blue color scheme
BLUE_PALETTE = {
    'primary': '#0047AB',
    'secondary': '#4682B4',
    'accent': '#00BFFF',
    'light': '#E6F2FF',
    'dark': '#00008B'
}

app = Flask(__name__)
app.secret_key = 'data_analyzer_secret_key'

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle pandas Timestamp objects
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_read_file(file_path):
    """Read files with proper encoding and Excel engine detection"""
    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            encoding = result['encoding'] or 'utf-8'
            
            return pd.read_csv(
                file_path,
                encoding=encoding,
                engine='python',
                on_bad_lines='skip',
                parse_dates=True,
                dtype_backend='pyarrow'
            )
        else:
            engine = 'openpyxl' if file_path.endswith(('.xlsx', '.xlsm')) else 'xlrd'
            return pd.read_excel(
                file_path,
                engine=engine,
                # Removed the infer_datetime_format parameter
                parse_dates=True,
                dtype_backend='pyarrow'
            )
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def clean_dataframe(df):
    """Comprehensive data cleaning and preprocessing"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert object columns to string
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype('string')
    
    # Convert datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
    
    cat_cols = df.select_dtypes(include=['string', 'category']).columns
    for col in cat_cols:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)
        
        df = safe_read_file(file_path)
        df = clean_dataframe(df)
        
        cleaned_path = file_path + '_cleaned.parquet'
        df.to_parquet(cleaned_path)
        session['current_file_path'] = cleaned_path
        
        result = analyze_data(df)
        result['data_preview'] = {
            'columns': df.columns.tolist(),
            'head': df.head(10).replace({np.nan: None}).to_dict('records'),
            'total_rows': len(df)
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_data', methods=['GET'])
def get_data():
    if 'current_file_path' not in session:
        return jsonify({'error': 'No file uploaded'})
    
    file_path = session['current_file_path']
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})
    
    try:
        df = pd.read_parquet(file_path)
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        
        # Convert datetime columns to string before serialization
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']).columns:
            df_copy[col] = df_copy[col].astype(str)
        
        return jsonify({
            'columns': df.columns.tolist(),
            'data': df_copy.iloc[start_idx:end_idx].replace({np.nan: None}).to_dict('records'),
            'total_rows': len(df),
            'total_pages': (len(df) + page_size - 1) // page_size,
            'current_page': page
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def analyze_data(df):
    """Main analysis function"""
    result = {
        'data_profile': generate_data_profile(df),
        'outliers': detect_outliers(df),
        'correlations': analyze_correlations(df),
        'visualizations': generate_visualizations(df),
        'insights': generate_insights(df),
        'cluster_analysis': perform_clustering(df) if len(df) > 10 else None
    }
    return result

def generate_data_profile(df):
    """Generate dataset profile"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['string', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    profile = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'column_types': {
            'numeric': len(numeric_columns),
            'categorical': len(categorical_columns),
            'datetime': len(datetime_columns)
        },
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    column_stats = {}
    for col in df.columns:
        col_stats = {'type': str(df[col].dtype)}
        
        if df[col].dtype.kind in 'iufc':
            col_stats.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'unique_values': int(df[col].nunique()),
                'missing_rate': float(df[col].isna().mean())
            })
        else:
            col_stats.update({
                'unique_values': int(df[col].nunique()),
                'top_values': {str(k): v for k, v in df[col].value_counts().head(5).to_dict().items()},
                'missing_rate': float(df[col].isna().mean())
            })
            
        column_stats[col] = col_stats
    
    profile['column_stats'] = column_stats
    return profile

def detect_outliers(df):
    """Outlier detection implementation"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 1:
        return {'message': 'No numeric columns available for outlier detection'}
    
    numeric_df = numeric_df.fillna(numeric_df.median())
    outliers = {}
    
    # Z-score method
    z_score_outliers = {}
    for col in numeric_df.columns:
        z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
        z_outliers = df.index[z_scores > 3].tolist()
        if z_outliers:
            z_score_outliers[col] = {
                'count': len(z_outliers),
                'indices': z_outliers[:10],
                'values': numeric_df.loc[z_outliers, col].tolist()[:10]
            }
    
    # IQR method
    iqr_outliers = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = float(Q1 - 1.5 * IQR)
        upper_bound = float(Q3 + 1.5 * IQR)
       
        outliers_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
        iqr_outlier_indices = df.index[outliers_mask].tolist()
        
        if iqr_outlier_indices:
            iqr_outliers[col] = {
                'count': len(iqr_outlier_indices),
                'indices': iqr_outlier_indices[:10],
                'values': [float(v) for v in numeric_df.loc[iqr_outlier_indices, col].tolist()[:10]],
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
    
    # Isolation Forest
    iso_forest = None
    if numeric_df.shape[1] >= 2 and len(numeric_df) > 10:
        try:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            iso_forest.fit(numeric_df)
            scores = iso_forest.decision_function(numeric_df)
            outlier_mask = iso_forest.predict(numeric_df) == -1
            iso_outlier_indices = df.index[outlier_mask].tolist()
            
            if iso_outlier_indices:
                iso_outliers = {
                    'count': len(iso_outlier_indices),
                    'indices': iso_outlier_indices[:10],
                    'score_quantiles': {
                        '10%': float(np.quantile(scores, 0.1)),
                        '25%': float(np.quantile(scores, 0.25)),
                        '50%': float(np.quantile(scores, 0.5)),
                        '75%': float(np.quantile(scores, 0.75)),
                        '90%': float(np.quantile(scores, 0.9))
                    }
                }
            else:
                iso_outliers = {'count': 0}
        except:
            iso_outliers = {'error': 'Failed to run Isolation Forest'}
    else:
        iso_outliers = {'message': 'Not enough data for Isolation Forest'}
    
    outliers['z_score'] = z_score_outliers
    outliers['iqr'] = iqr_outliers
    outliers['isolation_forest'] = iso_outliers
    
    return outliers

def analyze_correlations(df):
    """Correlation analysis with Plotly"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return {'message': 'Not enough numeric columns for correlation analysis'}
    
    corr_matrix = numeric_df.corr().round(3)
    
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            if not np.isnan(corr):
                corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': float(corr)
                })
    
    corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Fix: Added closing parenthesis for Heatmap and Figure
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Blues',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=450,
        width=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return {
        'pairs': corr_pairs[:10],
        'heatmap_plotly': fig.to_json()
    }

def generate_visualizations(df):
    """Generate Plotly visualizations"""
    visualizations = []
    
    # Numerical distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:5]:
            fig = make_subplots(rows=1, cols=1)
            hist_data = df[col].dropna()
            
            fig.add_trace(
                go.Histogram(
                    x=hist_data,
                    name='Histogram',
                    marker_color=BLUE_PALETTE['primary'],
                    opacity=0.7,
                    nbinsx=30,
                    histnorm='probability density'
                )
            )
            
            from scipy import stats
            if len(hist_data) > 1:
                kde_x = np.linspace(min(hist_data), max(hist_data), 100)
                kde = stats.gaussian_kde(hist_data)
                kde_y = kde(kde_x)
                
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode='lines',
                        name='Density',
                        line=dict(color=BLUE_PALETTE['dark'], width=2)
                    )
                )
            
            fig.update_layout(
                title=f'Distribution of {col}',
                height=350,
                width=500,
                template='plotly_white',
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            visualizations.append({
                'title': f'Distribution of {col}',
                'type': 'histogram',
                'plotly_figure': fig.to_json()
            })
    
    # Categorical value counts
    cat_cols = df.select_dtypes(include=['string', 'category']).columns.tolist()
    if cat_cols:
        for col in cat_cols[:3]:
            if df[col].nunique() < 15:
                value_counts = df[col].value_counts().head(10)
                
                fig = go.Figure(data=go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color=BLUE_PALETTE['secondary']
                ))
                
                fig.update_layout(
                    title=f'Top values in {col}',
                    height=350,
                    width=500,
                    template='plotly_white',
                    xaxis_tickangle=-45,
                    margin=dict(l=40, r=40, t=50, b=70)
                )
                
                visualizations.append({
                    'title': f'Value Counts for {col}',
                    'type': 'bar',
                    'plotly_figure': fig.to_json()
                })
    
    # Scatter plot for top correlated features
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        max_corr_idx = np.argmax(corr_matrix.values)
        i, j = np.unravel_index(max_corr_idx, corr_matrix.shape)
        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
        
        fig = px.scatter(
            df, 
            x=col1, 
            y=col2, 
            color_discrete_sequence=[BLUE_PALETTE['accent']],
            opacity=0.7,
            title=f'Scatter Plot: {col1} vs {col2}'
        )
        
        fig.update_layout(
            height=350,
            width=500,
            template='plotly_white',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        visualizations.append({
            'title': f'Relationship between {col1} and {col2}',
            'type': 'scatter',
            'plotly_figure': fig.to_json()
        })

    
    return visualizations





def perform_clustering(df):
    """Clustering analysis with Plotly"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2 or len(numeric_df) < 20:
        return {'message': 'Not enough numerical data for clustering'}
    
    numeric_df = numeric_df.fillna(numeric_df.median())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    result = {}
    
    # Elbow method
    inertia = []
    k_range = range(2, min(10, len(numeric_df) // 5 + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    fig = go.Figure(data=go.Scatter(
        x=list(k_range), 
        y=inertia, 
        mode='lines+markers',
        marker=dict(color=BLUE_PALETTE['primary'], size=8),
        line=dict(color=BLUE_PALETTE['primary'], width=2)
    ))
    
    fig.update_layout(
        title='Elbow Method for Optimal k',
        height=350,
        width=500,
        template='plotly_white',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    result['elbow_curve_plotly'] = fig.to_json()
    
    # Optimal clusters detection
    from scipy.signal import argrelextrema
    n_clusters = 3
    
    if len(inertia) > 3:
        diffs = np.diff(inertia)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 1:
            inflection_points = argrelextrema(np.array(second_diffs), np.less)[0]
            if inflection_points.size > 0:
                n_clusters = k_range[inflection_points[0] + 2]
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # PCA visualization
    if len(numeric_df.columns) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        cluster_df_pca = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        fig = px.scatter(
            cluster_df_pca, 
            x='PCA1', 
            y='PCA2', 
            color='Cluster',
            color_continuous_scale='Blues',
            title=f'Cluster Visualization using PCA (K={n_clusters})'
        )
        
        fig.update_layout(
            height=450,
            width=600,
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        result['cluster_visualization_plotly'] = fig.to_json()
        result['pca_variance'] = {
            'pc1': float(pca.explained_variance_ratio_[0]),
            'pc2': float(pca.explained_variance_ratio_[1])
        }
    
    # Cluster statistics
    cluster_df = numeric_df.copy()
    cluster_df['cluster'] = clusters
    
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_data = cluster_df[cluster_df['cluster'] == i].drop('cluster', axis=1)
        cluster_stats[f'Cluster_{i}'] = {
            'size': int(len(cluster_data)),
            'percentage': float(round(len(cluster_data) / len(cluster_df) * 100, 2)),
            'mean': {k: float(v) for k, v in cluster_data.mean().items()}
        }
    
    result['optimal_clusters'] = n_clusters
    result['cluster_stats'] = cluster_stats
    
    return result

def generate_insights(df):
    """Generate automated insights"""
    insights = []
    
    # Dataset size insights
    if len(df) < 10:
        insights.append({
            'type': 'warning',
            'message': 'The dataset is very small (less than 10 records). Results may not be statistically significant.'
        })
    elif len(df) > 10000:
        insights.append({
            'type': 'info',
            'message': f'Large dataset with {len(df)} records - analysis may take longer but should be more reliable.'
        })
    
    # Missing values
    missing_counts = df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        if (missing_cols / len(df) > 0.5).any():
            insights.append({
                'type': 'warning',
                'message': f'Some columns have more than 50% missing values: {missing_cols[missing_cols / len(df) > 0.5].index.tolist()}'})
        else:
            insights.append({
                'type': 'info',
                'message': f'{len(missing_cols)} columns have missing values. Consider imputation for better analysis.'})
    else:
        insights.append({
            'type': 'success',
            'message': 'No missing values found in the dataset - good data quality!'})
    
    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['string', 'category']).columns.tolist()
    
    insights.append({
        'type': 'info',
        'message': f'Dataset has {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.'})
    
    # Numeric distributions
    for col in numeric_cols[:5]:  # Check first 5 numeric columns
        # Convert to numpy dtype for skew calculation
        col_data = df[col].astype('float64')
        skew = col_data.skew()
        
        if abs(skew) > 1:
            skew_direction = 'right' if skew > 0 else 'left'
            insights.append({
                'type': 'info',
                'message': f"'{col}' has a {skew_direction}-skewed distribution (skew = {round(skew, 2)}). Consider transformation for analysis."
            })
    # Categorical insights
    for col in categorical_cols[:5]:
        unique_count = df[col].nunique()
        if unique_count == 1:
            insights.append({
                'type': 'warning',
                'message': f"'{col}' has only one unique value - may not be useful for analysis."})
        elif unique_count > 100:
            insights.append({
                'type': 'info',
                'message': f"'{col}' has high cardinality ({unique_count} unique values). Consider grouping or encoding."})
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        insights.append({
            'type': 'warning',
            'message': f'Found {dup_count} duplicate rows ({round(dup_count/len(df)*100, 2)}% of data). Consider removing duplicates.'})
    
    return insights

if __name__ == '__main__':
    app.run(debug=True, port=5000)