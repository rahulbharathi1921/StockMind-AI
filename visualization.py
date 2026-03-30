import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class StockVisualizationEngine:
    """Interactive visualization engine for stock intelligence"""

    def __init__(self):
        self.color_bullish = '#26a69a'  # Green
        self.color_bearish = '#ef5350'  # Red
        self.color_neutral = '#78909c'  # Gray
        self.color_primary = '#2196f3'  # Blue

    def create_main_chart(self, data: pd.DataFrame,
                          predictions: List[Dict] = None,
                          show_predictions: bool = True) -> go.Figure:
        """
        Create main interactive candlestick chart with predictions

        Args:
            data: OHLCV data
            predictions: List of prediction dictionaries
            show_predictions: Whether to show prediction overlays

        Returns:
            Plotly Figure object
        """
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Predictions', 'Volume', 'RSI')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.color_bullish,
                decreasing_line_color=self.color_bearish
            ),
            row=1, col=1
        )

        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )

        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )

        # Add prediction markers if available
        if show_predictions and predictions and len(predictions) > 0:
            self._add_prediction_overlays(fig, data, predictions)

        # Volume bars
        if 'Volume' in data.columns:
            colors = []
            for i in range(len(data)):
                if i < len(data) and 'Close' in data.columns and 'Open' in data.columns:
                    if data['Close'].iloc[i] >= data['Open'].iloc[i]:
                        colors.append(self.color_bullish)
                    else:
                        colors.append(self.color_bearish)
                else:
                    colors.append(self.color_neutral)

            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

        # RSI if available
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    opacity=0.8
                ),
                row=3, col=1
            )

            # Add RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          opacity=0.5, row=3, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1,
                          layer="below", line_width=0, row=3, col=1)

        # Update layout
        fig.update_layout(
            title='Stock Analysis Dashboard',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            showlegend=True,
            hovermode='x unified',
            height=800,
            xaxis_rangeslider_visible=False
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)

        return fig

    def _add_prediction_overlays(self, fig: go.Figure, data: pd.DataFrame,
                                 predictions: List[Dict]):
        """Add prediction markers and trend lines to chart"""

        # Find prediction points
        pred_dates = []
        pred_values = []
        pred_colors = []
        pred_text = []

        for i, pred in enumerate(predictions):
            if i < len(data) and i >= 0:  # Ensure we have a valid index
                pred_date = data.index[i]
                pred_value = data['Close'].iloc[i] if 'Close' in data.columns else 0

                pred_dates.append(pred_date)
                pred_values.append(pred_value)

                # Color based on prediction
                if pred.get('direction') == 'bullish':
                    color = self.color_bullish
                elif pred.get('direction') == 'bearish':
                    color = self.color_bearish
                else:
                    color = self.color_neutral

                pred_colors.append(color)

                # Tooltip text
                text = f"Prediction: {pred.get('direction', 'neutral').upper()}<br>"
                text += f"Confidence: {pred.get('confidence', 0):.1%}<br>"
                text += f"Prob: {pred.get('bullish_prob', 0.5):.1%}"
                pred_text.append(text)

        # Add prediction markers
        if pred_dates:
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_values,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=10,
                        color=pred_colors,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    text=pred_text,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

    def create_performance_gauge(self, confidence: float, direction: str) -> go.Figure:
        """
        Create confidence gauge chart

        Args:
            confidence: Confidence score (0-1)
            direction: Prediction direction

        Returns:
            Gauge chart figure
        """
        if direction == 'bullish':
            color = self.color_bullish
            title = 'Bullish Confidence'
        elif direction == 'bearish':
            color = self.color_bearish
            title = 'Bearish Confidence'
        else:
            color = self.color_neutral
            title = 'Neutral Confidence'

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgray'},
                    {'range': [33, 66], 'color': 'gray'},
                    {'range': [66, 100], 'color': 'darkgray'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))

        fig.update_layout(
            height=250,
            margin=dict(t=50, b=10, l=10, r=10)
        )

        return fig

    def create_model_comparison(self, model_insights: Dict) -> go.Figure:
        """Create model performance comparison chart"""

        models = list(model_insights.get('model_performance', {}).keys())
        accuracies = []

        for model in models:
            perf = model_insights['model_performance'].get(model, {})
            accuracies.append(perf.get('accuracy', 0) * 100)

        if not models:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(text="No model data available",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=accuracies,
                text=[f'{acc:.1f}%' for acc in accuracies],
                textposition='auto',
                marker_color=[self.color_primary, self.color_bullish]
            )
        ])

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy (%)',
            yaxis_range=[0, 100],
            template='plotly_white',
            height=300
        )

        return fig

    def create_feature_importance_chart(self, features: List[str],
                                        importances: List[float]) -> go.Figure:
        """Create feature importance horizontal bar chart"""

        if not features or not importances:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(text="No feature importance data available",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        # Sort by importance
        sorted_idx = np.argsort(importances)[-20:]  # Top 20 features
        sorted_features = [features[i] for i in sorted_idx if i < len(features)]
        sorted_importances = [importances[i] for i in sorted_idx if i < len(importances)]

        if not sorted_features:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(text="No valid features available",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        fig = go.Figure(data=[
            go.Bar(
                y=sorted_features,
                x=sorted_importances,
                orientation='h',
                marker_color=self.color_primary,
                opacity=0.7
            )
        ])

        fig.update_layout(
            title='Top Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white',
            height=400
        )

        return fig
